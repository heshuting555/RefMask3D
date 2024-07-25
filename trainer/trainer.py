###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2024
###########################################################################

import gc
from contextlib import nullcontext
from pathlib import Path
import statistics
import itertools
import os
import math
from torch_scatter import scatter_mean
from benchmark.evaluate_metrics import eval_grounding, eval_mIoU
from collections import defaultdict
from detectron2.utils.comm import all_gather, is_main_process, synchronize
import hydra
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
from models.metrics import IoU
import random
import colorsys
from typing import List, Tuple
import functools


@functools.lru_cache(20)
def get_evenly_distributed_colors(
    count: int,
) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    # lru cache caches color tuples
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    random.shuffle(HSV_tuples)
    return list(
        map(
            lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(
                np.uint8
            ),
            HSV_tuples,
        )
    )


class RegularCheckpointing(pl.Callback):
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        general = pl_module.config.general
        trainer.save_checkpoint(f"{general.save_dir}/last-epoch.ckpt")
        print("Checkpoint created")


class InstanceSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.decoder_id = config.general.decoder_id

        if config.model.train_on_segments:
            self.mask_type = "segment_mask"
        else:
            self.mask_type = "masks"

        self.eval_on_segments = config.general.eval_on_segments

        self.config = config
        self.save_hyperparameters()
        # model
        self.model = hydra.utils.instantiate(config.model)

        self.optional_freeze = nullcontext
        if config.general.freeze_backbone:
            self.optional_freeze = torch.no_grad
        # loss
        self.ignore_label = config.data.ignore_label

        matcher = hydra.utils.instantiate(config.matcher)
        weight_dict = {
            "loss_ce": matcher.cost_class,
            "loss_mask": matcher.cost_mask,
            "loss_dice": matcher.cost_dice,
            "loss_embed": 0.3,
        }

        aux_weight_dict = {}
        for i in range(self.model.num_levels * self.model.num_decoders):
            if i not in self.config.general.ignore_mask_idx:
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict.items()}
                )
            else:
                aux_weight_dict.update(
                    {k + f"_{i}": 0.0 for k, v in weight_dict.items()}
                )
        weight_dict.update(aux_weight_dict)

        self.preds = []
        self.bbox_preds = []
        self.bbox_gt = []
        self.multiple = []
        self.criterion = hydra.utils.instantiate(
            config.loss, matcher=matcher, weight_dict=weight_dict
        )

        # metrics
        self.confusion = hydra.utils.instantiate(config.metrics)
        self.iou = IoU()
        # misc
        self.labels_info = dict()

    def forward(
        self, x, point2segment=None, raw_coordinates=None, is_eval=False, lang_emb=None, lang_mask=None
    ):
        with self.optional_freeze():
            x = self.model(
                x,
                point2segment,
                raw_coordinates=raw_coordinates,
                is_eval=is_eval,
                lang_emb=lang_emb,
                lang_mask=lang_mask,
            )
        return x

    def training_step(self, batch, batch_idx):
        data, target, file_names, input_ids, attention_mask = batch

        if data.features.shape[0] > self.config.general.max_batch_size:
            print("data exceeds threshold")
            raise RuntimeError("BATCH TOO BIG")

        if len(target) == 0:
            print("no targets")
            return None

        lang_emb = [x for x in input_ids]
        lang_emb = torch.cat(lang_emb, dim=0)

        lang_mask = [x for x in attention_mask]
        lang_mask = torch.cat(lang_mask, dim=0)

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )

        try:
            output = self.forward(
                data,
                point2segment=[
                    target[i]["point2segment"] for i in range(len(target))
                ],
                raw_coordinates=raw_coordinates,
                lang_emb=lang_emb,
                lang_mask=lang_mask,
            )
        except RuntimeError as run_err:
            print(run_err)
            if (
                "only a single point gives nans in cross-attention"
                == run_err.args[0]
            ):
                return None
            else:
                raise run_err

        contrastive_ep = self.current_epoch >= 2

        try:
            losses = self.criterion(output, target, mask_type=self.mask_type)
            #losses = self.criterion(output, target, mask_type=self.mask_type, contrastive=contrastive_ep)
        except ValueError as val_err:
            print(f"ValueError: {val_err}")
            print(f"data shape: {data.shape}")
            print(f"data feat shape:  {data.features.shape}")
            print(f"data feat nans:   {data.features.isnan().sum()}")
            print(f"output: {output}")
            print(f"target: {target}")
            print(f"filenames: {file_names}")
            raise val_err

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        logs = {
            f"train_{k}": v.detach().cpu().item() for k, v in losses.items()
        }
        if "loss_ce" in losses.keys():
            logs["train_mean_loss_ce"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_ce" in k]]
        )

        logs["train_mean_loss_mask"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_mask" in k]]
        )

        logs["train_mean_loss_dice"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_dice" in k]]
        )

        self.log_dict(logs, sync_dist=True)

        del output, lang_mask, lang_emb, batch, data, target, file_names, input_ids, attention_mask
        # torch.cuda.empty_cache()
        return sum(losses.values())


    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def export(self, pred_masks, scores, pred_classes, file_names, decoder_id):
        root_path = f"eval_output"
        base_path = f"{root_path}/instance_evaluation_{self.config.general.experiment_name}_{self.current_epoch}/decoder_{decoder_id}"
        pred_mask_path = f"{base_path}/pred_mask"

        Path(pred_mask_path).mkdir(parents=True, exist_ok=True)

        file_name = file_names
        with open(f"{base_path}/{file_name}.txt", "w") as fout:
            real_id = -1
            for instance_id in range(len(pred_classes)):
                real_id += 1
                pred_class = pred_classes[instance_id]
                score = scores[instance_id]
                mask = pred_masks[:, instance_id].astype("uint8")

                if score > self.config.general.export_threshold:
                    # reduce the export size a bit. I guess no performance difference
                    np.savetxt(
                        f"{pred_mask_path}/{file_name}_{real_id}.txt",
                        mask,
                        fmt="%d",
                    )
                    fout.write(
                        f"pred_mask/{file_name}_{real_id}.txt {pred_class} {score}\n"
                    )

    def training_epoch_end(self, outputs):
        train_loss = sum([out["loss"].cpu().item() for out in outputs]) / len(
            outputs
        )
        results = {"train_loss_mean": train_loss}
        self.log_dict(results, sync_dist=True)


    def validation_epoch_end(self, outputs):
        self.test_epoch_end(outputs)
        del outputs

    def eval_step(self, batch, batch_idx):
        data, target, file_names, input_ids, attention_mask = batch
        inverse_maps = data.inverse_maps
        target_full = data.target_full
        original_colors = data.original_colors
        data_idx = data.idx
        original_normals = data.original_normals
        original_coordinates = data.original_coordinates
        lang_emb = [x for x in input_ids]
        lang_emb = torch.cat(lang_emb, dim=0)

        lang_mask = [x for x in attention_mask]
        lang_mask = torch.cat(lang_mask, dim=0)

        if len(data.coordinates) == 0:
            return 0.0

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]

        if raw_coordinates.shape[0] == 0:
            return 0.0

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )

        try:
            output = self.forward(
                data,
                point2segment=[
                    target[i]["point2segment"] for i in range(len(target))
                ],
                raw_coordinates=raw_coordinates,
                is_eval=True,
                lang_emb=lang_emb,
                lang_mask=lang_mask,
            )


        except RuntimeError as run_err:
            print(run_err)
            if (
                "only a single point gives nans in cross-attention"
                == run_err.args[0]
            ):
                return None
            else:
                raise run_err
        # target[0]['center_label'] = output['center_label']

        if self.config.data.test_mode != "test":
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(False)

            try:
                losses = self.criterion(
                    output, target, mask_type=self.mask_type
                )
            except ValueError as val_err:
                print(f"ValueError: {val_err}")
                print(f"data shape: {data.shape}")
                print(f"data feat shape:  {data.features.shape}")
                print(f"data feat nans:   {data.features.isnan().sum()}")
                print(f"output: {output}")
                print(f"target: {target}")
                print(f"filenames: {file_names}")
                raise val_err

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(True)

        if self.config.general.save_visualizations:
            backbone_features = (
                output["backbone_features"].F.detach().cpu().numpy()
            )
            from sklearn import decomposition

            pca = decomposition.PCA(n_components=3)
            pca.fit(backbone_features)
            pca_features = pca.transform(backbone_features)
            rescaled_pca = (
                255
                * (pca_features - pca_features.min())
                / (pca_features.max() - pca_features.min())
            )
        self.eval_instance_step(
            output,
            target,
            target_full,
            inverse_maps,
            file_names,
            original_coordinates,
            original_colors,
            original_normals,
            raw_coordinates,
            data_idx,
            backbone_features=rescaled_pca
            if self.config.general.save_visualizations
            else None,
        )
        del output, batch, data
        del target, file_names, input_ids, attention_mask
        if self.config.data.test_mode != "test":
            return {
                f"val_{k}": v.detach().cpu().item() for k, v in losses.items()
            }
        else:
            return 0.0

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def get_full_res_mask(
        self, mask, inverse_map, point2segment_full, is_heatmap=False
    ):
        mask = mask.detach().cpu()[inverse_map]  # full res

        if self.eval_on_segments and is_heatmap == False:
            mask = scatter_mean(
                mask, point2segment_full, dim=0
            )  # full res segments
            mask = (mask > 0.5).float()
            mask = mask.detach().cpu()[
                point2segment_full.cpu()
            ]  # full res points

        return mask

    def get_mask_and_scores(
        self, mask_cls, mask_pred, num_queries=100, num_classes=18, device=None
    ):
        if device is None:
            device = self.device
        labels = (
            torch.arange(num_classes, device=device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )

        if self.config.general.topk_per_image != -1:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                self.config.general.topk_per_image, sorted=True
            )
        else:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                num_queries, sorted=True
            )

        labels_per_query = labels[topk_indices]
        topk_indices = topk_indices // num_classes

        # if mask_pred.shape[1] != 1:
        mask_pred = mask_pred[:, topk_indices]

        result_pred_mask = (mask_pred > 0).float()
        heatmap = mask_pred.float().sigmoid()

        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
            result_pred_mask.sum(0) + 1e-6
        )
        score = scores_per_query * mask_scores_per_image
        classes = labels_per_query

        return score, result_pred_mask, classes, heatmap

    def eval_instance_step(
        self,
        output,
        target_low_res,
        target_full_res,
        inverse_maps,
        file_names,
        full_res_coords,
        original_colors,
        original_normals,
        raw_coords,
        idx,
        first_full_res=False,
        backbone_features=None,
    ):
        label_offset = self.validation_dataset.label_offset
        prediction = output["aux_outputs"]
        prediction.append(
            {
                "pred_logits": output["pred_logits"],
                "pred_masks": output["pred_masks"],
            }
        )
        prediction[self.decoder_id]["pred_logits"] = torch.functional.F.softmax(
            prediction[self.decoder_id]["pred_logits"], dim=-1)[..., :-1]

        all_pred_classes = list()
        all_pred_masks = list()
        all_pred_scores = list()
        all_heatmaps = list()
        all_query_pos = list()
        all_target_masks = list()
        offset_coords_idx = 0
        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            if not first_full_res:
                if self.model.train_on_segments:
                    masks = (
                        prediction[self.decoder_id]["pred_masks"][bid]
                        .detach()
                        .cpu()[target_low_res[bid]["point2segment"].cpu()]
                    )
                else:
                    masks = (
                        prediction[self.decoder_id]["pred_masks"][bid]
                        .detach()
                        .cpu()
                    )

                scores, masks, classes, heatmap = self.get_mask_and_scores(
                        prediction[self.decoder_id]["pred_logits"][bid]
                        .detach()
                        .cpu(),
                        masks,
                        prediction[self.decoder_id]["pred_logits"][bid].shape[
                            0
                        ],
                        self.model.num_classes - 1,
                    )

                masks = self.get_full_res_mask(
                    masks,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                )

                heatmap = self.get_full_res_mask(
                    heatmap,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                    is_heatmap=True,
                )

                if backbone_features is not None:
                    backbone_features = self.get_full_res_mask(
                        torch.from_numpy(backbone_features),
                        inverse_maps[bid],
                        target_full_res[bid]["point2segment"],
                        is_heatmap=True,
                    )
                    backbone_features = backbone_features.numpy()
            else:
                assert False, "not tested"
                masks = self.get_full_res_mask(
                    prediction[self.decoder_id]["pred_masks"][bid].cpu(),
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                )

                scores, masks, classes, heatmap = self.get_mask_and_scores(
                    prediction[self.decoder_id]["pred_logits"][bid].cpu(),
                    masks,
                    prediction[self.decoder_id]["pred_logits"][bid].shape[0],
                    self.model.num_classes - 1,
                    device="cpu",
                )

            masks = masks.numpy()
            heatmap = heatmap.numpy()

            sort_scores = scores.sort(descending=True)
            sort_scores_index = sort_scores.indices.cpu().numpy()
            sort_scores_values = sort_scores.values.cpu().numpy()

            sort_classes = classes[sort_scores_index]

            sorted_masks = masks[:, sort_scores_index]
            sorted_heatmap = heatmap[:, sort_scores_index]

            all_pred_classes.append(sort_classes)
            all_pred_masks.append(sorted_masks)
            all_pred_scores.append(sort_scores_values)
            all_heatmaps.append(sorted_heatmap)
            all_target_masks.append(target_full_res[bid]["masks"])

        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            all_pred_classes[
                bid
            ] = self.validation_dataset._remap_model_output(
                all_pred_classes[bid].cpu() + label_offset
            )

            if (
                self.config.data.test_mode != "test"
                and len(target_full_res) != 0
            ):
                target_full_res[bid][
                    "labels"
                ] = self.validation_dataset._remap_model_output(
                    target_full_res[bid]["labels"].cpu() + label_offset
                )

                # PREDICTION BOX
                bbox_data = []
                for query_id in range(
                    all_pred_masks[bid].shape[1]
                ):  # self.model.num_queries
                    obj_coords = full_res_coords[bid][
                        all_pred_masks[bid][:, query_id].astype(bool), :
                    ]
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(
                            axis=0
                        ) - obj_coords.min(axis=0)

                        bbox = np.concatenate((obj_center, obj_axis_length))

                        bbox_data.append(
                            (
                                all_pred_classes[bid][query_id].item(),
                                bbox,
                                all_pred_scores[bid][query_id],
                            )
                        )
                self.bbox_preds.append(bbox_data)

                # GT BOX
                bbox_data = []
                multiple_data = []
                for obj_id in range(target_full_res[bid]["masks"].shape[0]):
                    if target_full_res[bid]["labels"][obj_id].item() == 255:
                        continue

                    obj_coords = full_res_coords[bid][
                        target_full_res[bid]["masks"][obj_id, :]
                        .cpu()
                        .detach()
                        .numpy()
                        .astype(bool),
                        :,
                    ]
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(
                            axis=0
                        ) - obj_coords.min(axis=0)

                        bbox = np.concatenate((obj_center, obj_axis_length))
                        bbox_data.append(
                            (
                                target_full_res[bid]["labels"][obj_id].item(),
                                bbox,
                            )
                        )
                        multiple_data.append(target_full_res[bid]["is_unique"])
                self.bbox_gt.append(bbox_data)
                del bbox_data
                self.multiple.append(multiple_data)
                del multiple_data
            if self.config.general.eval_inner_core == -1:
                self.preds.append({
                    "pred_masks": all_pred_masks[bid],
                    "pred_scores": all_pred_scores[bid],
                    "pred_classes": all_pred_classes[bid],
                    "target_masks": all_target_masks[bid],
                })
            else:
                # prev val_dataset
                self.preds.append({
                    "pred_masks": all_pred_masks[bid][
                        self.test_dataset.data[idx[bid]]["cond_inner"]
                    ],
                    "target_masks": all_target_masks[bid],
                    "pred_scores": all_pred_scores[bid],
                    "pred_classes": all_pred_classes[bid],
                })
            del all_pred_masks
            del all_pred_scores
            del all_pred_classes
            del all_target_masks

    def eval_instance_epoch_end(self, bbox_preds, bbox_gt, multiple, preds):
        log_prefix = f"val"
        ap_results = {}

        try:
            ious, masks = eval_grounding(bbox_preds, bbox_gt, multiple)
            print('box results')
            scores = self.get_result(ious, masks)
        except:
            scores = {}
            scores["overall"] = {}
            scores["overall"]["overall"] = {}
            scores["overall"]["overall"]["acc@0.25iou"] = 0.0
            scores["overall"]["overall"]["acc@0.5iou"] = 0.0
        ap_results[f"{log_prefix}_grounding_acc_25"] = scores["overall"]["overall"]["acc@0.25iou"]
        ap_results[f"{log_prefix}_grounding_acc_50"] = scores["overall"]["overall"]["acc@0.5iou"]

        root_path = f"eval_output"
        base_path = f"{root_path}/instance_evaluation_{self.config.general.experiment_name}_{self.current_epoch}"

        if self.validation_dataset.dataset_name in [
            "scannet",
            "stpls3d",
            "scannet200",
        ]:
            gt_data_path = f"{self.validation_dataset.data_dir[0]}/instance_gt/{self.validation_dataset.mode}"
        else:
            gt_data_path = f"{self.validation_dataset.data_dir[0]}/instance_gt/Area_{self.config.general.area}"

        pred_path = f"{base_path}/tmp_output.txt"

        log_prefix = f"val"

        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)

        try:
            res, ious = eval_mIoU(
                    preds,
                    gt_data_path,
                    None,
                    dataset=self.validation_dataset.dataset_name,
                )
            print('mask results')
            scores = self.get_result(ious, masks)

            ap_results[f"{log_prefix}_mIoU"] = res['mIoU']
            ap_results[f"{log_prefix}_mask_acc_50"] = res['P0.50']
            ap_results[f"{log_prefix}_mask_acc_25"] = res['P0.25']

        except (IndexError, OSError) as e:
            print('error!')

        ap_results[f"{log_prefix}_mean_ap"] = res['mIoU']
        ap_results[f"{log_prefix}_mean_ap_50"] = res['P0.50']
        ap_results[f"{log_prefix}_mean_ap_25"] = res['P0.25']
        self.log_dict(ap_results, sync_dist=True)

        del bbox_preds
        del bbox_gt
        del multiple
        del preds
        del self.preds
        del self.bbox_preds
        del self.bbox_gt
        del self.multiple
        gc.collect()

        self.preds = []
        self.bbox_preds = []
        self.bbox_gt = []
        self.multiple = []

    def test_epoch_end(self, outputs):
        if self.config.general.export:
            return

        synchronize()
        preds = all_gather(self.preds)
        preds = list(itertools.chain(*preds))
        bbox_preds = all_gather(self.bbox_preds)
        bbox_preds = list(itertools.chain(*bbox_preds))
        bbox_gt = all_gather(self.bbox_gt)
        bbox_gt = list(itertools.chain(*bbox_gt))
        multiple = all_gather(self.multiple)
        multiple = list(itertools.chain(*multiple))
        # if not is_main_process():
        #     return
        self.eval_instance_epoch_end(bbox_preds, bbox_gt, multiple, preds)
        dd = defaultdict(list)
        for output in outputs:
            for key, val in output.items():  # .items() in Python 3.
                dd[key].append(val)

        dd = {k: statistics.mean(v) for k, v in dd.items()}
        if "loss_ce" in dd.keys():
            dd["val_mean_loss_ce"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_ce" in k]]
        )
        dd["val_mean_loss_mask"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_mask" in k]]
        )
        dd["val_mean_loss_dice"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_dice" in k]]
        )

        del preds
        del bbox_gt
        del multiple
        del bbox_preds
        self.log_dict(dd, sync_dist=True)
        del self.preds
        del self.bbox_preds
        del self.bbox_gt
        del self.multiple
        gc.collect()

        self.preds = []
        self.bbox_preds = []
        self.bbox_gt = []
        self.multiple = []

    def configure_optimizers(self):

        params_to_optimize = [
            {
                "params": [p for n, p in self.named_parameters() if "text_encoder" not in n and p.requires_grad],
                "lr": self.config.optimizer.lr,
            },
            {
                "params": [p for n, p in self.named_parameters() if "text_encoder" in n and p.requires_grad],
                "lr": self.config.optimizer.lr * 0.2,
            },
        ]
        # 0.1 - 0.2
        optimizer = torch.optim.AdamW(params_to_optimize)

        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = math.ceil(len(
                self.train_dataloader()) / self.config.general.gpus)
            # self.config.scheduler.scheduler.steps_per_epoch = len(self.train_dataloader())

        lr_scheduler = hydra.utils.instantiate(
            self.config.scheduler.scheduler, optimizer=optimizer
        )
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def prepare_data(self):
        self.train_dataset = hydra.utils.instantiate(
            self.config.data.train_dataset
        )
        self.validation_dataset = hydra.utils.instantiate(
            self.config.data.validation_dataset
        )
        self.test_dataset = hydra.utils.instantiate(
            self.config.data.test_dataset
        )
        self.labels_info = self.train_dataset.label_info

    def train_dataloader(self):
        self.prepare_data()
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    def val_dataloader(self):
        self.prepare_data()
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

    def test_dataloader(self):
        self.prepare_data()
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
        )

    def get_result(self, ious, masks):
        multiple_dict = {
            "unique": 0,
            "multiple": 1
        }

        # evaluation stats
        stats = {k: np.sum(masks == v) for k, v in multiple_dict.items()}
        stats["overall"] = masks.shape[0]
        stats = {}
        for k, v in multiple_dict.items():
            stats[k] = {}
            stats[k]["overall"] = np.sum(masks == v)

        stats["overall"] = {}
        stats["overall"]["overall"] = masks.shape[0]

        scores = {}
        for k, v in multiple_dict.items():
            if k not in scores:
                scores[k] = {}
            scores[k]["overall"] = {}
            scores[k]["overall"]["acc@0.25iou"] = ious[masks == v][ious[masks == v] >= 0.25].shape[0]/ious[masks == v].shape[0] * 100.0 if ious[masks == v].shape[0] !=0 else 0.0
            scores[k]["overall"]["acc@0.5iou"] = ious[masks == v][ious[masks == v] >= 0.50].shape[0]/ious[masks == v].shape[0] * 100.0 if ious[masks == v].shape[0] !=0 else 0.0

        scores["overall"] = {}
        # aggregate
        scores["overall"]["overall"] = {}
        scores["overall"]["overall"]["acc@0.25iou"] = ious[ious >= 0.25].shape[0]/ious.shape[0] * 100.0
        scores["overall"]["overall"]["acc@0.5iou"] = ious[ious >= 0.50].shape[0]/ious.shape[0] * 100.0

        # report
        print("\nstats:")
        for k_s in stats.keys():
            for k_o in stats[k_s].keys():
                print("{} | {}: {}".format(k_s, k_o, stats[k_s][k_o]))

        for k_s in scores.keys():
            print("\n{}:".format(k_s))
            for k_m in scores[k_s].keys():
                for metric in scores[k_s][k_m].keys():
                    print("{} | {} | {}: {}".format(k_s, k_m, metric, scores[k_s][k_m][metric]))
        return scores


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.zeros_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
