###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2024
###########################################################################

import logging
import os
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from utils.utils import (
    flatten_dict,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)
from pytorch_lightning import Trainer, seed_everything
import MinkowskiEngine as ME
import torch
torch.multiprocessing.set_sharing_strategy('file_system')


def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        print("EXPERIMENT ALREADY EXIST")
        cfg["trainer"][
            "resume_from_checkpoint"
        ] = f"{cfg.general.save_dir}/last-epoch.ckpt"

    for log in cfg.logging:
        print(log)
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(
            flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        )

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def train(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    callbacks = []
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))
    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    callbacks.append(RegularCheckpointing())
    print(cfg.general.gpus, 'gpus')
    runner = Trainer(
        logger=loggers,
        callbacks=callbacks,
        weights_save_path=str(cfg.general.save_dir), strategy='ddp', accelerator="gpu", devices=cfg.general.gpus, num_nodes=1,
        **cfg.trainer,
    )
    runner.fit(model)


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def test(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    runner = Trainer(
        logger=loggers,
        weights_save_path=str(cfg.general.save_dir), strategy='ddp', accelerator="gpu", devices=cfg.general.gpus, num_nodes=1,
        **cfg.trainer,
    )
    runner.test(model)


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    if cfg["general"]["train_mode"]:
        train(cfg)
    else:
        test(cfg)


if __name__ == "__main__":
    main()
