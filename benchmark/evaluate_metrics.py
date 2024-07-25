###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2024
###########################################################################

import numpy as np
from utils.votenet_utils.metric_util import calc_iou  # axis-aligned 3D box IoU


def get_iou(bb1, bb2):
    """Compute IoU of two bounding boxes.
    ** Define your bod IoU function HERE **
    """
    # pass
    iou3d = calc_iou(bb1, bb2)
    return iou3d


def eval_ref_one_sample(pred_bbox, gt_bbox):
    """ Evaluate one grounding prediction

    Args:
        pred_bbox: 8 corners of prediction bounding box, (8, 3)
        gt_bbox: 8 corners of ground truth bounding box, (8, 3)
    Returns:
        iou: intersection over union score
    """

    iou = get_aabb3d_iou(pred_bbox.detach().cpu().numpy(), gt_bbox.detach().cpu().numpy())

    return iou


def get_aabb3d_min_max(corner):
    ''' Compute min and max coordinates for 3D bounding box
        Note: only for axis-aligned bounding boxes

    Input:
        corners: numpy array (8,3), assume up direction is Z (batch of N samples)
    Output:
        box_min_max: an array for min and max coordinates of 3D bounding box IoU

    '''
    min_coord = corner.min(axis=0)
    max_coord = corner.max(axis=0)
    x_min, x_max = min_coord[0], max_coord[0]
    y_min, y_max = min_coord[1], max_coord[1]
    z_min, z_max = min_coord[2], max_coord[2]

    return x_min, x_max, y_min, y_max, z_min, z_max


def get_aabb3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is Z
        corners2: numpy array (8,3), assume up direction is Z
    Output:
        iou: 3D bounding box IoU

    '''
    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_aabb3d_min_max(corners1)
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_aabb3d_min_max(corners2)
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)

    return iou


def eval_grounding(
    preds, gts, masks
):
    print("evaluating grounding", len(preds), "scans...")
    ious = []
    multiple = []
    for pred_bbox, gt_bbox, mask in zip(preds, gts, masks):
        # compute the iou
        if pred_bbox == []:
            iou = 0.0
        else:
            iou = get_iou(pred_bbox[0][1], gt_bbox[0][1])
        ious.append(iou)
        multiple.append(mask[0])

    ious = np.array(ious)
    multiple = np.array(multiple)
    return ious, multiple


def eval_mIoU(
    preds: dict,  gt_path: str, output_file: str, dataset: str = "scannet"
):
    print("evaluating mIoU", len(preds), "scans...")
    accum_I = 0.
    accum_U = 0.
    accum_IoU = 0.
    pr_count_50 = 0.
    pr_count_25 = 0.
    count = 0.
    ious = []

    for v in preds:
        pred_mask, target_mask = v['pred_masks'], v['target_masks']
        pred_mask = np.squeeze(pred_mask, axis=-1)
        target_mask = np.array(target_mask[0], dtype=np.int8)
        I, U = computeIoU(pred_mask, target_mask)
        this_iou = float(0) if U == 0 else float(I) / float(U)
        ious.append(this_iou)
        accum_IoU += this_iou
        accum_I += I
        accum_U += U
        if this_iou >= 0.25:
            pr_count_25 += 1
        if this_iou >= 0.5:
            pr_count_50 += 1
        count += 1

    res = {}
    res['mIoU'] = 100. * (accum_IoU / count)
    res['cIoU'] = accum_I * 100. / accum_U
    res['P0.25'] = pr_count_25 * 100. / count
    res['P0.50'] = pr_count_50 * 100. / count

    print(res)
    ious = np.array(ious)
    return res, ious


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U