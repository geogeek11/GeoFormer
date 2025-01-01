import torchvision.transforms as T
import cv2
import numpy as np

import torch
import torchmetrics.functional as tmfunctional
from pycocotools import mask as maskUtils
from src.utils.utils import rotate_points


def prep_sample_for_geoformer(
    image,
    annotations,
    rotate_degrees=90,
    scale_factor=2,
    pertubation="downsample",
    device="cuda",
):

    image = image
    if len(image.shape) < 4:
        _, h, w = image.shape
    else:
        b, _, h, w = image.shape

    if pertubation == "dropout":

        image_tensor = T.RandomErasing(
            p=1, scale=(0.03 * scale_factor, 0.03 * scale_factor)
        )(image)

    elif pertubation == "downsample":
        image_tensor = T.Resize((h, w))(
            T.Resize((h // scale_factor, w // scale_factor))(image)
        )

    elif pertubation == "rotation":
        image_tensor = T.functional.rotate(image, angle=rotate_degrees)
        annotations = [
            rotate_points(x, degrees=rotate_degrees, image_dims=(h, w)).clip(0, h - 1)
            for x in annotations
        ]
    else:
        image_tensor = image_tensor

    return image_tensor, annotations


def bounding_box(points):
    """returns a list containing the bottom left and the top right
    points in the sequence
    Here, we traverse the collection of points only once,
    to find the min and max for x and y
    """
    bot_left_x, bot_left_y = float("inf"), float("inf")
    top_right_x, top_right_y = float("-inf"), float("-inf")
    for x, y in points:
        bot_left_x = min(bot_left_x, x)
        bot_left_y = min(bot_left_y, y)
        top_right_x = max(top_right_x, x)
        top_right_y = max(top_right_y, y)

    return [bot_left_x, bot_left_y, top_right_x - bot_left_x, top_right_y - bot_left_y]


def compute_poly_dependent_metrics(gts, dts, gt_masks, pred_masks):

    gt_bboxs = [bounding_box(gt) for gt in gts]
    dt_bboxs = [bounding_box(dt) for dt in dts]
    gt_polygons = [gt for gt in gts]
    dt_polygons = [dt for dt in dts]

    # IoU match
    iscrowd = [0] * len(gt_bboxs)
    ious = maskUtils.iou(dt_bboxs, gt_bboxs, iscrowd)
    if len(ious) == 0:
        return 0, 0  # No overlap

    # compute polis
    img_polis_avg = 0
    biou_avg = []
    num_sample = 0
    for i, gt_poly in enumerate(gt_polygons):
        try:
            matched_idx = np.argmax(ious[:, i])
        except Exception as e:
            print(f"ERORR OCCURRED AT {i}th image, {e}, ious shape: {ious}, i: {i}")
            continue

        matched_idx = np.argmax(ious[:, i])
        iou = ious[matched_idx, i]
        if iou > 0.5:  # iouThres:
            polis = compute_POLiS(gt_poly, dt_polygons[matched_idx])
            biou_avg.append(
                boundary_iou(
                    gt=gt_masks[i], dt=pred_masks[matched_idx], dilation_ratio=0.02
                )
            )

            img_polis_avg += polis
            num_sample += 1
        else:
            biou_avg.append(0)

    pls = img_polis_avg / (num_sample + 1e-9)
    bAP = np.nanmean(biou_avg)

    return pls, bAP


def compute_POLiS(a, b):
    """
    Computes the PoLiS distance between to polygons (Avbelj, et al. 2015)
    given by their N or M points in two dimensions

    :param a: Array of points in Nx2
    :param b: Array of points in Mx2
    """

    p1, p2 = 0, 0

    q, r = a.shape[0], b.shape[0]

    for i in range(q):
        p1 += np.min(np.linalg.norm(a[i, :] - b, axis=-1))

    for j in range(r):
        p2 += np.min(np.linalg.norm(b[j, :] - a, axis=-1))

    return p1 / (2 * a.shape[0]) + p2 / (2 * b.shape[0])


def compute_aux_metrics(pred_dict, gt_dict, pred_seqs_list, gt_verts_batch):

    iou = tmfunctional.jaccard_index(
        preds=pred_dict["masks"].sum(0) > 0,
        target=gt_dict["masks"].sum(0) > 0,
        task="binary",
    )

    gt_masks, pred_masks = gt_dict["masks"].cpu().numpy().astype(np.uint8), pred_dict[
        "masks"
    ].cpu().numpy().astype(np.uint8)

    len_pred_seqs = np.array(list(map(len, pred_seqs_list))).sum()
    len_gt_seqs = np.array(list(map(len, gt_verts_batch))).sum()

    N_A, N_B = len_pred_seqs, len_gt_seqs

    ciou = iou.cpu() * np.mean(
        1
        - np.absolute(np.array(N_A)[None,] - np.array(N_B)[None,])
        / (np.array(N_A)[None,] + np.array(N_B)[None,])
    )

    n_ratio = np.mean(len_pred_seqs / len_gt_seqs)

    ## Computing polygon dependent metrics
    polis, bAP = compute_poly_dependent_metrics(
        gts=gt_verts_batch, dts=pred_seqs_list, gt_masks=gt_masks, pred_masks=pred_masks
    )

    return {
        "iou": iou.item(),  # Convert to Python scalar if it's a tensor
        "ciou": (
            ciou.item() if isinstance(ciou, torch.Tensor) else ciou
        ),  # Convert to Python scalar if it's a tensor
        "nratio": n_ratio,
        "polis": polis,
        "bAP": bAP,
    }


# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    # Taken from GitHub repo: https://github.com/bowenc0221/boundary-iou-api
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h**2 + w**2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    # Taken from GitHub repo: https://github.com/bowenc0221/boundary-iou-api
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / (union + 1e-9)
    return boundary_iou
