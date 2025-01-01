from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from torchmetrics.detection import MeanAveragePrecision
import datetime
import pickle
from src.utils.utils import load_yaml_config
from src.utils.metrics import compute_aux_metrics
from src.models.xtransformer import init_models, XTransformerTrainer


def ensure_closed_loop(matrix):
    """
    Ensure that the final row in the matrix is the same as the first row.
    If not, append the first row to the matrix.
    """
    if not (matrix[0] == matrix[-1]).all():
        matrix = np.vstack([matrix, matrix[0]])
    return matrix


def ensure_closed_loop_torch(matrix):
    """
    Ensure that the final row in the matrix is the same as the first row.
    If not, append the first row to the matrix.
    Args:
        matrix (torch.Tensor): A 2D tensor where rows represent points.
    Returns:
        torch.Tensor: The modified tensor ensuring a closed loop.
    """
    if not torch.equal(matrix[0], matrix[-1]):
        matrix = torch.cat((matrix, matrix[0].unsqueeze(0)), dim=0)
    return matrix


def save_inferred_samples(
    sample_array,
    output_path,
    weight_name,
    ds_name,
):
    timestamp = str(datetime.datetime.now().date())

    with open(
        f"{output_path}/{weight_name}_{timestamp}_{ds_name}_inference_samples.pkl", "wb"
    ) as file:
        pickle.dump(sample_array, file)

    return sample_array


def load_model_from_chkpt(out_ckpt_dir, model_id, use_latest=False, cfg=None, **kwargs):
    """
    Loads the XTransformer model based on the given parameters.

    :param out_ckpt_dir: The directory containing the output checkpoints
    :param model_id: The unique identifier of the model
    :param use_latest: Whether to use the latest checkpoint or the one with the lowest validation loss. Defaults to False.
    :return: The loaded XTransformer model
    """

    if not cfg:
        config_path = os.path.join(out_ckpt_dir, model_id, f"{model_id}.yaml")
        cfg = load_yaml_config(config_path)
    else:
        print("Using provided config")

    enc, dec = init_models(cfg)
    model = XTransformerTrainer(
        enc,
        dec,
        encoder_backbone=cfg.model.transformer_xformer.backbone,
        global_context=cfg.model.transformer_xformer.get("custom_embeddings", False),
    )

    checkpoint_dir = os.path.join(out_ckpt_dir, model_id)
    if use_latest:
        checkpoint_path = os.path.join(checkpoint_dir, "last.ckpt")
    else:
        checkpoint_files = [
            f
            for f in os.listdir(checkpoint_dir)
            if f.endswith(".ckpt") and f != "last.ckpt"
        ]
        val_losses = [
            float(f.split("=")[-1].split(".ckpt")[0])
            for f in checkpoint_files
            if "val_loss" in f
        ]
        min_loss_file = checkpoint_files[val_losses.index(min(val_losses))]
        checkpoint_path = os.path.join(checkpoint_dir, min_loss_file)

    try:
        model = XTransformerTrainer.load_from_checkpoint(
            encoder=enc,
            decoder=dec,
            checkpoint_path=checkpoint_path,
            encoder_backbone=cfg.model.transformer_xformer.backbone,
        )
        print(f"Loaded model from checkpoint {checkpoint_path}")
    except:
        print(f"Could not load model from checkpoint {checkpoint_path}")
        print("Attempting alternative")
        weights = torch.load(checkpoint_path)
        model = model.load_state_dict(weights["state_dict"])
        model.encoder_backbone = cfg.model.transformer_xformer.backbone

    return model, cfg


def compute_map(out_array, image_size=224, compute_by_object=True, device="cpu"):
    metric = MeanAveragePrecision(
        iou_type="segm",
        class_metrics=False,
        extended_summary=True,
    )

    aux_metrics = []

    for i, d in tqdm(enumerate(out_array), total=len(out_array)):
        gen_samples = d["gen_samples"]
        gt_vert_list = d["gt_verts"]

        for batch_num in range(len(gt_vert_list)):
            gt_verts_batch = gt_vert_list[batch_num]
            gt_mask = [poly_to_mask(x, image_size=image_size) for x in gt_verts_batch]
            gt_masks = np.stack(gt_mask)

            pred_seqs_list = conv_multiple_genseq_to_matrix(gen_samples[batch_num, :])
            pred_seqs_list = [x - 4 for x in pred_seqs_list]
            pred_seqs_list = [
                ensure_closed_loop(matrix).clip(0, image_size - 1)
                for matrix in pred_seqs_list
                if len(matrix) > 1
            ]
            pred_masks_list = [
                poly_to_mask(x, image_size=image_size) for x in pred_seqs_list
            ]

            pred_masks = (
                np.stack(pred_masks_list)
                if len(pred_masks_list) > 0
                else np.zeros((1, image_size, image_size))
            )

            if compute_by_object:
                gt_dict = {
                    "masks": torch.tensor(gt_masks, dtype=torch.bool).to(device),
                    "labels": torch.tensor([1] * len(gt_masks), dtype=torch.long).to(
                        device
                    ),
                }
                pred_dict = {
                    "masks": torch.tensor(pred_masks, dtype=torch.bool).to(device),
                    "labels": torch.tensor([1] * len(pred_masks), dtype=torch.long).to(
                        device
                    ),
                    "scores": torch.tensor([1.0] * len(pred_masks)).to(device),
                }
            else:
                gt_dict = {
                    "masks": torch.tensor(
                        gt_masks.sum(0) > 0, dtype=torch.bool
                    ).unsqueeze(0),
                    "labels": torch.tensor([1.0]).long(),
                }
                pred_dict = {
                    "masks": torch.tensor(
                        pred_masks.sum(0) > 0, dtype=torch.bool
                    ).unsqueeze(0),
                    "labels": torch.tensor([1.0]).long(),
                    "scores": torch.tensor([1.0]),
                }

            metric.update([pred_dict], [gt_dict])

            aux_metrics.append(
                compute_aux_metrics(
                    pred_dict=pred_dict,
                    gt_dict=gt_dict,
                    pred_seqs_list=pred_seqs_list,
                    gt_verts_batch=gt_verts_batch,
                )
            )

    mAP = metric.compute()

    ## Output towards latex table
    selected_values = {key: mAP[key] for key in ["map", "map_50", "map_75"]}
    pred_df = pd.DataFrame([selected_values])
    pred_df = pred_df.astype("float32")

    # Dimensions assumed here are (iou, category, area range, maxDets), TxRxARxMD
    pred_df["mAR"] = mAP["recall"][
        :, 0, -1, -1
    ].mean()  # Averaging over all IoU thresholds, categories, and area sizes for maxDets=100
    pred_df["ar50"] = mAP["recall"][0, 0, -1, -1].mean()  # IoU = 0.5, last maxDets
    pred_df["ar75"] = mAP["recall"][5, 0, -1, -1].mean()  # IoU = 0.75, last maxDets

    ## auxillary metrics
    aux_metrics_df = pd.DataFrame(aux_metrics)
    aux_metrics_df = aux_metrics_df.mean()
    pred_df["bAP"] = aux_metrics_df["bAP"]
    pred_df["ciou"] = aux_metrics_df["ciou"]
    pred_df["iou"] = aux_metrics_df["iou"]
    pred_df["nratio"] = aux_metrics_df["nratio"]
    pred_df["polis"] = aux_metrics_df["polis"]

    return pred_df


def poly_to_mask(vertices, image_size):

    img = np.zeros((image_size, image_size), np.uint8)
    cv2.fillPoly(img, [vertices.astype(np.int32)], 1)
    # convert image to binary mask
    mask = np.clip(img, 0, 1).astype(np.uint8)
    return mask


def conv_multiple_genseq_to_matrix(genseq, eos_token=1, global_stop_token=3):

    stop_indices = torch.where(genseq == eos_token)[0].tolist()

    # Add end of sequence to handle the last segment
    if len(genseq) not in stop_indices:
        stop_indices.append(len(genseq))

    segments = []
    start_idx = 0
    for stop_idx in stop_indices:
        subsamp = genseq[start_idx:stop_idx]

        # Ensure even length
        if len(subsamp) % 2 != 0:
            subsamp = subsamp[:-1]

        # Reshape to matrix
        try:
            vtxes = subsamp.reshape(len(subsamp) // 2, 2).cpu().numpy()
        except RuntimeError:
            continue  # If we can't reshape, skip to the next segment

        segments.append(vtxes)
        start_idx = stop_idx + 1  # Move to the next segment starting point

    only_2_and_3 = np.all((segments[-1] == 2) | (segments[-1] == 3))

    if only_2_and_3:  # Drop padding and global EOS token
        segments = segments[:-1]

    return segments


def plot_sequences(
    batch, gen_samples, num_batch=0, eos_token=1, num_spc_tokens=4, ax=None
):
    """
    Plots the ground truth and predicted sequences for a given batch and sample number.
    """

    if ax is None:
        fig, ax = plt.subplots()

    if gen_samples.shape[0] < gen_samples.shape[1]:  # assuming batch-first
        genseq = gen_samples[num_batch]
    else:  # if sequence-first
        genseq = gen_samples[:, num_batch]

    pred_seq = conv_multiple_genseq_to_matrix(genseq, eos_token=eos_token)

    ax.imshow(batch["image"][num_batch].transpose(0, -1).cpu().numpy())

    # pls = []
    for num_iter, pvtx in enumerate(pred_seq):
        # subtract the special tokens
        pvtx = pvtx - num_spc_tokens

        # Plot the predicted sequence
        ax.plot(pvtx[:, 1], pvtx[:, 0], "x-")

        # Plot ground truth if available
        try:
            gt_verts = batch["vertices"][num_batch][num_iter]
            ax.plot(gt_verts[:, 1], gt_verts[:, 0], "ko--", alpha=0.5)
            # pls.append((np.round(compute_POLiS(gt_verts, pvtx)), num_iter))
        except IndexError:
            continue

        # Add other annotations (centroid and vertex labels)
        centroid = np.mean(gt_verts, axis=0)
        ax.text(
            centroid[1],
            centroid[0],
            str(num_iter),
            color="red",
            fontsize=12,
            ha="center",
            bbox=dict(facecolor="white", edgecolor="none", boxstyle="circle,pad=0.2"),
        )
        for i, vtx in enumerate(pvtx):
            if i == len(pvtx) - 1 and np.array_equal(vtx, pvtx[0]):
                continue
            ax.text(
                vtx[1],
                vtx[0],
                str(i),
                color="blue",
                fontsize=8,
                ha="center",
                bbox=dict(
                    facecolor="white", edgecolor="none", boxstyle="circle,pad=0.2"
                ),
            )

    ax.axis("off")

    return ax
