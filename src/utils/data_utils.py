# Copyright 2020 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mesh data utilities."""
import matplotlib.pyplot as plt
import numpy as np
from six.moves import range
import torch
import torchvision.transforms as T

import yaml
from munch import munchify
import cv2

import shapely
import rasterio as rst
from shapely.geometry import mapping
import random
from PIL import Image


def poly_to_idx(im_bounds, poly, resolution=512, return_with_z=False):
    """
    Returns image coordinates given bounding box geospatial bounds
    and geospatial polygon

    im_bounds: Bounding box limits of the image queried
    poly: polygon-shape to be mapped onto the image
    """

    xmin, ymin, xmax, ymax = im_bounds

    # params: west, south, east, north -> ymin, xmax, ymax, xmin
    aff = rst.transform.from_bounds(xmin, ymin, xmax, ymax, resolution, resolution)

    orig_poly = poly
    try:
        X = np.array(mapping(orig_poly)["coordinates"]).squeeze()
    except:
        raw_cds = mapping(orig_poly)["coordinates"]
        getlen = list(map(lambda x: len(x), raw_cds))
        maxidx = np.argmax(getlen)
        X = np.array(raw_cds[maxidx])

    X_orig = X

    if X.shape[-1] > 2:
        X = X[:, :2]

    if type(orig_poly) == shapely.geometry.point.Point:

        xcds, ycds = rst.transform.rowcol(
            aff,
            X[0],
            X[1],
        )
        poly = np.array([xcds, ycds])

        if return_with_z:
            poly = np.array([xcds, ycds, X_orig[:, 3]])

    elif type(orig_poly) == shapely.geometry.polygon.Polygon:
        if len(X.shape) > 2:
            X = np.concatenate(X, 0)  # flatten the polygon
        xcds, ycds = rst.transform.rowcol(
            aff,
            X[:, 0],
            X[:, 1],
        )

        poly = np.array(list(zip(xcds, ycds)))
        if return_with_z:
            poly = np.array(list(zip(xcds, ycds, X_orig[:, 2])))

    return poly


def cycle_starting_coordinate(sequence):
    # Ensure the sequence has an even number of elements
    assert len(sequence) % 2 == 0, "Sequence must have an even number of elements."

    num_coords = len(sequence) // 2
    random_start = random.randint(0, num_coords - 1)

    # Move the starting coordinate to the first position while preserving the order
    cycled_sequence = torch.cat(
        [sequence[random_start * 2 :], sequence[: random_start * 2]]
    )

    if torch.equal(cycled_sequence[:2], cycled_sequence[-2:]):
        cycled_sequence = cycled_sequence[:-2]
    else:
        # If the first and last coordinates are not the same, append the first coordinate to the end
        cycled_sequence = torch.cat([cycled_sequence, cycled_sequence[:2]])

    # Check and remove any intermediate duplicate coordinates
    i = 2
    while (
        i < len(cycled_sequence) - 2
    ):  # Ensure we don't compare the last coordinate, which is intentionally repeated
        if torch.equal(cycled_sequence[i : i + 2], cycled_sequence[i - 2 : i]):
            cycled_sequence = torch.cat([cycled_sequence[:i], cycled_sequence[i + 2 :]])
        else:
            i += 2

    return cycled_sequence


def collate_fn_multipolygon(
    batch,
    pad_token=2,
    eos_token=1,
    sos_token=0,
    global_stop_token=3,
    num_spc_tokens=4,
    max_seq_len = None,
    subset=None,
    random_shuffle=False,
    cycle_start_token=False,
    **kwargs
):
    """
    Pads batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.

    args:
      batch: list of dictionaries
      pad_token: value to pad with
      grouped_by_im: if True, assumes batch is grouped by image, otherwise by annotation
      subset: Keys by which to only select part of the batch
      random_shuffle: If True, shuffles the sequence before adding the global stopping token
      cycle_start_token: If True, cycles the starting coordinate to another random position in the sequence
      max_seq_len: Maximum sequence length to pad to (if None, pads to the longest sequence in the batch)
    """

    bdict = {
        "image": [],
        "img_mask": [],
        "vertices": [],
        "vertices_flat": [],
        "vertices_flat_mask": [],
        "num_vertices": [],
        "id": [],
        "pcls": [],
        "annotation_id": [],
        "image_id": [],
        "area": [],
        "metadata": [],
    }
    
    def fix_seq_length(tensor, max_seq_len, pad_value):
        seq_len, batch_size = tensor.shape[0], tensor.shape[1]
        if seq_len > max_seq_len:
            return tensor[:max_seq_len, :]
        elif seq_len < max_seq_len:
            pad_tensor = torch.full(
                (max_seq_len - seq_len, batch_size),
                pad_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            return torch.cat([tensor, pad_tensor], dim=0)
        return tensor

    def pad_seq(
        batch, key, with_padding=True, random_shuffle=False, cycle_start_token=False
    ):

        if with_padding:
            ## Concat global stopping token to the end of each sequence (per image)
            batch = [
                torch.cat(
                    [
                        torch.cat(t[1][key]).type(torch.int64),
                        torch.tensor([3], dtype=torch.int64),
                    ]
                )
                for t in batch
            ]

        batch = torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=False, padding_value=pad_token
        )

        return batch

    def obj_tensor(input_tensor):
        output_tensor = torch.zeros_like(input_tensor)
        counter = 0

        final_end_token = torch.where(input_tensor == 1)[0][-1]

        for i, value in enumerate(input_tensor):
            output_tensor[i] = counter + 1
            if value == 1:
                counter += 1
                output_tensor[i] = counter + 1

        output_tensor[final_end_token:] = 0

        return output_tensor

    bdict["image"] = torch.cat([b["image"].unsqueeze(0) for b in batch], 0)

    # Flatten and add special tokens to vertices
    def get_flat_seqs(batch):

        closed_regions = batch["vtx_list"]
        flat_seq = []
        seq_label = []
        for labels, r in enumerate(closed_regions):
            r = r + num_spc_tokens  # Add special tokens before flattening
            obj_seq = list(np.concatenate([r.flatten(), np.array([eos_token])]))
            flat_seq += obj_seq
            seq_label += [labels] * len(
                obj_seq
            )  # labels for each polygon in image + eos token

        vertices_flat = np.array(flat_seq).astype(np.int64)
        vertices_flat = np.insert(vertices_flat, 0, sos_token)
        seq_label = (
            np.array(seq_label) + num_spc_tokens
        )  # Add special tokens such that they are not confused with the object embeddings
        seq_label = np.insert(seq_label, 0, pad_token)  # Ignore start token
        return vertices_flat, seq_label

    bdict["vertices_flat"] = torch.nn.utils.rnn.pad_sequence(
        [
            torch.cat(
                [
                    torch.tensor(get_flat_seqs(b)[0], dtype=torch.int64),
                    torch.tensor([global_stop_token]),
                ]
            )
            for b in batch
        ],
        batch_first=False,
        padding_value=pad_token,
    )

    if cycle_start_token:
        for b in batch:
            new_verts = []
            for seq in b["vertices_flat"]:
                cycled_seq = cycle_starting_coordinate(seq[:-1])  # Exclude eos_token
                new_verts.append(torch.cat([cycled_seq, torch.tensor([eos_token])]))
            b["vertices_flat"] = new_verts
            
    if max_seq_len is not None:
        bdict["vertices_flat"] = fix_seq_length(bdict["vertices_flat"], max_seq_len, pad_token)


    if random_shuffle:
        for b in batch:
            random.shuffle(b["vertices_flat"])

    bdict["vert_obj_embeds"] = torch.nn.utils.rnn.pad_sequence(
        [
            torch.cat(
                [
                    torch.tensor(get_flat_seqs(b)[1], dtype=torch.int64),
                    torch.tensor([global_stop_token]),
                ]
            )
            for b in batch
        ],
        batch_first=False,
        padding_value=pad_token,
    )
    
    if max_seq_len is not None:
        bdict["vert_obj_embeds"] = fix_seq_length(bdict["vert_obj_embeds"], max_seq_len, pad_token)

    if random_shuffle:
        bdict["vert_obj_embds"] = []  # TODO: doesn't work with_shuffle

    if "masks" in batch[0].keys():
        bdict["masks"] = [im["masks"] for im in batch]

    ## compute mask
    mask = bdict["vertices_flat"] != pad_token

    bdict["vertices_flat_mask"] = mask
    bdict["num_vertices"] = bdict["vertices_flat_mask"].sum()
    bdict["pcls"] = ["furukawa" for x in range(len(batch))]

    bdict["vtx_list"] = [b["vtx_list"] for b in batch]
    bdict["vertices"] = bdict["vtx_list"]

    return bdict


# def view_sample_coco(batch,
#                 model,
#                 device='cuda',
#                 model_name='detr',
#                 threshold=0.5,
#                 num_samp=0,
#                 print_count_difference=False):
#     '''
#     Code taken from Peter's Kernel
#     https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train
#     '''

#     images, targets = batch

#     images = [x.to(device) for x in images]

#     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#     _,h,w = images[num_samp].shape # for de normalizing images

#     boxes = targets[num_samp]['boxes'].cpu().numpy()
#    #  print(boxes)
#     boxes = [np.array(box).astype(np.int32) for box in denormalize_bboxes(boxes,h,w)]
#     sample = np.array(T.ToPILImage()(images[num_samp]))

#     model.eval()
#     cpu_device = torch.device("cpu")

#     with torch.no_grad():
#         if model_name=='detr':
#            outputs = model(images)
#         else:
#            outputs = model(torch.stack(images))
#            outputs['pred_logits'] = outputs['logits']


#     outputs = [{k: v.to(cpu_device) for k, v in outputs.items()}][0]

#     fig, ax = plt.subplots(1, 1, figsize=(16, 8))

#     for box in boxes:
#         cv2.rectangle(sample,
#                   (box[0], box[1]),
#                   (box[2]+box[0], box[3]+box[1]),
#                   (220, 0, 0), 1)


#     oboxes = outputs['pred_boxes'][num_samp].detach().cpu().numpy()


#     oboxes = [np.array(box).astype(np.int32) for box in denormalize_bboxes(oboxes,h,w)]


#     prob = outputs['pred_logits'][num_samp].softmax(1).detach().cpu().numpy()[:,0] #positive class is 0

#     num_pred_boxes = []
#     for box,p in zip(oboxes,prob):

#         if p >threshold:
#             num_pred_boxes.append(len(box))
#             color = (0,0,220) #if p>0.5 else (0,0,0)
#             cv2.rectangle(sample,
#                   (box[0], box[1]),
#                   (box[2]+box[0], box[3]+box[1]),
#                   color, 1)

#     if print_count_difference:
#       print(f"Number of GT boxes: {len(boxes)}, number of PR boxes: {len(num_pred_boxes)}")

#     ax.set_axis_off()
#     ax.imshow(sample)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def append_images(
    images, direction="horizontal", bg_color=(255, 255, 255), aligment="center"
):
    """
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    widths, heights = zip(*(i.size for i in images))

    if direction == "horizontal":
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new("RGB", (new_width, new_height), color=bg_color)

    offset = 0
    for im in images:
        if direction == "horizontal":
            y = 0
            if aligment == "center":
                y = int((new_height - im.size[1]) / 2)
            elif aligment == "bottom":
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == "center":
                x = int((new_width - im.size[0]) / 2)
            elif aligment == "right":
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im


def read_cfg(conf_path="../config/default.yaml"):

    with open(conf_path) as f:
        yamldict = yaml.safe_load(f)

    return munchify(yamldict)


def coco_to_pascal_voc(x1, y1, w, h):
    return [x1, y1, x1 + w, y1 + h]


def convert_bbox_poly_to_pascal(box):

    xmin, miny, maxx, maxy = (
        box[:, 0].min(),
        box[:, 1].min(),
        box[:, 0].max(),
        box[:, 1].max(),
    )

    return [xmin, miny, maxx, maxy]


def rescale_boundingbox(original_dims, new_dims, bbox_bounds):
    """
    original_dims: Tuple/List of Ints
    new_dims: Tuple/List of Ints
    bbox_bounds: Tuple/List of Ints in pascal format: [xmin,ymin,xmax,ymax]
    """

    original_h, original_w = original_dims
    new_h, new_w = new_dims
    xmin, ymin, xmax, ymax = bbox_bounds

    # Calculate the scaling factor
    h_scale = new_h / original_h
    w_scale = new_w / original_w

    # Rescale the bounding box
    xmin = int(xmin * w_scale)
    ymin = int(ymin * h_scale)
    xmax = int(xmax * w_scale)
    ymax = int(ymax * h_scale)

    # Updated bounding box
    return [xmin, ymin, xmax, ymax]


def get_len_from_predseq(pred_seqs):

    seqlen = []

    for x in range(pred_seqs.shape[0]):

        if sum(pred_seqs[x, :] == 1) > 0:
            seqlen.append(
                len(pred_seqs[x, : (torch.where(pred_seqs[x, :] == 1)[0].min())])
            )
        else:
            seqlen.append(len(pred_seqs[x, :]))

    return seqlen


def get_masks(
    pred_seqs,
    plyformer,
    batch,
    return_stacked=False,
    return_gt_only=False,
    return_preds_only=False,
    device="cpu",
):

    gt_masks = []
    pred_masks = []

    batch["masks"] = batch["masks"].to(device)

    if return_gt_only:
        for samp in zip(batch["masks"], batch["metadata"]):
            gt_masks.append(
                torch.tensor(
                    invert_cropped_mask_to_origin(samp[0].numpy() > 0, samp[1])
                ).to(device)
            )
        return gt_masks
    elif return_preds_only:
        t1 = plyformer.decoder.convert_sequence_to_matrix(pred_seqs)
        t2 = [plyformer.metrics.comp_one_mask(x.cpu().numpy()) for x in t1]

        for samp in zip(t2, batch["metadata"]):
            pred_masks.append(
                invert_cropped_mask_to_origin(samp[0].squeeze().numpy(), samp[1]) > 0
            ).to(device)

        return pred_masks

    t1 = plyformer.decoder.convert_sequence_to_matrix(pred_seqs)
    t2 = [plyformer.metrics.comp_one_mask(x.cpu().numpy()) for x in t1]

    for samp in zip(batch["masks"], batch["metadata"]):
        gt_masks.append(invert_cropped_mask_to_origin(samp[0].numpy() > 0, samp[1]))

    for samp in zip(t2, batch["metadata"]):
        pred_masks.append(
            invert_cropped_mask_to_origin(samp[0].squeeze().numpy(), samp[1]) > 0
        )

    gt_masks_tensor = torch.stack([torch.tensor(x).to(device) for x in gt_masks])
    pred_masks_tensor = torch.stack([torch.tensor(x).to(device) for x in pred_masks])

    if return_stacked:
        return gt_masks_tensor, pred_masks_tensor

    return gt_masks, pred_masks


def invert_cropped_mask_to_origin(img_mask, metadata, resize_im_size=224):
    """
    Takes a cropped representation of a mask and converts it back
    into the original input space (max_dim is 300,300)
    """

    orig_mask = np.zeros((300, 300))

    img_mask = Image.fromarray(img_mask).resize(
        (metadata["orig_mask_size"][0], metadata["orig_mask_size"][1])
    )

    pad_dim_r, pad_dim_c = metadata["padding"][0][:2]
    img_mask = np.array(img_mask)

    xmin, ymin, xmax, ymax = metadata["orig_bounds"]

    orig_mask[ymin:(ymax), xmin:(xmax)] = img_mask[:pad_dim_r, :pad_dim_c]

    if resize_im_size not in orig_mask.shape:
        orig_mask = np.array(
            Image.fromarray(orig_mask).resize((resize_im_size, resize_im_size))
        )

    return orig_mask


def convert_seq_to_matrix(sequence, sample=None):

    sequence = sequence[sample]
    stop_idx = sequence == 1
    if sum(stop_idx > 0):
        sequence = sequence[: torch.where(stop_idx)[0][0]]
        sequence = sequence.reshape(sequence.shape[0] // 2, 2)

    else:
        seq_len = sequence.shape[0]
        if seq_len % 2 != 0:
            # Take one vertex less if sequence is incomplete
            sequence = sequence[: (seq_len - 1)]
            seq_len = sequence.shape[0]

        sequence = sequence.reshape(seq_len // 2, 2)

    return sequence
