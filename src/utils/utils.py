import pickle

import torch

import os
import numpy as np
import json
import pytorch_lightning as pl
from tqdm import tqdm
from pycocotools.coco import COCO
from PIL import Image
import math
import torch.nn.functional as F
from omegaconf import OmegaConf


def exists(obj):
    return obj is not None


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_yaml_config(config_file: str):
    """Load a YAML file and return the configuration as an OmegaConf object."""
    cfg = OmegaConf.load(config_file)
    return cfg


class AggregatedPredictionWriterCallback(pl.Callback):
    def __init__(self, save_path):
        super().__init__()
        self.predictions = []
        self.save_path = save_path

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None
    ):
        self.predictions.append(outputs)

    def on_predict_end(self, trainer, pl_module):
        # Check if running in a distributed environment
        if trainer.world_size > 1:
            # Gather predictions from all GPUs to the main GPU
            gathered_predictions = [None] * trainer.world_size
            torch.distributed.all_gather_object(gathered_predictions, self.predictions)

            if trainer.global_rank == 0:
                # Flatten the list of lists
                flat_predictions = [
                    item for sublist in gathered_predictions for item in sublist
                ]
        else:
            # In a single GPU or CPU setup, just flatten the predictions
            flat_predictions = self.predictions

        if trainer.global_rank == 0 or trainer.world_size == 1:
            # Save the predictions on the main process or in a non-distributed environment
            with open(self.save_path, "wb") as f:
                pickle.dump(flat_predictions, f)
                print(f"\n\n Aggregated predictions saved to {self.save_path} \n\n")

    def get_aggregated_predictions(self):

        return self.predictions


def rotate_points(points, degrees, image_dims=None):
    """
    Rotates points by a given degree clockwise around the center of an image or the mean center of points.

    Args:
    points (np.ndarray): An array of points with shape Nx2.
    degrees (float): The angle in degrees to rotate the points.
    image_dims (tuple or None): Dimensions of the associated image (height, width).
                                If provided, uses the image center for rotation.

    Returns:
    np.ndarray: Rotated points array.
    """
    # Convert degrees to radians
    radians = math.radians(degrees)

    # Clockwise rotation matrix
    rotation_matrix = np.array(
        [
            [math.cos(radians), math.sin(radians)],
            [-math.sin(radians), math.cos(radians)],
        ]
    )

    # Finding the rotation center
    if image_dims is not None:
        center = np.array(
            [image_dims[1] / 2, image_dims[0] / 2]
        )  # Center is (width/2, height/2)
    else:
        center_x, center_y = np.mean(points, axis=0)
        center = np.array([center_x, center_y])

    # Translate points to center around the rotation center
    translated_points = points - center

    # Rotate points
    rotated_translated_points = np.dot(translated_points, rotation_matrix.T)

    # Translate points back to original position
    rotated_points = rotated_translated_points + center

    return rotated_points


def rotate_image_and_annotation(point_data, image_tensor, degrees):
    """
    Rotates both point_data (BxNx2) and image_tensor (BxCxHxW) by a given degree.
    Converts point_data to a tensor if it's not already.

    Args:
    point_data: Data for points. Could be a tensor of shape BxNx2 or a list-like structure.
    image_tensor (torch.Tensor): Tensor of images with shape BxCxHxW.
    degrees (float): The angle in degrees to rotate the tensor.

    Returns:
    torch.Tensor, torch.Tensor: Rotated point_tensor and image_tensor.
    """
    # Check if point_data is not a tensor, convert it to a tensor
    if not isinstance(point_data, torch.Tensor):
        point_tensor = torch.tensor(point_data, dtype=torch.float32)
    else:
        point_tensor = point_data

    # Convert degrees to radians
    radians = math.radians(degrees)

    # Rotation matrix for BxNx2 points
    rotation_matrix = torch.tensor(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ],
        dtype=torch.float32,
    )

    # Find the center of the image (assuming all images in the batch have the same dimensions)
    _, _, H, W = image_tensor.shape
    image_center = torch.tensor([W / 2, H / 2], dtype=torch.float32)

    # Translate points to center around the image center
    translated_points = point_tensor - image_center

    # Rotate points
    rotated_translated_points = torch.matmul(translated_points, rotation_matrix)

    # Translate points back to original position
    rotated_points = rotated_translated_points + image_center

    # Rotate images
    B, C, H, W = image_tensor.shape
    theta = (
        torch.tensor(
            [
                [math.cos(radians), -math.sin(radians), 0],
                [math.sin(radians), math.cos(radians), 0],
            ],
            dtype=torch.float32,
        )
        .unsqueeze(0)
        .repeat(B, 1, 1)
    )

    grid = F.affine_grid(theta, image_tensor.size(), align_corners=True)
    rotated_images = F.grid_sample(image_tensor, grid, align_corners=True)

    return rotated_points, rotated_images


def pad_and_stack_tensors(tensor_list):
    # Find the maximum length
    max_length = max(t.shape[0] for t in tensor_list)

    # Pad each tensor and store original lengths
    padded_tensors = []
    original_lengths = []
    for tensor in tensor_list:
        original_length = tensor.shape[0]
        original_lengths.append(original_length)

        # Pad the tensor
        pad_size = max_length - original_length
        padded_tensor = torch.nn.functional.pad(
            tensor, (0, 0, 0, pad_size), "constant", 0
        )
        padded_tensors.append(padded_tensor)

    # Stack the padded tensors
    stacked_tensor = torch.stack(padded_tensors)

    return stacked_tensor, original_lengths


def unstack_to_original_tensors(stacked_tensor, original_lengths):
    # Restore the original list of tensors
    original_tensors = []
    for i, length in enumerate(original_lengths):
        original_tensor = stacked_tensor[i, :length, :]
        original_tensors.append(original_tensor)

    return original_tensors


def pad_and_stack_arrays(array_list):
    # Find the maximum length
    max_length = max(arr.shape[0] for arr in array_list)

    # Pad each array and store original lengths
    padded_arrays = []
    original_lengths = []
    for arr in array_list:
        original_length = arr.shape[0]
        original_lengths.append(original_length)

        # Pad the array
        pad_size = max_length - original_length
        padded_array = np.pad(
            arr, ((0, pad_size), (0, 0)), mode="constant", constant_values=0
        )
        padded_arrays.append(padded_array)

    # Stack the padded arrays
    stacked_array = np.stack(padded_arrays)

    return stacked_array, original_lengths


def unstack_to_original_arrays(stacked_array, original_lengths):
    # Restore the original list of arrays
    original_arrays = []
    for i, length in enumerate(original_lengths):
        original_array = stacked_array[i, :length, :]
        original_arrays.append(original_array)

    return original_arrays


def compute_mean_std(image_folder):
    pixel_nums = []
    means = []
    stds = []

    for img_filename in tqdm(os.listdir(image_folder)):
        if img_filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(image_folder, img_filename)
            img = np.array(Image.open(img_path)) / 255.0
            pixel_nums.append(img.size)
            means.append(np.mean(img, axis=(0, 1)))
            stds.append(np.std(img, axis=(0, 1)))

    total_pixels = sum(pixel_nums)
    overall_mean = sum([m * n for m, n in zip(means, pixel_nums)]) / total_pixels
    overall_std = np.sqrt(
        sum([((s**2) * n) for s, n in zip(stds, pixel_nums)]) / total_pixels
    )

    return overall_mean, overall_std


def normalize_image(image, mean, std):
    normalized_image = (image - mean) / std
    return normalized_image


def inverse_normalize_image(normalized_image, mean, std):
    original_image = (normalized_image * std) + mean
    return original_image


def torch_normalize_batch(batch_images, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    normalized_images = (batch_images - mean) / std
    return normalized_images


def torch_inverse_normalize_batch(normalized_images, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    original_images = normalized_images * std + mean
    return original_images


def sort_coco_images_by_annotation_length(coco_json_path, output_json_path):
    # Load the COCO JSON file
    coco = COCO(coco_json_path)

    # Calculate the sum of annotation lengths for each image
    annotation_lengths = {
        image_id: sum(len(ann["segmentation"][0]) for ann in coco.imgToAnns[image_id])
        for image_id in coco.imgs
    }

    # Sort images by the sum of their annotation lengths
    sorted_image_ids = sorted(
        annotation_lengths, key=annotation_lengths.get, reverse=True
    )

    # Reorder the images in coco dataset
    coco.dataset["images"] = sorted(
        coco.dataset["images"], key=lambda x: sorted_image_ids.index(x["id"])
    )

    # Save the sorted COCO data
    with open(output_json_path, "w") as f:
        json.dump(coco.dataset, f, indent=4)


def sort_polygons(polygons, return_indices=False, im_dim=224):
    """
    Sorts a list of polygons counterclockwise based on their distance to the centroid of a 224x224 square.
    Optionally returns the indices to sort another list with corresponding properties.

    Parameters
    ----------
    polygons : list
        A list of polygons, each polygon is represented as an array of Nx2 points.
    return_indices : bool
        Flag to control the return type. If True, return indices. If False, return sorted polygons.

    Returns
    -------
    sorted_polygons : list or indices : list
        The sorted list of polygons or the sorting indices, depending on return_indices flag.
    """
    square_centroid = np.array([im_dim, im_dim])

    def polygon_centroid(polygon):
        return np.mean(polygon, axis=0)

    def angle_to_centroid(polygon_centroid):
        delta = polygon_centroid - square_centroid
        angle = np.arctan2(delta[1], delta[0])
        return angle if angle >= 0 else angle + 2 * np.pi

    # Pair each polygon with its index and angle
    indexed_polygons_with_angle = [
        (index, angle_to_centroid(polygon_centroid(polygon)))
        for index, polygon in enumerate(polygons)
    ]

    # Sort indices by angle
    sorted_indices_with_angle = sorted(
        indexed_polygons_with_angle, key=lambda x: x[1], reverse=True
    )

    sorted_polygons = [polygons[index] for index, angle in sorted_indices_with_angle]

    # If return_indices is True, return the sorted indices
    if return_indices:
        sorted_indices = [index for index, angle in sorted_indices_with_angle]
        return sorted_polygons, sorted_indices

    return sorted_polygons
