import numpy as np
import os
import cv2
import shapely
from shapely import geometry
from PIL import Image
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
import os.path
import rasterio as rst

import os
from torchvision.transforms import Resize


class NationalRegistersDataset(Dataset):
    """
    Dataset loader for ERBD dataset: https://ieeexplore.ieee.org/document/10495708
    """
    
    def __init__(
        self,
        dataframe,
        image_size=224,
        start_token=1,
        eos_token=2,
        pad_token=0,
        dataset_split="train",
        num_spc_tokens=3,
        pertube_vertices=False,
        data_root_path=None,
        flip_bbox=False,
        absolute_path_column=None,
        convert_to_pixel_space=True,
        **kwargs,
    ):

        self.dataframe = dataframe
        self.pertube_vertices = pertube_vertices
        self.training = True if dataset_split == "train" else False
        self.data_root_path = data_root_path
        self.flip_bbox = flip_bbox
        self.absolute_path_column = absolute_path_column
        self.convert_to_pixel_space = convert_to_pixel_space

        self.dataframe = dataframe

        self.image_size = image_size
        self.num_unq_buildings = self.dataframe.shape[0]
        self.start_token = start_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.num_spc_tokens = 3

        print(
            f"\n\n DATASET SPLIT: {dataset_split}, contains {self.dataframe.shape[0]} polygons and {self.num_unq_buildings} buildings \n\n"
        )

    def poly_to_mask(self, vertices, image_size):

        img = np.zeros((image_size, image_size), np.uint8)
        cv2.fillPoly(img, [vertices.astype(np.int32)], 1)
        mask = np.clip(img, 0, 1).astype(np.uint8)
        return mask

    def __len__(self):
        return self.num_unq_buildings

    def random_translation_pertubation(self, vertices, max_translation_error=5):

        translation_error_x = np.random.uniform(
            -max_translation_error, max_translation_error
        )
        translation_error_y = np.random.uniform(
            -max_translation_error, max_translation_error
        )

        # Add translation errors to the vertices
        translated_vertices = vertices + np.array(
            [translation_error_x, translation_error_y]
        )

        return translated_vertices

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        if not self.data_root_path:
            if self.absolute_path_column:
                im_path = row[self.absolute_path_column]
            else:
                im_path = row["impaths_abs"]
        else:
            im_path = os.path.join(
                self.data_root_path, row["impaths_abs"].split("builds/")[-1]
            )

        # load image and resize
        image = Image.open(im_path)
        width, height = image.size
        assert width == height, "Image is not square"
        resolution = width
        image = Resize((self.image_size, self.image_size))(image)

        if self.convert_to_pixel_space:
            im_bounds = row["bbox_bounds"]

            if self.flip_bbox:
                im_bounds = [im_bounds[1], im_bounds[0], im_bounds[3], im_bounds[2]]
            # map vertices onto resized image
            geom = row["geometry"]
            vtx = poly_to_idx(im_bounds, geom, resolution=resolution)
        else:
            vtx = row["vertices"]

        vtx = np.round(vtx * self.image_size / resolution).astype(
            int
        )  # resize vertices
        vtx = vtx[:, ::-1]
        vtx = vtx + self.num_spc_tokens  # Exclude special tokens from coordinates

        out_vtx = vtx

        if self.training and self.pertube_vertices:
            out_vtx = self.random_translation_pertubation(out_vtx)

        full_vtx = np.concatenate(out_vtx)
        full_vtx = np.clip(full_vtx, 0, self.image_size + self.num_spc_tokens - 1)
        full_vtx_flat = full_vtx.flatten()

        # add start and end tokens to vertices
        full_vtx_flat = np.insert(full_vtx_flat, 0, self.start_token)
        full_vtx_flat = np.append(full_vtx_flat, self.eos_token)

        # create binary mask for vertices
        px = geometry.Polygon(out_vtx)  # Because we only have one
        vtx_ext = np.array(list(shapely.ops.unary_union(px).exterior.coords))
        mask = self.poly_to_mask(vertices=vtx_ext, image_size=self.image_size)

        sample = {
            "image": T.ToTensor()(image)[:3, ::],
            "vertices": torch.from_numpy(full_vtx).long(),
            "vertices_flat": torch.from_numpy(full_vtx_flat).long(),
            "mask": torch.from_numpy(mask).float(),
            "vtx_list": out_vtx,
            "img_id": row["impaths_abs"].split("/")[-1],
        }

        return sample


def custom_collate_fn(batch, padding_value=0):
    images = torch.stack([item["image"] for item in batch])
    vertices = [item["vertices_flat"] for item in batch]
    vertices_lengths = torch.tensor([len(v) for v in vertices])
    vertices_flat = torch.nn.utils.rnn.pad_sequence(
        vertices, batch_first=False, padding_value=padding_value
    )
    vertices_full = torch.nn.utils.rnn.pad_sequence(
        [item["vertices"] for item in batch],
        batch_first=False,
        padding_value=padding_value,
    )
    vertices_flat_mask = (
        torch.arange(vertices_flat.size(1))[None, :] < vertices_lengths[:, None]
    )
    masks = torch.stack([item["mask"] for item in batch])
    vtx_list = [item["vtx_list"] for item in batch]

    return {
        "image": images,
        "vertices_flat": vertices_flat,
        "vertices_flat_mask": vertices_flat_mask,
        "mask": masks,
        "vertices": vertices_full,
        "vtx_list": vtx_list,
    }


def poly_to_idx(im_bounds, poly, resolution=256):
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
    X = np.array(orig_poly.exterior.coords).squeeze()

    if X.shape[-1] > 2:
        X = X[:, :2]

    if type(orig_poly) == shapely.geometry.point.Point:

        xcds, ycds = rst.transform.rowcol(
            aff,
            X[0],
            X[1],
        )
        poly = np.array([xcds, ycds])

    elif type(orig_poly) == shapely.geometry.polygon.Polygon:
        xcds, ycds = rst.transform.rowcol(
            aff,
            X[:, 0],
            X[:, 1],
        )

        poly = np.array(list(zip(xcds, ycds)))

    return poly
