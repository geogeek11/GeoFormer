from pycocotools.coco import COCO
import numpy as np
import os

import matplotlib.pyplot as plt
from PIL import Image
import torch

from torchvision import transforms as T
from torch.utils.data import Dataset

import os.path

from src.utils.utils import sort_polygons
from src.utils.augmentations import Augmentations


import os
import torchvision.transforms.functional as F
import matplotlib.patches as patches


class Aicrowd(Dataset):
    def __init__(
        self,
        json_path,
        img_dir,
        output_size=224,
        rand_augs=False,
        sort_polygons=True,
        normalize_ims=True,
        custom_transforms=None,
        **kwargs
    ):
        # Initialize COCO API
        self.coco = COCO(json_path)
        self.img_dir = img_dir
        self.output_size = (output_size, output_size)
        self.rand_augs = rand_augs
        self.sort_polygons = sort_polygons
        if normalize_ims:
            self.normalizer = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalizer = None

        if self.rand_augs:
            if "small" in json_path:
                clip_output = True
            else:
                clip_output = False
            self.transforms = Augmentations(None, output_size, clip_output=clip_output)

        if (
            custom_transforms
        ):  # Function which takes both image and annotations, and returns both
            self.custom_transforms = custom_transforms
        else:
            self.custom_transforms = None

        self.ids = sorted(self.coco.getImgIds())

    def plot_sample(self, index):
        
        sample = self.__getitem__(index)
        image = sample["image"]
        annotations = sample["vtx_list"]

        
        image = F.to_pil_image(image)

        
        plt.imshow(image)
        ax = plt.gca()

        
        for annotation in annotations:
            polygon = annotation.numpy()  
            poly_patch = patches.Polygon(
                polygon, linewidth=1, edgecolor="r", facecolor="none"
            )
            ax.add_patch(poly_patch)

        plt.axis("off")
        plt.show()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")
        image = F.resize(image, self.output_size)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Scale annotations
        scale_x, scale_y = (self.output_size[0] - 1) / img_info["width"], (
            self.output_size[1] - 1
        ) / img_info["height"]
        scaled_annotations = []
        for annotation in annotations:
            if "segmentation" in annotation:
                for seg in annotation["segmentation"]:
                    seg = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    seg[:, 0] *= scale_x
                    seg[:, 1] *= scale_y
                    scaled_annotations.append(seg)

        
        image = F.to_tensor(image)
        if self.normalizer:
            image = self.normalizer(image)

        scaled_annotations = [
            torch.tensor(seg, dtype=torch.float32) for seg in scaled_annotations
        ]

        if self.rand_augs:
            image, scaled_annotations = self.transforms.get_augmentations(
                image, scaled_annotations
            )
            scaled_annotations = [x.numpy() for x in scaled_annotations]

        if self.sort_polygons:
            if isinstance(scaled_annotations[0], np.ndarray):
                scaled_annotations = sort_polygons(
                    scaled_annotations, return_indices=False, im_dim=self.output_size[0]
                )
            elif isinstance(scaled_annotations[0], torch.Tensor):
                scaled_annotations = sort_polygons(
                    [x.numpy() for x in scaled_annotations],
                    return_indices=False,
                    im_dim=self.output_size[0],
                )

        if self.custom_transforms:
            image, scaled_annotations = self.custom_transforms(
                (image, scaled_annotations)
            )

        return {
            "image": image,
            "vtx_list": scaled_annotations,
            "vertices_flat": [],  # Produced in the dataloader
            "img_id": img_id,
            "img_info": img_info,
        }
