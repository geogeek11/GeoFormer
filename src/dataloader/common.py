import torch
from torch.utils.data import Subset
from src.dataloader.aicrowd import Aicrowd
import matplotlib.pyplot as plt
import wandb
import os


def dataset_loader(cfg, module_root_path=None, **kwargs):
    """
    A common dataset loader fn for input to arbitrary sequence model

    :param cfg: Hydra-core configuration file defined in ./config
    :param module_root_path: Absolute path to the root of the module repository

    :returns ds,valid_ds: Torch.utils.data.Dataset-objects for training and validation dataset
    """

    if cfg.dataset.get("aicrowd"):

        train_path_to_images = cfg.dataset.aicrowd.train_image_path
        train_path_to_json = cfg.dataset.aicrowd.train_annot_path

        valid_path_to_images = cfg.dataset.aicrowd.valid_image_path
        valid_path_to_json = cfg.dataset.aicrowd.valid_annot_path

        perform_rand_augs = cfg.dataset.aicrowd.get("enable_rand_augs", True)

        print(f"PERFORMING AUGMENTATIONS: Using RandAugs: {perform_rand_augs}")

        ds = Aicrowd(
            json_path=os.path.join(module_root_path, train_path_to_json),
            img_dir=os.path.join(module_root_path, train_path_to_images),
            rand_augs=perform_rand_augs,
            **cfg.dataset.aicrowd,
        )

        valid_ds = Aicrowd(
            json_path=os.path.join(module_root_path, valid_path_to_json),
            img_dir=os.path.join(module_root_path, valid_path_to_images),
            rand_augs=False,
            **cfg.dataset.aicrowd,
        )

        test_ds = valid_ds  # Test set equals validation set for aicrowd

        if exists(test_ds):
            return ds, valid_ds, test_ds

        return ds, valid_ds, None


def exists(obj):
    return obj is not None


def plot_samples(dataset, output_resolution, num_samples=20, batched_loader=False):

    if batched_loader:

        for s, ds in enumerate(dataset):

            vtx = ds["vertices_flat"][:, 0][ds["vertices_flat_mask"][:, 0]][
                :-1
            ]  # Account for EOS token
            vshape = vtx.shape[0]
            vtx = vtx.reshape(vshape // 2, 2)

            plt.plot(vtx[:, 0], vtx[:, 1], "o-")
            plt.xlim([0, output_resolution])
            plt.ylim([0, output_resolution])

            if s == num_samples:
                break

        wandb.log({"Raw data sample": plt})
    else:

        for s, ds in enumerate(dataset):

            vtx = ds["vertices"]  # Account for EOS token
            vshape = vtx.shape[0]
            

            plt.plot(vtx[:, 0], vtx[:, 1], "o-")
            plt.xlim([0, output_resolution])
            plt.ylim([0, output_resolution])

            if s == num_samples:
                break

        wandb.log({"Raw data sample": plt})


def get_subset_dataset(dataset, num_samples=1000):
    """
    Converts dataset into a subset 
    """

    total_samples = len(dataset)

    if total_samples < num_samples:
        subset_size = total_samples

    else:
        subset_size = num_samples

    subset_dataset = Subset(
        dataset, torch.randperm(total_samples)[:subset_size].numpy()
    )

    print(f"\n\n TRAINING ON SUBSET: {len(subset_dataset)} \n\n")

    return subset_dataset


def get_dataset_config(metadata_path):
    """
    Returns the configuration parameters based on the provided metadata path
    """
    convert_to_pixel_space, flip_bbox = False, False

    if "dk" in metadata_path:
        convert_to_pixel_space, flip_bbox = True, False
    elif "nl" in metadata_path:
        convert_to_pixel_space, flip_bbox = False, True

    return convert_to_pixel_space, flip_bbox
