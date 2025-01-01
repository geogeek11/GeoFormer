from kornia import augmentation as K
import torch
from src.pipelines.inference import ensure_closed_loop_torch
import numpy as np


class Augmentations:
    def __init__(self, cfg, image_size, clip_output=False):
        self.cfg = cfg

        self.transform = K.AugmentationSequential(
            K.RandomRotation(degrees=[-270, 270], p=0.5),
            K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "keypoints"],
            same_on_batch=False,
        )

        self.kpt_min = 0
        self.kpt_max = image_size - 1
        self.clip_output = clip_output

    def pad_and_stack_tensors(self, tensor_list):
        # Find the maximum length
        max_length = max(t.shape[0] for t in tensor_list)

        # Pad each tensor and store original lengths
        padded_tensors = []
        original_lengths = []
        for tensor in tensor_list:
            original_length = tensor.shape[0]
            original_lengths.append(original_length)

            pad_size = max_length - original_length
            padded_tensor = torch.nn.functional.pad(
                tensor, (0, 0, 0, pad_size), "constant", 0
            )
            padded_tensors.append(padded_tensor)

        stacked_tensor = torch.stack(padded_tensors)

        return stacked_tensor, original_lengths

    def unstack_to_original_tensors(self, stacked_tensor, original_lengths):
        # Restore the original list of tensors
        original_tensors = []
        for i, length in enumerate(original_lengths):
            original_tensor = stacked_tensor[i, :length, :]
            original_tensors.append(original_tensor)

        return original_tensors

    def get_augmentations(self, image, keypoints):

        if isinstance(keypoints[0], np.ndarray):
            keypoints = [torch.from_numpy(x).float() for x in keypoints]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float().permute(2, 0, 1)

        orig_image, orig_keypoints = image, keypoints

        stacked_verts, orig_lengths = self.pad_and_stack_tensors(keypoints)

        imgout, keypoints = self.transform(image, stacked_verts)

        if not self.clip_output:

            if (
                torch.min(keypoints) < self.kpt_min
                or torch.max(keypoints) > self.kpt_max
            ):
                # print('RESAMPLING')
                iter = 0
                while torch.min(keypoints) < 0 or torch.max(keypoints) > self.kpt_max:
                    imgout, keypoints = self.transform(image.cpu(), stacked_verts)
                    iter += 1
                    if iter > 5:
                        return orig_image, orig_keypoints
        elif self.clip_output:
            ## Allows for non square images
            keypoints = torch.clamp(keypoints, self.kpt_min, self.kpt_max)
            keypoints = self.unstack_to_original_tensors(keypoints, orig_lengths)
            keypoints = [ensure_closed_loop_torch(k) for k in keypoints]
            return imgout[0, ::], keypoints

        keypoints = self.unstack_to_original_tensors(keypoints, orig_lengths)

        return imgout[0, ::], keypoints
