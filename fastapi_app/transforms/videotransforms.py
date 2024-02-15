from typing import Tuple

import torch
import torch.nn as nn
from pytorchvideo.transforms import Normalize, ShortSideScale, UniformTemporalSubsample
from torchvision.transforms import (
    CenterCrop,
    Compose,
    ConvertImageDtype,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)


class ConvertTCHWtoCTHW(nn.Module):
    """Convert tensor from (T, C, H, W) to (C, T, H, W)."""

    def forward(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Permute the tensor.

        Args:
            video_tensor (torch.Tensor): tensor to permute.

        Returns:
            torch.Tensor: Result tensor.
        """
        return video_tensor.permute(1, 0, 2, 3)


class VideoTransform:
    """Constructs a transform for training or evaluation."""

    def __init__(
        self,
        clip_len: int,
        mode: str,
        resize_size: int,
        crop_size: int,
        mean: Tuple[float, float, float] = (0.43216, 0.394666, 0.37645),
        std: Tuple[float, float, float] = (0.22803, 0.22145, 0.216989),
        hflip_prob: float = 0.5,
    ) -> None:
        """Initializes transform.

        Args:
            clip_len (int): Number of frames per video.
            mode (str): Train or val mode.
            resize_size (int): the short side for scaling the height and width.
            crop_size (int): the size to crop.
            mean (Tuple[float, float, float]): Defaults (0.43216, 0.394666, 0.37645).
            std (Tuple[float, float, float]): Defaults (0.43216, 0.394666, 0.37645).
            hflip_prob (float): Random flip prob in training. Defaults to 0.5.

        """
        # Pytorchvideo transforms expect CTHW instead of TCHW
        transform_list = [
            ConvertImageDtype(torch.float32),
            ConvertTCHWtoCTHW(),
            UniformTemporalSubsample(clip_len),
            ShortSideScale(size=resize_size),
            Normalize(mean, std),
        ]
        if mode == "train":
            if hflip_prob > 0:
                transform_list.append(RandomHorizontalFlip(p=hflip_prob))
            transform_list.append(RandomCrop(crop_size))
        if mode == "test":
            transform_list.append(CenterCrop(crop_size))

        self.transforms = Compose(transform_list)

    def __call__(self, x):
        """Use video transform.

        Args:
            x (torch.Tensor): the tensor to transform.

        Returns:
            torch.Tensor: transformed tensor.
        """
        return self.transforms(x)


class InverseNormalize:
    """Reverts normalization for plotting purposes."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.43216, 0.394666, 0.37645),
        std: Tuple[float, float, float] = (0.22803, 0.22145, 0.216989),
    ) -> None:
        """Given original mean and std, make reverse normalize transform.

        Args:
            mean (Tuple[float, float, float]): Defaults to
                (0.43216, 0.394666, 0.37645).
            std (Tuple[float, float, float]): Defaults to
                (0.22803, 0.22145, 0.216989).
        """
        inv_mean = (-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2])
        inv_std = (1 / std[0], 1 / std[1], 1 / std[2])
        self.transform = Compose(
            [Normalize(inv_mean, inv_std), Lambda(lambda x: x * 255.0)],
        )

    def __call__(self, x):
        """Use inverse normalize.

        Args:
            x (torch.Tensor): the tensor to transform.

        Returns:
            torch.Tensor: transformed tensor.
        """
        return self.transform(x)
