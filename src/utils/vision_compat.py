from __future__ import annotations

import math
import types
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from torchvision import transforms as _tv_transforms  # type: ignore
    from torchvision import utils as _tv_utils  # type: ignore

    transforms = _tv_transforms
    utils = _tv_utils
except Exception:
    class _Identity:
        def __call__(self, x):
            return x

    class _Compose:
        def __init__(self, transforms_list: Sequence):
            self.transforms_list = list(transforms_list)

        def __call__(self, value):
            for transform in self.transforms_list:
                value = transform(value)
            return value

    class _ToTensor:
        def __call__(self, value):
            if isinstance(value, torch.Tensor):
                tensor = value.float()
            else:
                array = np.asarray(value)
                if array.ndim == 2:
                    array = array[None, ...]
                elif array.ndim == 3 and array.shape[-1] in (1, 3, 4):
                    array = np.moveaxis(array, -1, 0)
                elif array.ndim != 3:
                    raise ValueError(f'Unsupported input shape for ToTensor: {array.shape}')
                tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
            return tensor

    class _RandomVerticalFlip:
        def __call__(self, value):
            if torch.rand(1).item() < 0.5:
                if isinstance(value, torch.Tensor):
                    return torch.flip(value, dims=[-2])
                return np.flip(np.asarray(value), axis=-2).copy()
            return value

    class _RandomAffine:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __call__(self, value):
            return value

    class _Resize:
        def __init__(self, size, antialias=True):
            if isinstance(size, int):
                self.size = (size, size)
            else:
                self.size = tuple(size)
            self.antialias = antialias

        def __call__(self, value):
            if not isinstance(value, torch.Tensor):
                value = torch.from_numpy(np.asarray(value)).float()
                if value.ndim == 2:
                    value = value.unsqueeze(0)
            if value.ndim != 3:
                return value
            resized = F.interpolate(value.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)
            return resized.squeeze(0)

    def _make_grid(tensor: torch.Tensor, nrow: int = 8, padding: int = 2) -> torch.Tensor:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f'Expected a 4D tensor, got {tensor.shape}')

        tensor = tensor.detach().cpu().float()
        num_images, channels, height, width = tensor.shape
        nrow = max(1, min(nrow, num_images))
        ncol = int(math.ceil(num_images / nrow))

        grid_height = ncol * height + padding * (ncol - 1)
        grid_width = nrow * width + padding * (nrow - 1)
        grid = torch.zeros(channels, grid_height, grid_width, dtype=tensor.dtype)

        index = 0
        for row in range(ncol):
            for col in range(nrow):
                if index >= num_images:
                    break
                top = row * (height + padding)
                left = col * (width + padding)
                grid[:, top:top + height, left:left + width] = tensor[index]
                index += 1
        return grid

    def _save_image(tensor, filename, nrow: int = 8):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        grid = _make_grid(tensor, nrow=nrow)
        grid = grid.clone()
        if grid.min() < 0 or grid.max() > 1:
            grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
        grid = (grid * 255.0).clamp(0, 255).byte()
        if grid.shape[0] == 1:
            image = Image.fromarray(grid[0].numpy(), mode='L')
        else:
            image = Image.fromarray(np.transpose(grid.numpy(), (1, 2, 0)))
        image.save(filename)

    transforms = types.SimpleNamespace(
        Compose=_Compose,
        ToTensor=_ToTensor,
        RandomVerticalFlip=_RandomVerticalFlip,
        RandomAffine=_RandomAffine,
        Resize=_Resize,
        Identity=_Identity,
    )
    utils = types.SimpleNamespace(save_image=_save_image)
