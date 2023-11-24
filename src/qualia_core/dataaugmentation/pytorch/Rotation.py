from __future__ import annotations

import logging
import sys

import torch

from .DataAugmentationPyTorch import DataAugmentationPyTorch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class Rotation(DataAugmentationPyTorch):

    def __init__(self,  # noqa: PLR0913
                 dimensions: int=3,
                 sigma: float=0.375,
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True) -> None:  # noqa: FBT001, FBT002
        super().__init__(evaluate=evaluate, before=before, after=after)

        if dimensions != 3:  # noqa: PLR2004
            logger.error('Only 3D rotations are supported, dimensions=%s', dimensions)
            raise ValueError
        self.dimensions = dimensions
        self.sigma = sigma

        self.angledist = None

    def apply(self, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        import pytorch3d.transforms

        if self.angledist is None: # Lazy init so that CUDA is only used when module is actually called
            self.angledist = torch.distributions.normal.Normal(torch.tensor(0.0, device=device),
                                                               torch.tensor(self.sigma, device=device))

        angle = self.angledist.sample((3, ))

        rotmat = pytorch3d.transforms.axis_angle_to_matrix(angle)

        transform = pytorch3d.transforms.Rotate(rotmat, device=device)

        # For some reason pytorch3d transforms work with 3D channels last
        x = torch.swapaxes(x, 1, 2)

        rotated_x_array = tuple(transform.transform_points(x[:, :, i:i+self.dimensions])
                                    if x.shape[-1] >= i + self.dimensions else x[:, :, i:]
                                    for i in range(0, x.shape[-1], self.dimensions))
        rotated_x = torch.cat(rotated_x_array, dim=-1)

        rotated_x = torch.swapaxes(rotated_x, 1, 2)

        return rotated_x, y

    @override
    def __call__(self, data: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(*data, device=device)
