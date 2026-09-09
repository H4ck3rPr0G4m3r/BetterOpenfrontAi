"""Masked multi-head policy for the flat OpenFront observation.

SB3 default policies assume an MLP over the whole vector or an image
observation. Ours is neither: a spatial raster plus a player table plus a
few scalars, packed flat for the bridge. The extractor below splits the
vector back apart, runs a small CNN over the raster and an MLP over the
table plus globals, and concatenates the two streams. Five masked
categorical heads sit on top, one per action head in bridge order, via
sb3-contrib Maskable distributions. Mask splitting reuses
``openfront_gym.spaces`` so Python and TypeScript cannot drift on order.
"""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from openfront_gym.adapter import (
    NUM_GLOBALS,
    OBSERVATION_SIZE,
    RASTER_HEIGHT,
    RASTER_PLANES,
    RASTER_SIZE,
    RASTER_WIDTH,
    TABLE_SIZE,
)


class OpenFrontExtractor(BaseFeaturesExtractor):
    """CNN over the raster, MLP over table plus globals, concat both."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        cnn_channels: tuple[int, ...] = (32, 64, 64),
        table_hidden: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        flat_size = int(np.prod(observation_space.shape))
        if flat_size != OBSERVATION_SIZE:
            raise ValueError(
                f"obs must hold {OBSERVATION_SIZE} floats, got {flat_size}"
            )
        raster_end = RASTER_SIZE
        self._raster_end = raster_end
        self._table_end = raster_end + TABLE_SIZE

        layers: list[nn.Module] = []
        in_ch = RASTER_PLANES
        for out_ch in cnn_channels:
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1), nn.ReLU()]
            in_ch = out_ch
        # Every layer is stride 2 with padding 1, so each spatial dim shrinks
        # by exactly half per layer for even inputs: 64x128 becomes 8x16.
        downscale = 2 ** len(cnn_channels)
        cnn_out = cnn_channels[-1] * (RASTER_HEIGHT // downscale) * (RASTER_WIDTH // downscale)
        self.cnn = nn.Sequential(*layers, nn.Flatten())
        self.table_mlp = nn.Sequential(
            nn.Linear(TABLE_SIZE + NUM_GLOBALS, table_hidden), nn.ReLU()
        )
        self.joint = nn.Sequential(
            nn.Linear(cnn_out + table_hidden, features_dim), nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        flat = observations.float()
        raster = flat[:, : self._raster_end].reshape(
            (-1, RASTER_PLANES, RASTER_HEIGHT, RASTER_WIDTH)
        )
        rest = flat[:, self._raster_end : self._table_end + NUM_GLOBALS]
        return self.joint(
            th.cat([self.cnn(raster), self.table_mlp(rest)], dim=1)
        )


class OpenFrontPolicy(MaskableActorCriticPolicy):
    """Maskable actor-critic with the OpenFront extractor by default."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", OpenFrontExtractor)
        kwargs.setdefault("features_extractor_kwargs", {})
        kwargs.setdefault("net_arch", dict(pi=[256], vf=[256]))
        kwargs.setdefault("activation_fn", nn.ReLU)
        super().__init__(*args, **kwargs)
