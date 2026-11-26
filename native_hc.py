# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Compute hyper-connection coefficients from FP32 residual statistics."""

from types import MethodType

import torch
import torch.nn.functional as F
from kernel import hc_split_sinkhorn


def install(official, block):
    def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        flat = x.flatten(2).float()
        normalized = flat * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.norm_eps
        )
        mixes = F.linear(normalized, hc_fn.float())
        return hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult,
            self.hc_sinkhorn_iters, self.hc_eps,
        )

    block.hc_mixes = MethodType(hc_mixes, block)
