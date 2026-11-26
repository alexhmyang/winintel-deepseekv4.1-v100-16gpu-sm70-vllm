# Author: 数萌AI服务器 (winintel)
"""Keep large hyper-connection residuals outside the FP16 compute range."""

from types import MethodType

import torch


def install(official, block):
    """Normalize in FP32 before narrowing sublayer inputs to FP16."""

    def hc_post(self, x, residual, post, comb):
        return official.Block.hc_post(self, x.float(), residual.float(), post, comb)

    def normalize(self, x):
        return official.RMSNorm.forward(self, x).to(torch.float16)

    block.hc_post = MethodType(hc_post, block)
    block.attn_norm.forward = MethodType(normalize, block.attn_norm)
    block.ffn_norm.forward = MethodType(normalize, block.ffn_norm)
