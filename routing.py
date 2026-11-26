# Author: 数萌AI服务器 (winintel)
"""Group decode routes with one host transfer, preserving expert addition order."""

from types import MethodType

import torch


def install(official, moe):
    if getattr(moe, "_winintel_grouped_decode_routes", False):
        return
    original = moe.forward

    def forward(self, x, image_mask=None):
        shape = x.size()
        if x.numel() // self.dim > 16:
            return original(x, image_mask)
        x = x.view(-1, self.dim)
        weights, indices = self.gate(
            x, None if image_mask is None else image_mask.flatten()
        )
        groups = {}
        for row, routes in enumerate(indices.cpu().tolist()):
            for slot, expert in enumerate(routes):
                if self.experts_start_idx <= expert < self.experts_end_idx:
                    groups.setdefault(expert, []).append((row, slot))
        ordered = sorted(groups)
        positions = [position for expert in ordered for position in groups[expert]]
        selections = (
            torch.tensor(positions, dtype=torch.int64, device=x.device).T
            if positions
            else None
        )
        y = torch.zeros_like(x, dtype=torch.float32)
        offset = 0
        for expert_id in ordered:
            count = len(groups[expert_id])
            rows, slots = selections[:, offset : offset + count]
            y[rows] += self.experts[expert_id](x[rows], weights[rows, slots, None])
            offset += count
        if official.world_size > 1:
            official.dist.all_reduce(y)
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)

    moe.forward = MethodType(forward, moe)
    moe._winintel_grouped_decode_routes = True
