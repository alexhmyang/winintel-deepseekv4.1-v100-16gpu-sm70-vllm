# Author: 数萌AI服务器 (winintel)
"""Text pipeline boundary contracts for the pinned reference architecture."""

from dataclasses import dataclass

import torch


@dataclass
class Boundary:
    hidden: torch.Tensor
    pre_mix: torch.Tensor
    compress_kv: torch.Tensor | None
    index_k: torch.Tensor | None
    topk_idxs: torch.Tensor | None
    candidates: torch.Tensor | None
    engram_hashes: torch.Tensor | None
    main_hiddens: tuple[torch.Tensor, ...]

    def copy_to(self, device):
        """Own transported tensors; never retain aliases to a sender's caches."""

        def transfer(value):
            return None if value is None else value.to(device).clone()

        return Boundary(
            transfer(self.hidden),
            transfer(self.pre_mix),
            transfer(self.compress_kv),
            transfer(self.index_k),
            transfer(self.topk_idxs),
            transfer(self.candidates),
            transfer(self.engram_hashes),
            tuple(transfer(value) for value in self.main_hiddens),
        )

    def install_attention(self, runtime):
        for name in ("compress_kv", "index_k", "topk_idxs", "candidates"):
            setattr(runtime, name, getattr(self, name))


def capture_boundary(
    official, args, next_layer, hidden, pre_mix, hashes, targets, start_pos
):
    """Trim persistent source caches to valid rows for the next consumer.

    Source layers retain their complete caches locally. A receiver only needs
    the prefix valid for this forward; its next source overwrites these slots.
    """
    shared = official.shared_attn
    end_pos = start_pos + hidden.size(1)
    previous_sources = [layer for layer in args.kv_source_layers if layer < next_layer]
    ratio = args.compress_ratios[previous_sources[-1]] if previous_sources else 0
    valid = end_pos // ratio if ratio else 0
    batch = hidden.size(0)

    def prefix(value):
        return None if value is None else value[:batch, :valid]

    return Boundary(
        hidden,
        pre_mix,
        prefix(shared.compress_kv),
        prefix(shared.index_k),
        shared.topk_idxs,
        shared.candidates,
        hashes,
        tuple(targets),
    )


@torch.inference_mode()
def text_logits(
    official, model, args, tokens, stage_sizes, start_pos=0, transport=None
):
    """Check staged execution with supplied layers, without distributed serving.

    Transport receives a boundary and destination stage index. This reference
    harness intentionally uses an already built model, not rank-local loading.
    """
    if (
        not stage_sizes
        or any(not isinstance(size, int) or size < 1 for size in stage_sizes)
        or sum(stage_sizes) != len(model.layers)
    ):
        raise ValueError("Stages must partition all backbone layers")
    if model.vision is not None:
        raise ValueError("Pipeline harness currently supports text models only")
    # A fresh request must not export stale pointers before its first source.
    official.shared_attn = official.SharedAttentionRuntime()
    hashes = (
        model.engram_hash(tokens, start_pos) if model.engram_hash is not None else None
    )
    hidden = model.embed(tokens).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre_mix = official.make_identity_pre_mix(hidden, model.hc_mult)
    targets = []
    offset = 0
    for stage, size in enumerate(stage_sizes):
        for layer_id in range(offset, offset + size):
            layer = model.layers[layer_id]
            if layer.engram is not None:
                hidden = layer.engram(
                    hidden, hashes[:, :, layer.engram.layer_hash_index, :]
                )
            if layer_id in model.target_layer_ids:
                targets.append(hidden.mean(dim=2))
            hidden, pre_mix = layer(hidden, start_pos, pre_mix, None)
        offset += size
        if offset < len(model.layers):
            boundary = capture_boundary(
                official, args, offset, hidden, pre_mix, hashes, targets, start_pos
            )
            boundary = (
                transport(boundary, stage + 1)
                if transport is not None
                else boundary.copy_to(hidden.device)
            )
            boundary.install_attention(official.shared_attn)
            hidden, pre_mix = boundary.hidden, boundary.pre_mix
            hashes, targets = boundary.engram_hashes, list(boundary.main_hiddens)
    hidden = model.layers[-1].hc_pre(hidden, pre_mix)
    logits = model.head(model.norm(hidden), full_logits=True)
    main_hidden = torch.cat(targets, dim=-1) if targets else None
    return logits, main_hidden
