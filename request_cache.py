# Author: 数萌AI服务器 (winintel)
"""Resize resident request caches without losing buffer specifications on OOM."""

from types import MethodType

import torch

CACHE_NAMES = {
    "window_kv_cache",
    "compress_kv_cache",
    "k_cache",
    "kv_state",
    "score_state",
}


def specifications(stage):
    if hasattr(stage, "_winintel_request_buffers"):
        return stage._winintel_request_buffers
    return [
        (child, name, tuple(value.shape), value.dtype, layer.attn.compress_ratio)
        for layer in stage.layers
        for child in layer.attn.modules()
        for name, value in child.named_buffers(recurse=False)
        if name in CACHE_NAMES
    ]


def reset(stage, batch_size, sequence_length):
    """Release old caches and use immutable templates for every allocation."""
    if not hasattr(stage, "_winintel_request_buffers"):
        stage._winintel_request_buffers = specifications(stage)
    official, args = stage.official, stage.args
    official.shared_attn = official.SharedAttentionRuntime()
    args.max_batch_size, args.max_seq_len = batch_size, sequence_length
    stage.hash_state.cache = None
    for child, name, _, _, _ in stage._winintel_request_buffers:
        child._buffers[name] = None
    if stage.device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        stage.hash_state.cache = torch.empty(
            batch_size, sequence_length, dtype=torch.int64, device=stage.device
        )
        frequencies = {}
        for layer in stage.layers:
            attention = layer.attn
            ratio = attention.compress_ratio
            key = bool(ratio)
            if key not in frequencies:
                frequencies[key] = official.precompute_freqs_cis.__wrapped__(
                    args.rope_head_dim,
                    sequence_length,
                    args.original_seq_len if ratio else 0,
                    args.compress_rope_theta if ratio else args.rope_theta,
                    args.rope_factor,
                    args.beta_fast,
                    args.beta_slow,
                ).to(stage.device)
            attention.freqs_cis = frequencies[key]
            if attention.indexer is not None:
                attention.indexer.freqs_cis = attention.freqs_cis
        for child, name, template, dtype, ratio in stage._winintel_request_buffers:
            shape = list(template)
            shape[0] = batch_size
            if name in ("compress_kv_cache", "k_cache"):
                shape[1] = sequence_length // ratio
            child._buffers[name] = torch.full(
                shape,
                -torch.inf if name == "score_state" else 0,
                dtype=dtype,
                device=stage.device,
            )
    except torch.cuda.OutOfMemoryError:
        stage.hash_state.cache = None
        for child, name, _, _, _ in stage._winintel_request_buffers:
            child._buffers[name] = None
        if stage.device.type == "cuda":
            torch.cuda.empty_cache()
        raise


def install(stage):
    """Keep loaded stages compatible with the resident benchmark cache allocator."""
    if not hasattr(stage, "_winintel_request_buffers"):
        stage._winintel_request_buffers = specifications(stage)
    stage.reset_request = MethodType(reset, stage)
