# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Bounded FP8 compatibility execution using existing PyTorch matrix operations."""

import os
from collections import OrderedDict

import torch

_decoded_weights = OrderedDict()


def cache_statistics(device):
    """Describe resident decoded weights and graphs on one CUDA device."""
    entries = [
        entry for entry in _decoded_weights.values() if entry[0].device == device
    ]
    return {
        "resident_matrices": len(entries),
        "decoded_bytes": sum(entry[3].numel() * 4 for entry in entries),
        "captured_graphs": sum(
            "graph" in entry[5].get("graph", {}) for entry in entries
        ),
        "budget_mib": int(os.environ.get("WININTEL_FP8_CACHE_MIB", "256")),
    }


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


def _cached_matrix(weight, scales, block_size):
    """Cache immutable FP32 weights within a per-device byte budget.

    Capture decodes into graph-owned storage instead of retaining cache entries.
    Inference tensors without mutation counters use the uncached path.
    Keep admitted weights resident so a model larger than the budget cannot
    repeatedly evict and decode every matrix on each token.
    """
    limit = int(os.environ.get("WININTEL_FP8_CACHE_MIB", "256")) * 1024**2
    device = weight.device
    size = weight.numel() * 4
    versions = (_version(weight), _version(scales))
    key = id(weight)
    used = sum(
        v[3].numel() * 4 for v in _decoded_weights.values() if v[0].device == device
    )
    for old_key, old in list(_decoded_weights.items()):
        if old[0].device == device and used > max(limit, 0):
            used -= old[3].numel() * 4
            del _decoded_weights[old_key]
    if limit <= 0 or size > limit or None in versions:
        return None
    from kernel import _scale_float

    if key not in _decoded_weights and used + size > limit:
        return None
    if torch.cuda.is_current_stream_capturing():
        rows = torch.arange(weight.shape[0], device=device) // block_size
        return weight.float() * _scale_float(scales)[rows].repeat_interleave(
            block_size, -1
        )
    entry = _decoded_weights.pop(key, None)
    if (
        entry is not None
        and entry[0] is weight
        and entry[1] is scales
        and entry[2] == versions
        and entry[5]["block"] == block_size
    ):
        torch.cuda.current_stream(device).wait_event(entry[4])
        entry[3].record_stream(torch.cuda.current_stream(device))
        _decoded_weights[key] = entry
        return entry[3]
    rows = torch.arange(weight.shape[0], device=device) // block_size
    matrix = weight.float() * _scale_float(scales)[rows].repeat_interleave(
        block_size, -1
    )
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(device))
    _decoded_weights[key] = (
        weight,
        scales,
        versions,
        matrix,
        ready,
        {"block": block_size},
    )
    return matrix


def _graph_matmul(a, a_s, matrix, block_size, state, quantization=None):
    """Replay activation conversion and GEMM with a cache-owned static matrix."""
    from kernel import _activation, act_quant

    key = (
        a.shape,
        a.dtype,
        None if a_s is None else a_s.shape,
        None if a_s is None else a_s.dtype,
        block_size,
        torch.get_default_dtype(),
        quantization,
    )
    stream = torch.cuda.current_stream(a.device)
    if "last_use" in state:
        stream.wait_event(state["last_use"])
    if state.get("key") != key:
        state.clear()
        with torch.inference_mode(False):
            activation = a.clone()
            scales = None if a_s is None else a_s.clone()

        def compute():
            encoded, factors = (
                (activation, scales)
                if quantization is None
                else act_quant(activation, block_size, *quantization)
            )
            x = _activation(encoded, factors, block_size).reshape(-1, a.shape[-1])
            return (
                (x @ matrix.T)
                .reshape(*a.shape[:-1], matrix.shape[0])
                .to(torch.get_default_dtype())
            )

        warmup = torch.cuda.Stream(device=a.device)
        warmup.wait_stream(stream)
        with torch.cuda.stream(warmup):
            compute()
        stream.wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            output = compute()
        state.update(
            key=key, activation=activation, scales=scales, graph=graph, output=output
        )
    else:
        state["activation"].copy_(a)
        if a_s is not None:
            state["scales"].copy_(a_s)
    state["graph"].replay()
    result = state["output"].clone()
    for name in ("activation", "scales", "output"):
        if state[name] is not None:
            state[name].record_stream(stream)
    event = torch.cuda.Event()
    event.record(stream)
    state["last_use"] = event
    return result


def install_fp8_linear(official):
    """Capture the official input rounding together with cached FP8 GEMM."""
    if getattr(official, "_winintel_fp8_linear_graph", False):
        return
    original = official.linear

    def linear(x, weight, bias=None):
        import qwen_adapter

        block = getattr(weight, "_winintel_q8", None)
        if (
            bias is not None
            or block is None
            or not x.is_cuda
            or x.numel() // x.shape[-1] > 16
            or not qwen_adapter._cuda_graph_enabled
            or os.environ.get("WININTEL_FP8_GRAPH", "1") != "1"
            or torch.cuda.is_current_stream_capturing()
        ):
            return original(x, weight, bias)
        if (
            block != official.fp8_block_size
            or weight._winintel_q8_source_scales is not weight.scale
        ):
            return original(x, weight, bias)
        matrix = _cached_matrix(weight, weight.scale, block)
        if matrix is None:
            return original(x, weight, bias)
        return _graph_matmul(
            x,
            None,
            matrix,
            block,
            _decoded_weights[id(weight)][5].setdefault("graph", {}),
            (official.scale_fmt, official.scale_dtype),
        )

    official.linear = linear
    official._winintel_fp8_linear_graph = True


def enabled():
    return os.environ.get("WININTEL_NATIVE_SM70", "1") == "1"


def prepare_fp8(weight, scales):
    """Validate block scales while retaining checkpoint weights in E4M3."""
    from kernel import _scale_float

    if weight.ndim != 2 or not weight.is_cuda or not weight.is_contiguous():
        raise ValueError("Supply a contiguous CUDA FP8 weight matrix")
    if weight.dtype != torch.float8_e4m3fn or scales.device != weight.device:
        raise ValueError("FP8 storage and scale devices must match")
    n, k = weight.shape
    block = k // scales.shape[-1] if scales.ndim == 2 and scales.shape[-1] else 0
    if (
        block not in (32, 128)
        or k % block
        or scales.shape != ((n + block - 1) // block, k // block)
    ):
        raise ValueError("FP8 weight scales must cover N-by-K blocks")
    decoded = _scale_float(scales).contiguous()
    if not (torch.isfinite(decoded) & (decoded > 0)).all().item():
        raise ValueError("FP8 weight scales must be finite and positive")
    weight._winintel_q8 = block
    weight._winintel_q8_source_scales = scales


def fp8_matmul(a, a_s, weight, scales, block_size):
    from kernel import _activation, _scale_float

    if weight._winintel_q8_source_scales is not scales:
        raise ValueError("Reprepare native FP8 after replacing scales")
    prepared_block = weight._winintel_q8
    if block_size != prepared_block:
        raise ValueError("Activation and prepared weight blocks must match")
    matrix = _cached_matrix(weight, scales, block_size)
    if matrix is not None:
        import qwen_adapter

        if (
            qwen_adapter._cuda_graph_enabled
            and os.environ.get("WININTEL_FP8_GRAPH", "1") == "1"
            and a.numel() // a.shape[-1] <= 16
            and not torch.cuda.is_current_stream_capturing()
        ):
            entry = _decoded_weights[id(weight)]
            return _graph_matmul(
                a, a_s, matrix, block_size, entry[5].setdefault("graph", {})
            )
        x = _activation(a, a_s, block_size).reshape(-1, a.shape[-1])
        return (
            (x @ matrix.T)
            .reshape(*a.shape[:-1], weight.shape[0])
            .to(torch.get_default_dtype())
        )
    x = _activation(a, a_s, block_size).reshape(-1, a.shape[-1])
    decoded = _scale_float(scales)
    result = torch.empty(
        (x.shape[0], weight.shape[0]),
        dtype=torch.get_default_dtype(),
        device=a.device,
    )
    for begin in range(0, weight.shape[0], 256):
        end = min(begin + 256, weight.shape[0])
        rows = torch.arange(begin, end, device=weight.device) // block_size
        expanded = decoded[rows].repeat_interleave(block_size, -1)
        matrix = weight[begin:end].float() * expanded
        result[:, begin:end] = x @ matrix.T
    return result.reshape(*a.shape[:-1], weight.shape[0]).to(torch.get_default_dtype())
