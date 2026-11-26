# Author: 数萌AI服务器 (winintel)
"""Adapt the local Apache-2.0 Qwen SM70 FP4 kernel to DeepSeek E8M0 groups."""

import hashlib
import importlib.util
import os
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import torch

_prepared_cache = OrderedDict()
_cuda_graph_enabled = False


def enable_cuda_graph(enabled):
    """Capture native matrix operations while keeping routing and paging eager."""
    global _cuda_graph_enabled
    _cuda_graph_enabled = enabled
    if not enabled:
        while len(_prepared_cache) > 64:
            _, evicted = _prepared_cache.popitem(last=False)
            del evicted._winintel_fp4


def cached_weight(packed, scales, limit=None):
    """Bound expanded scales while keeping all expert weights packed on device."""
    key = packed.data_ptr()
    limit = limit or (512 if _cuda_graph_enabled else 64)
    if key in _prepared_cache:
        _prepared_cache.move_to_end(key)
        prepared = _prepared_cache[key]._winintel_fp4
    else:
        prepared = getattr(packed, "_winintel_graph_fp4", None)
        if prepared is None:
            prepared = prepare_weight(packed, scales)
        packed._winintel_fp4 = prepared
        _prepared_cache[key] = packed
    if _cuda_graph_enabled:
        # Captured kernels reference these inputs outside the graph memory pool.
        packed._winintel_graph_fp4 = prepared
    while len(_prepared_cache) > limit:
        _, evicted = _prepared_cache.popitem(last=False)
        del evicted._winintel_fp4
    return prepared


@lru_cache(maxsize=1)
def load_extension():
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("The Qwen FP4 operator requires CUDA")
    source = Path(__file__).parent / "upstream" / "qwen_fp4.cu"
    expected = "721d8d0c36d572ce8c8f864f002edc7ab4cf16270c53cfb78a2f28d62cec14ce"
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Vendored Qwen FP4 source failed integrity verification")
    prebuilt = os.environ.get("WININTEL_FP4_PREBUILT")
    if prebuilt:
        path = Path(prebuilt)
        digest = os.environ.get("WININTEL_FP4_PREBUILT_SHA256")
        if not digest or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Prebuilt native FP4 extension failed verification")
        spec = importlib.util.spec_from_file_location(
            "winintel_v41_qwen_fp4_cuda", path
        )
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        return extension
    os.environ.setdefault("MAX_JOBS", "4")
    return load(
        name="winintel_v41_qwen_fp4_cuda",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
    )


class PreparedFP4:
    """Keep packed weights and expand each per-32 scale into two per-16 scales.

    Prepare after final checkpoint loading, before CUDA Graph capture.
    Reprepare if weights or scales are changed. Arithmetic rounds input,
    decoded weights and output to FP16; model-quality qualification is required.
    """

    def __init__(self, packed, scales):
        from kernel import _scale_float

        if (
            not packed.is_cuda
            or packed.ndim != 2
            or not packed.is_contiguous()
            or packed.dtype not in (torch.uint8, torch.float4_e2m1fn_x2)
            or packed.shape[1] == 0
            or packed.shape[1] % 16
        ):
            raise ValueError("Supply contiguous CUDA packed FP4 with K divisible by 32")
        if scales.device != packed.device or scales.shape != (
            packed.shape[0],
            packed.shape[1] // 16,
        ):
            raise ValueError("DeepSeek FP4 requires one scale per row and 32 weights")
        decoded = _scale_float(scales)
        if (
            not (torch.isfinite(decoded) & (decoded >= 2**-23) & (decoded <= 65504 / 6))
            .all()
            .item()
        ):
            raise ValueError("Weight scales exceed the lossless FP16 decode range")
        with torch.cuda.device(packed.device):
            self.extension = load_extension()
        self.packed = packed.view(torch.uint8)
        self.source_scales = scales
        if scales.dtype != torch.float8_e8m0fnu:
            raise ValueError("Native DeepSeek FP4 requires E8M0 scale storage")
        self.scale_ids = (
            scales.view(torch.uint8).to(torch.int16).repeat_interleave(2, dim=-1)
        )
        self.scale_table = scale_table(packed.device)
        self.graph = None
        self.graph_key = None

    @property
    def scales(self):
        """Expose decoded per-sixteen scales for diagnostics without retaining them."""
        from kernel import _scale_float

        return _scale_float(self.source_scales).repeat_interleave(2, dim=-1)

    def matmul(self, activation, scales, block_size=32):
        if not _cuda_graph_enabled or torch.cuda.is_current_stream_capturing():
            return self._matmul(activation, scales, block_size)
        key = (
            activation.shape,
            activation.dtype,
            scales.shape,
            scales.dtype,
            block_size,
            torch.get_default_dtype(),
        )
        if key != self.graph_key:
            self.graph = None
            self.graph_activation = activation.clone()
            self.graph_scales = scales.clone()
            stream = torch.cuda.Stream(device=activation.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._matmul(self.graph_activation, self.graph_scales, block_size)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.graph_output = self._matmul(
                    self.graph_activation, self.graph_scales, block_size
                )
            self.graph, self.graph_key = graph, key
        else:
            self.graph_activation.copy_(activation)
            self.graph_scales.copy_(scales)
        self.graph.replay()
        return self.graph_output.clone()

    def _matmul(self, activation, scales, block_size):
        from kernel import _activation

        if activation.device != self.packed.device:
            raise ValueError("Activation and weight devices must match")
        if activation.shape[-1] != self.packed.shape[-1] * 2:
            raise ValueError("Activation and weight reduction dimensions must match")
        if block_size not in (32, 128):
            raise ValueError("Activation blocks must contain 32 or 128 elements")
        x = (
            _activation(activation, scales, block_size)
            .half()
            .reshape(-1, activation.shape[-1])
            .contiguous()
        )
        return self._matmul_decoded(x, activation.shape)

    def _matmul_decoded(self, x, shape):
        """Reuse an already decoded FP16 activation across expert branches."""
        if (
            x.device != self.packed.device
            or x.dtype != torch.float16
            or x.ndim != 2
            or x.shape[-1] != self.packed.shape[-1] * 2
            or not x.is_contiguous()
        ):
            raise ValueError("Supply a contiguous FP16 activation matching the weight")
        threshold = int(os.environ.get("WININTEL_FP4_PREFILL_ROWS", "0"))
        if threshold > 0 and x.shape[0] >= threshold:
            from kernel import dequant_fp4

            output = torch.empty(
                (x.shape[0], self.packed.shape[0]), device=x.device, dtype=torch.float16
            )
            for begin in range(0, self.packed.shape[0], 256):
                end = min(begin + 256, self.packed.shape[0])
                matrix = dequant_fp4(
                    self.packed[begin:end], self.source_scales[begin:end]
                ).half()
                output[:, begin:end] = x @ matrix.T
            return output.reshape(*shape[:-1], self.packed.shape[0]).to(
                torch.get_default_dtype()
            )
        output = torch.empty(
            (x.shape[0], self.packed.shape[0]), device=x.device, dtype=torch.float16
        )
        for begin in range(0, x.shape[0], 16):
            output[begin : begin + 16] = self.extension.gemm_lookup(
                x[begin : begin + 16], self.packed, self.scale_ids, self.scale_table
            )
        return output.reshape(*shape[:-1], self.packed.shape[0]).to(
            torch.get_default_dtype()
        )


def prepare_weight(packed, scales):
    """Opt this tensor into the native operator after checkpoint preparation."""
    prepared = PreparedFP4(packed, scales)
    packed._winintel_fp4 = prepared
    return prepared


@lru_cache(maxsize=16)
def scale_table(device):
    """Reuse the original kernel's FP16 lookup for each E8M0 exponent and FP4 code."""
    from kernel import _scale_float

    exponents = torch.arange(256, dtype=torch.uint8, device=device)
    factors = _scale_float(exponents.view(torch.float8_e8m0fnu))
    magnitudes = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=device
    )
    values = torch.cat((magnitudes, -magnitudes))
    return (factors[:, None] * values[None, :]).half().unsqueeze(0).contiguous()
