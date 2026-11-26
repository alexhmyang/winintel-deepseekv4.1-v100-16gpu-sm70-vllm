# Author: 数萌AI服务器 (winintel)
"""Portable reference backend for the official DeepSeek V4.1 kernel API.

Storage formats and interfaces follow DeepSeek's MIT-licensed reference.
Arithmetic uses PyTorch FP32 and needs no architecture-specific FP8/FP4 GEMM.
This is a correctness backend, not a qualified high-throughput serving engine.
"""

import torch

BACKEND_NAME = "WININTEL_SM70"
FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _groups(x, size):
    if size <= 0 or x.shape[-1] % size:
        raise ValueError("The last dimension must be divisible by block size")
    return x.float().reshape(*x.shape[:-1], -1, size)


def _scale_float(scales):
    if scales.dtype == torch.float8_e8m0fnu:
        codes = scales.view(torch.uint8)
        result = torch.ldexp(
            torch.ones_like(codes, dtype=torch.float32), codes.int() - 127
        )
        return result.masked_fill(codes == 255, float("nan"))
    return scales.float()


def _store_scale(scales, dtype):
    if dtype == torch.float8_e8m0fnu:
        # E8M0 is an unsigned exponent with bias 127; 255 is NaN.
        return (torch.log2(scales).round().int() + 127).to(torch.uint8).view(dtype)
    return scales.to(dtype)


def _power_scale(amax, maximum):
    exponent = torch.ceil(torch.log2(amax / maximum)).clamp(-126, 127)
    return torch.ldexp(torch.ones_like(amax), exponent.int())


def act_quant(
    x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False
):
    values = _groups(x, block_size)
    peak = values.abs().amax(-1).clamp_min(1e-4)
    scales = _power_scale(peak, 448.0) if scale_fmt is not None else peak / 448.0
    stored = _store_scale(scales, scale_dtype)
    encoded = (
        (values / scales.unsqueeze(-1))
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .reshape(x.shape)
    )
    if inplace:
        decoded = (
            encoded.float().reshape(values.shape) * _scale_float(stored).unsqueeze(-1)
        ).reshape(x.shape)
        x.copy_(decoded.to(x.dtype))
        return x
    return encoded, stored


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    if scale_dtype not in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
        raise ValueError("FP4 scales must use E8M0 or E4M3")
    values = _groups(x, block_size)
    if block_size % 2:
        raise ValueError("FP4 packing requires an even block size")
    peak = values.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        scales = (peak.clamp_min(6 * 2**-9) / 6).to(scale_dtype).float()
    else:
        scales = _power_scale(peak.clamp_min(6 * 2**-126), 6)
    normalized = (values / scales.unsqueeze(-1)).clamp(-6, 6)
    magnitude = normalized.abs()
    # Midpoints select the even significand on an exact tie.
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    midpoints = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    for index, boundary in enumerate(midpoints):
        selected = magnitude > boundary if index % 2 == 0 else magnitude >= boundary
        code += selected.to(torch.uint8)
    code |= torch.signbit(normalized).to(torch.uint8) << 3
    code = code.reshape(x.shape)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()
    stored = _store_scale(scales, scale_dtype)
    packed = packed.view(torch.float4_e2m1fn_x2)
    if inplace:
        x.copy_(dequant_fp4(packed, stored, block_size).to(x.dtype))
        return x
    return packed, stored


def dequant_fp4(packed, scales, block_size=32):
    raw = packed.view(torch.uint8)
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
    exponent = ((codes >> 1) & 3).int()
    mantissa = (codes & 1).float()
    values = torch.where(
        exponent == 0,
        mantissa * 0.5,
        torch.ldexp(1 + mantissa * 0.5, exponent - 1),
    )
    values = torch.where((codes & 8) != 0, -values, values)
    grouped = _groups(values, block_size)
    if scales.shape != grouped.shape[:-1]:
        raise ValueError("FP4 scales must contain one value per row and K block")
    return (grouped * _scale_float(scales).unsqueeze(-1)).reshape(values.shape)


def _activation(a, scales, size):
    grouped = _groups(a, size)
    expected = grouped.shape[:-1]
    if scales.numel() != grouped.numel() // size:
        raise ValueError("Activation scale count does not match the tensor")
    return (grouped * _scale_float(scales).reshape(expected).unsqueeze(-1)).reshape(
        a.shape
    )


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    if block_size not in (32, 128) or b.ndim != 2:
        raise ValueError("FP8 GEMM requires a matrix and block size 32 or 128")
    n, k = b.shape
    if a.shape[-1] != k or k % block_size:
        raise ValueError("Incompatible GEMM reduction dimensions")
    expected = ((n + block_size - 1) // block_size, k // block_size)
    if tuple(b_s.shape) != expected:
        raise ValueError("FP8 weight scales must cover N-by-K blocks")
    if getattr(b, "_winintel_q8", None) is not None:
        from native_ops import fp8_matmul

        return fp8_matmul(a, a_s, b, b_s, block_size)
    rows = torch.arange(n, device=b.device) // block_size
    weight_scales = _scale_float(b_s)[rows].repeat_interleave(block_size, -1)
    weight = b.float() * weight_scales
    return (_activation(a, a_s, block_size) @ weight.T).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    if act_block_size not in (32, 128) or b.ndim != 2:
        raise ValueError("FP4 GEMM requires a matrix and block size 32 or 128")
    if b.shape[-1] * 2 != a.shape[-1]:
        raise ValueError("Packed FP4 weights have the wrong reduction dimension")
    prepared = getattr(b, "_winintel_fp4", None)
    if getattr(b, "_winintel_native", False):
        from qwen_adapter import cached_weight

        prepared = cached_weight(b, b_s)
    if prepared is not None:
        if prepared.source_scales is not b_s:
            raise ValueError("Reprepare the native weight after replacing its scales")
        return prepared.matmul(a, a_s, act_block_size)
    return (_activation(a, a_s, act_block_size) @ dequant_fp4(b, b_s).T).to(
        torch.get_default_dtype()
    )


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Gather shared latent KV with a zero-valued sink and bounded workspace."""
    batch, sequence, heads, dim = q.shape
    parts = kv if isinstance(kv, tuple) else (kv,)
    total = sum(part.size(1) for part in parts)
    if (
        not parts
        or any(
            part.ndim != 3 or part.shape[0] != batch or part.shape[-1] != dim
            for part in parts
        )
        or topk_idxs.shape[:2] != (batch, sequence)
        or attn_sink.shape != (heads,)
        or total == 0
    ):
        raise ValueError("Incompatible sparse attention shapes")
    output = torch.empty_like(q)
    batches = torch.arange(batch, device=q.device)[:, None, None]
    for begin in range(0, sequence, 16):
        query = q[:, begin : begin + 16].float()
        shape = query.shape[:-1]
        maximum = attn_sink.float().expand(shape)
        denominator = torch.ones(shape, device=q.device, dtype=torch.float32)
        accumulator = torch.zeros_like(query)
        for offset in range(0, topk_idxs.shape[-1], 64):
            indices = topk_idxs[:, begin : begin + 16, offset : offset + 64].long()
            valid = (indices >= 0) & (indices < total)
            selected = None
            start = 0
            for part in parts:
                if part.size(1):
                    local = indices - start
                    values = part[batches, local.clamp(0, part.size(1) - 1)].float()
                    selected = (
                        values
                        if selected is None
                        else torch.where(
                            ((local >= 0) & (local < part.size(1))).unsqueeze(-1),
                            values,
                            selected,
                        )
                    )
                start += part.size(1)
            scores = torch.einsum("bshd,bstd->bsht", query, selected)
            scores = (scores * softmax_scale).masked_fill(
                ~valid.unsqueeze(-2), float("-inf")
            )
            next_max = torch.maximum(maximum, scores.amax(-1))
            correction = torch.exp(maximum - next_max)
            probabilities = torch.exp(scores - next_max.unsqueeze(-1))
            accumulator = accumulator * correction.unsqueeze(-1) + torch.einsum(
                "bsht,bstd->bshd", probabilities, selected
            )
            denominator = denominator * correction + probabilities.sum(-1)
            maximum = next_max
        output[:, begin : begin + 16] = (accumulator / denominator.unsqueeze(-1)).to(
            q.dtype
        )
    return output


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    if (
        hc_mult < 1
        or sinkhorn_iters < 1
        or eps <= 0
        or mixes.shape[-1] != (2 + hc_mult) * hc_mult
        or hc_scale.shape != (3,)
        or hc_base.shape != (mixes.shape[-1],)
    ):
        raise ValueError("Invalid hyper-connection dimensions or iterations")
    x = mixes.float()
    pre = torch.sigmoid(x[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        x[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = (x[..., 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]).reshape(
        *x.shape[:-1], hc_mult, hc_mult
    )
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
