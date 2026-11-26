# Author: 数萌AI服务器 (winintel)
"""Replay the complete expert FFN without changing checkpoint arithmetic."""

import os
from types import MethodType

import torch


def install(expert):
    """Keep one decode shape per expert; routing and TP reductions remain eager."""
    if getattr(expert, "_winintel_expert_graph", False):
        return
    original = expert.forward
    settings = original.__func__.__globals__
    state = {}

    def compute(x, weights):
        import kernel
        import qwen_adapter

        dtype = expert.w1.weight.dtype
        if (
            os.environ.get("WININTEL_EXPERT_SHARED_QUANT", "1") != "1"
            or dtype not in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn)
            or expert.w3.weight.dtype != dtype
            or expert.w1.bias is not None
            or expert.w3.bias is not None
        ):
            return original(x, weights)
        block = settings["fp8_block_size"]
        scale_dtype = settings["scale_dtype"]
        activation, scales = kernel.act_quant(
            x, block, settings["scale_fmt"], scale_dtype
        )
        decoded = None
        if (
            dtype == torch.float4_e2m1fn_x2
            and getattr(expert.w1.weight, "_winintel_native", False)
            and getattr(expert.w3.weight, "_winintel_native", False)
            and os.environ.get("WININTEL_EXPERT_SHARED_DECODE", "1") == "1"
            and (
                not qwen_adapter._cuda_graph_enabled
                or x.numel() // x.shape[-1] <= 16
                or torch.cuda.is_current_stream_capturing()
            )
        ):
            decoded = (
                kernel._activation(activation, scales, block)
                .half()
                .reshape(-1, x.shape[-1])
                .contiguous()
            )

        def project(layer):
            if dtype == torch.float4_e2m1fn_x2:
                if decoded is not None:
                    prepared = qwen_adapter.cached_weight(
                        layer.weight, layer.weight.scale
                    )
                    if prepared.source_scales is not layer.weight.scale:
                        raise ValueError("Reprepare native FP4 after replacing scales")
                    return prepared._matmul_decoded(decoded, x.shape).float()
                return kernel.fp4_gemm(
                    activation,
                    scales,
                    layer.weight,
                    layer.weight.scale,
                    scale_dtype,
                    act_block_size=block,
                ).float()
            return kernel.fp8_gemm(
                activation,
                scales,
                layer.weight,
                layer.weight.scale,
                scale_dtype,
                block_size=block,
            ).float()

        gate, up = project(expert.w1), project(expert.w3)
        if expert.swiglu_limit > 0:
            up = up.clamp(-expert.swiglu_limit, expert.swiglu_limit)
            gate = gate.clamp(max=expert.swiglu_limit)
        hidden = torch.nn.functional.silu(gate) * up
        if weights is not None:
            hidden = weights * hidden
        return expert.w2(hidden.to(x.dtype))

    def forward(self, x, weights=None):
        import qwen_adapter

        if (
            not qwen_adapter._cuda_graph_enabled
            or not x.is_cuda
            or x.ndim != 2
            or x.size(0) > 16
            or torch.cuda.is_current_stream_capturing()
        ):
            return compute(x, weights)
        key = (
            x.shape,
            x.dtype,
            x.device,
            None if weights is None else (weights.shape, weights.dtype),
            self.swiglu_limit,
            torch.get_default_dtype(),
            os.environ.get("WININTEL_EXPERT_SHARED_QUANT", "1"),
            os.environ.get("WININTEL_EXPERT_SHARED_DECODE", "1"),
            os.environ.get("WININTEL_FP8_GRAPH", "1"),
        )
        if state.get("key") != key:
            state.clear()
            activation = x.clone()
            routing = None if weights is None else weights.clone()
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(stream):
                compute(activation, routing)
            torch.cuda.current_stream(x.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = compute(activation, routing)
            state.update(
                key=key,
                activation=activation,
                routing=routing,
                graph=graph,
                output=output,
            )
        else:
            state["activation"].copy_(x)
            if weights is not None:
                state["routing"].copy_(weights)
        state["graph"].replay()
        return state["output"].clone()

    expert.forward = MethodType(forward, expert)
    expert._winintel_expert_graph = True
