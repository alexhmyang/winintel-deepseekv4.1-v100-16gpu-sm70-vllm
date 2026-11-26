# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Compare shared input quantization against the complete checkpoint expert."""

import argparse
import json
import math
import os
import statistics
import sys
from functools import partial
from pathlib import Path

import torch
from benchmark_winintel_v41_dispatch import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    from checkpoint import Checkpoint
    from expert_graph import install
    from loading import load_parameters
    from qwen_adapter import enable_cuda_graph
    from reference import load_official

    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This qualification requires an SM70 GPU")
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float16)
    os.environ["WININTEL_FP4_PREFILL_ROWS"] = "0"
    official = load_official()
    checkpoint = Checkpoint(args.model)
    name = sorted(
        name
        for name in checkpoint.index
        if ".ffn.experts." in name and name.endswith(".w1.weight")
    )[0]
    prefix = name.removesuffix(".w1.weight")
    shape = checkpoint.file_for(name).spec(name)[0]
    text = json.loads((checkpoint.root / "config.json").read_text())["text_config"]
    with torch.device("meta"):
        expert = official.Expert(
            shape[1] * 2,
            shape[0],
            dtype=torch.float4_e2m1fn_x2,
            swiglu_limit=text.get("swiglu_limit", 0),
        )
    load_parameters(official, expert, prefix, checkpoint, "cuda", 0, 1)
    original = expert.forward
    install(expert)
    results = []
    with torch.inference_mode():
        for rows in (1, 2, 16, 128, 2048):
            torch.manual_seed(82)
            x = torch.randn(rows, shape[1] * 2, device="cuda")
            route = torch.rand(rows, 1, device="cuda", dtype=torch.float32)
            enable_cuda_graph(False)
            expected = original(x, route)
            for graph in (False, True) if rows <= 16 else (False,):
                enable_cuda_graph(graph)
                timings = {"baseline": [], "candidate": []}
                for order in (
                    ("baseline", "candidate"),
                    ("candidate", "baseline"),
                ):
                    for mode in order:
                        os.environ["WININTEL_EXPERT_SHARED_QUANT"] = (
                            "0" if mode == "baseline" else "1"
                        )
                        actual = expert(x, route)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        timings[mode].append(
                            measure(partial(expert, x, route), args.repeats)
                        )
                baseline, candidate = (
                    statistics.median(timings[mode])
                    for mode in ("baseline", "candidate")
                )
                # Graph replay must use new inputs and new routing weights.
                x.mul_(0.5)
                route.mul_(0.25)
                actual = expert(x, route)
                enable_cuda_graph(False)
                torch.testing.assert_close(actual, original(x, route), atol=0, rtol=0)
                expected = original(x, route)
                result = dict(
                    rows=rows,
                    graph=graph,
                    baseline_ms=baseline,
                    candidate_ms=candidate,
                    speedup=baseline / candidate,
                    exact_match=True,
                )
                results.append(result)
                print(
                    f"EXPERT M={rows} graph={graph} "
                    f"{baseline:.4f}->{candidate:.4f}ms "
                    f"speedup={baseline / candidate:.3f} exact=True",
                    flush=True,
                )
    enable_cuda_graph(False)
    # Header-only inventory excludes embeddings and row-scaled tables.
    linear_weights = []
    for name in checkpoint.index:
        if not name.endswith(".weight"):
            continue
        scale_name = name.removesuffix(".weight") + ".scale"
        if scale_name not in checkpoint.index:
            continue
        dims, dtype, _, _ = checkpoint.file_for(name).spec(name)
        scale_dims = checkpoint.file_for(scale_name).spec(scale_name)[0]
        if (
            dtype == torch.float8_e4m3fn
            and len(dims) == 2
            and scale_dims == [(dims[0] + 31) // 32, dims[1] // 32]
        ):
            linear_weights.append(
                dict(name=name, shape=dims, decoded_bytes=math.prod(dims) * 4)
            )
    args.output.write_text(
        json.dumps(
            dict(
                author="数萌AI服务器 (winintel)",
                gpu=torch.cuda.get_device_name(),
                checkpoint_expert=prefix,
                dim=shape[1] * 2,
                intermediate=shape[0],
                results=results,
                fp8_linear_inventory=linear_weights,
                fp8_linear_unsharded_decoded_bytes=sum(
                    value["decoded_bytes"] for value in linear_weights
                ),
                scope="one real expert; FP8 inventory is unsharded metadata",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
