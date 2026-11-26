# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Qualify decoded activation reuse and cached graphs on checkpoint weights."""

import argparse
import json
import os
import statistics
import sys
from functools import partial
from pathlib import Path

import torch
from benchmark_winintel_v41_dispatch import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import native_ops
    from checkpoint import Checkpoint
    from expert_graph import install
    from loading import load_parameters
    from qwen_adapter import enable_cuda_graph
    from reference import load_official

    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("Qualification requires an SM70 GPU")
    torch.set_default_dtype(torch.float16)
    torch.set_num_threads(2)
    os.environ["WININTEL_FP8_CACHE_MIB"] = "256"
    os.environ["WININTEL_FP4_PREFILL_ROWS"] = "0"
    os.environ["WININTEL_EXPERT_SHARED_QUANT"] = "1"
    official = load_official()
    checkpoint = Checkpoint(args.model)
    text = json.loads((args.model / "config.json").read_text())["text_config"]
    cases = [
        ("fp4_expert", "layers.0.ffn.experts.0", torch.float4_e2m1fn_x2),
        ("fp8_shared_expert", "layers.0.ffn.shared_experts", torch.float8_e4m3fn),
        ("fp8_wq_a", "layers.0.attn.wq_a", torch.float8_e4m3fn),
        ("fp8_wq_b", "layers.0.attn.wq_b", torch.float8_e4m3fn),
        ("fp8_wkv", "layers.0.attn.wkv", torch.float8_e4m3fn),
    ]
    results = []
    for kind, prefix, dtype in cases:
        native_ops._decoded_weights.clear()
        expert_case = "expert" in kind
        name = prefix + (".w1.weight" if expert_case else ".weight")
        shape = checkpoint.file_for(name).spec(name)[0]
        k = shape[1] * (2 if dtype == torch.float4_e2m1fn_x2 else 1)
        with torch.device("meta"):
            module = (
                official.Expert(
                    k, shape[0], dtype=dtype, swiglu_limit=text.get("swiglu_limit", 0)
                )
                if expert_case
                else official.Linear(k, shape[0], dtype=dtype)
            )
        load_parameters(official, module, prefix, checkpoint, "cuda", 0, 1)
        original = module.forward
        if expert_case:
            install(module)
        with torch.inference_mode():
            for rows in (1, 2, 16):
                torch.manual_seed(82)
                x = torch.randn(rows, k, device="cuda")
                routing = torch.rand(rows, 1, device="cuda", dtype=torch.float32)
                run = partial(module, x, routing) if expert_case else partial(module, x)
                enable_cuda_graph(False)
                expected = original(x, routing) if expert_case else original(x)
                enable_cuda_graph(True)
                timings = {"baseline": [], "candidate": []}
                for order in (
                    ("baseline", "candidate"),
                    ("candidate", "baseline"),
                ):
                    for mode in order:
                        value = "0" if mode == "baseline" else "1"
                        os.environ["WININTEL_FP8_GRAPH"] = value
                        os.environ["WININTEL_EXPERT_SHARED_DECODE"] = value
                        torch.testing.assert_close(run(), expected, atol=0, rtol=0)
                        timings[mode].append(measure(run, args.repeats))
                baseline, candidate = (
                    statistics.median(timings[mode])
                    for mode in ("baseline", "candidate")
                )
                os.environ["WININTEL_FP8_GRAPH"] = "1"
                os.environ["WININTEL_EXPERT_SHARED_DECODE"] = "1"
                saved = run()
                x.mul_(0.5)
                routing.mul_(0.25)
                actual = run()
                enable_cuda_graph(False)
                changed = original(x, routing) if expert_case else original(x)
                torch.testing.assert_close(actual, changed, atol=0, rtol=0)
                # Previously returned output must survive replay and eviction.
                torch.testing.assert_close(saved, expected, atol=0, rtol=0)
                results.append(
                    dict(
                        kind=kind,
                        checkpoint_prefix=prefix,
                        rows=rows,
                        n=shape[0],
                        k=k,
                        baseline_ms=baseline,
                        candidate_ms=candidate,
                        speedup=baseline / candidate,
                        exact_match=True,
                        cache_bytes=sum(
                            entry[3].numel() * 4
                            for entry in native_ops._decoded_weights.values()
                        ),
                    )
                )
                print(
                    f"{kind} M={rows} {baseline:.4f}->{candidate:.4f}ms "
                    f"speedup={baseline / candidate:.3f} exact=True",
                    flush=True,
                )
        enable_cuda_graph(False)
        del module
    native_ops._decoded_weights.clear()
    args.output.write_text(
        json.dumps(
            dict(
                author="数萌AI服务器 (winintel)",
                gpu=torch.cuda.get_device_name(),
                results=results,
                scope="real checkpoint modules; warmed decode; two test orders",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
