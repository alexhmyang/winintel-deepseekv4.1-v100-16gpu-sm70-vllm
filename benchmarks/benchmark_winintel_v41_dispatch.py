# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Measure independent SM70 dispatch and exact NCCL candidate selection."""

import argparse
import json
import os
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist


def measure(function, repeats):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def matrix_cases(repeats):
    import kernel
    import native_ops
    from qwen_adapter import prepare_weight

    torch.manual_seed(83)
    result = []
    for kind in ("fp8", "fp4"):
        n, k = (1792, 5120) if kind == "fp8" else (1280, 5120)
        if kind == "fp8":
            weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
            scales = torch.full(
                (n // 32, k // 32), 123, device="cuda", dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
            native_ops.prepare_fp8(weight, scales)
        else:
            weight = torch.randint(
                0, 256, (n, k // 2), device="cuda", dtype=torch.uint8
            ).view(torch.float4_e2m1fn_x2)
            scales = torch.full(
                (n, k // 32), 123, device="cuda", dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
            prepared = prepare_weight(weight, scales)
        for rows in (1, 16, 128, 512, 2048):
            x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
            a, a_s = kernel.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
            if kind == "fp8":
                run = partial(kernel.fp8_gemm, a, a_s, weight, scales, block_size=32)
                variable, baseline, candidate = "WININTEL_FP8_CACHE_MIB", "0", "256"
            else:
                run = partial(prepared.matmul, a, a_s, 32)
                variable, baseline, candidate = "WININTEL_FP4_PREFILL_ROWS", "0", "1"
            timings = {"baseline": [], "candidate": []}
            errors = []
            for order in (("baseline", "candidate"), ("candidate", "baseline")):
                for mode in order:
                    os.environ[variable] = baseline if mode == "baseline" else candidate
                    actual = run()
                    if mode == "candidate":
                        reference = kernel._activation(a, a_s, 32) @ (
                            kernel.dequant_fp4(weight, scales).T
                            if kind == "fp4"
                            else (
                                weight.float()
                                * kernel._scale_float(scales)[
                                    torch.arange(n, device="cuda") // 32
                                ].repeat_interleave(32, -1)
                            ).T
                        )
                        errors.append(
                            float(
                                (actual.float() - reference).norm()
                                / reference.norm().clamp_min(1e-12)
                            )
                        )
                        torch.testing.assert_close(
                            actual, reference.half(), atol=0.04, rtol=0.02
                        )
                    timings[mode].append(measure(run, repeats))
            old = statistics.median(timings["baseline"])
            new = statistics.median(timings["candidate"])
            result.append(
                dict(
                    kind=kind,
                    rows=rows,
                    n=n,
                    k=k,
                    baseline_ms=old,
                    candidate_ms=new,
                    speedup=old / new,
                    relative_error=max(errors),
                )
            )
            print(
                f"{kind} M={rows} baseline={old:.4f}ms candidate={new:.4f}ms "
                f"speedup={old / new:.3f}",
                flush=True,
            )
        native_ops._decoded_weights.clear()
    return result


def distributed_cases(repeats):
    from distributed import TensorParallel
    from greedy import select

    dist.init_process_group("nccl")
    parallel = TensorParallel(None)
    count, rank = dist.get_world_size(), dist.get_rank()
    result = []
    for batch in (1, 2, 16):
        torch.manual_seed(91 + rank)
        logits = torch.randn(batch, 131072 // count, device="cuda")

        def full(local_logits=logits):
            gathered = [torch.empty_like(local_logits) for _ in range(count)]
            dist.all_gather(gathered, local_logits)
            joined = torch.cat(gathered, -1)
            if not torch.isfinite(joined).all().item():
                raise RuntimeError("Nonfinite pretrained logits")
            return joined.argmax(-1)

        for tie in (False, True):
            if tie:
                logits[:, 3] = 50
                logits[:, 17] = 50
            torch.testing.assert_close(select(logits, parallel), full(), atol=0, rtol=0)
        timings = {"baseline": [], "candidate": []}
        for order in (("baseline", "candidate"), ("candidate", "baseline")):
            for mode in order:
                dist.barrier()
                timings[mode].append(
                    measure(
                        full
                        if mode == "baseline"
                        else partial(select, logits, parallel),
                        repeats,
                    )
                )
        old, new = (
            statistics.median(timings[mode]) for mode in ("baseline", "candidate")
        )
        result.append(
            dict(
                kind="greedy",
                batch=batch,
                tp=count,
                baseline_ms=old,
                candidate_ms=new,
                speedup=old / new,
                baseline_payload_bytes=logits.numel() * 4,
                candidate_payload_bytes=batch * 24,
            )
        )
        if rank == count - 1:
            logits[0, 0] = float("nan")
        try:
            select(logits, parallel)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Remote nonfinite logits were accepted")
    dist.destroy_process_group()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.set_default_dtype(torch.float16)
    torch.set_num_threads(2)
    name = torch.cuda.get_device_name()
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This qualification requires an SM70 GPU")
    # Weight mutation counters remain available for the cache qualification.
    with torch.no_grad():
        results = (
            distributed_cases(args.repeats)
            if args.distributed
            else matrix_cases(args.repeats)
        )
    if int(os.environ.get("RANK", "0")) == 0:
        args.output.write_text(
            json.dumps(
                dict(
                    author="数萌AI服务器 (winintel)",
                    gpu=name,
                    torch=torch.__version__,
                    measurement="host/device synchronized median; warmed; two orders",
                    results=results,
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
