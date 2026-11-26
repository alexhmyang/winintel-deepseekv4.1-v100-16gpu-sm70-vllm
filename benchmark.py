# Author: 数萌AI服务器 (winintel)
"""Repeatable independent text requests on one resident distributed model."""

import hashlib
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist


class CacheCapacityError(RuntimeError):
    """Every rank has agreed that request-cache allocation failed."""


def speed_rejection(case, results):
    """Stop increasing concurrency after a completed request misses 20 tok/s."""
    for result in results:
        speed = result.get("decode_tokens_per_second_per_request")
        if (
            result.get("status") == "completed"
            and speed is not None
            and speed < 20
            and case["concurrency"] > result["concurrency"]
        ):
            return (
                f"Speed below threshold: {result['name']} reached "
                f"{speed:.3f} tok/s per request; higher concurrency stopped"
            )
    return None


def prompts(tokenizer, case):
    """Build equal-length requests with different identifiers and retrieval keys."""
    count = case["concurrency"]
    target = case.get("input_tokens")
    rows, markers = [], []
    for request in range(count):
        marker = f"WININTEL_KEY_{request:03d}_739126"
        prefix = tokenizer.encode(
            "<｜begin▁of▁sentence｜><｜User｜>"
            f"Request {request}. The secret key is {marker}.\n",
            add_special_tokens=False,
        )
        suffix = tokenizer.encode(
            "\n"
            + case.get(
                "prompt",
                "Return the secret key from the beginning, then implement a "
                "complete Python LRUCache class with get and put methods, "
                "eviction, type hints, and usage examples. Write only the key "
                "and code.",
            )
            + "<｜Assistant｜></think>",
            add_special_tokens=False,
        )
        filler = tokenizer.encode(
            "# Ordinary reference data. def square(x): return x * x\n",
            add_special_tokens=False,
        )
        length = target or len(prefix) + len(suffix) + len(filler)
        space = length - len(prefix) - len(suffix)
        if space < 0:
            raise ValueError("Input length cannot accommodate the request")
        rows.append(prefix + (filler * (space // len(filler) + 1))[:space] + suffix)
        markers.append(marker)
    if len({len(row) for row in rows}) != 1:
        raise ValueError("Benchmark requests must have equal token counts")
    if len({tuple(row) for row in rows}) != count:
        raise ValueError("Concurrent requests must have distinct token sequences")
    return rows, markers


@torch.inference_mode()
def run_case(
    stage, transport, tokenizer, options, case, supplied_prompts=None, on_token=None
):
    from qwen_adapter import enable_cuda_graph

    enable_cuda_graph(False)
    rank = dist.get_rank()
    stage_id = rank // options.tp
    stages = len(options.stage_sizes)
    decode_rank = (stages - 1) * options.tp
    rows, markers = supplied_prompts or prompts(tokenizer, case)
    batch, length = len(rows), len(rows[0])
    tokens = case.get("new_tokens", options.new_tokens)
    chunk_size = case.get("prefill_chunk", options.prefill_chunk)
    if length + tokens > options.max_seq_len:
        raise ValueError("Benchmark exceeds the configured context capacity")
    if min(tokens, chunk_size) < 1:
        raise ValueError("Output length and prefill chunk must be positive")
    transport.reset_request(length + tokens)
    allocation_error = None
    try:
        stage.reset_request(batch, length + tokens)
    except torch.cuda.OutOfMemoryError as error:
        allocation_error = f"rank {rank}: {error}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, allocation_error)
    if any(errors):
        raise CacheCapacityError("; ".join(error for error in errors if error))
    prompt = torch.tensor(rows, dtype=torch.int64, device=stage.device)
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    paging_bytes_before = sum(
        source.bytes_read for source in stage.checkpoint.files.values()
    )
    started = time.monotonic()
    position, completion, times = 0, [], []
    next_progress = 1024
    compact_greedy = os.environ.get("WININTEL_COMPACT_GREEDY", "1") == "1"

    def forward(chunk):
        boundary = None if stage_id == 0 else transport.receive(rank - options.tp)
        result = stage(
            chunk,
            position,
            boundary,
            greedy=compact_greedy,
            greedy_graph=options.cuda_graph,
        )
        if stage_id < stages - 1:
            transport.send(result, rank + options.tp)
        return result

    for begin in range(0, length, chunk_size):
        chunk = prompt[:, begin : begin + chunk_size]
        if begin + chunk.size(1) == length:
            break
        forward(chunk)
        position += chunk.size(1)
        if rank == 0 and position >= next_progress:
            print(
                f"CASE {case['name']} PREFILL {position}/{length} "
                f"seconds={time.monotonic() - started:.3f}",
                flush=True,
            )
            next_progress = position + 1024
    for step in range(tokens):
        enable_cuda_graph(options.cuda_graph and step > 0)
        profiling = case.get("profile_decode", False) and step == 8
        profile = None
        if profiling and rank % options.tp == 0:
            profile = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            )
        with profile if profile is not None else nullcontext():
            result = forward(chunk)
            if profile is not None:
                torch.cuda.synchronize()
        if profile is not None:
            trace = Path(options.output).with_name(
                f"{case['name']}-rank{rank}-decode.json"
            )
            profile.export_chrome_trace(str(trace))
            table = profile.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=30
            )
            trace.with_suffix(".txt").write_text(table)
        if stage_id < stages - 1:
            next_token = torch.empty(batch, dtype=torch.int64, device=stage.device)
        else:
            if not compact_greedy and not torch.isfinite(result).all().item():
                raise RuntimeError(f"{case['name']}: nonfinite pretrained logits")
            next_token = (
                result.flatten() if compact_greedy else result.argmax(-1).flatten()
            )
        dist.broadcast(next_token, src=decode_rank)
        torch.cuda.synchronize()
        times.append(time.monotonic() - started)
        completion.append(next_token.cpu().tolist())
        if on_token is not None:
            on_token(completion[-1], step)
        position += chunk.size(1)
        chunk = next_token.unsqueeze(1)
        if rank == 0:
            print(
                f"CASE {case['name']} TOKEN {step} ids={completion[-1]} "
                f"seconds={times[-1]:.3f}",
                flush=True,
            )
            if step % 8 == 7:
                partial = torch.tensor(completion).T.tolist()
                print(
                    f"CASE {case['name']} PARTIAL_TEXT "
                    + json.dumps(
                        [tokenizer.decode(row) for row in partial[:4]],
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    from native_ops import cache_statistics

    local_memory = {
        "rank": rank,
        "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
        "reserved_peak_bytes": torch.cuda.max_memory_reserved(),
        "fp8_weight_cache": cache_statistics(stage.device),
        "checkpoint_bytes_read_during_case": (
            sum(source.bytes_read for source in stage.checkpoint.files.values())
            - paging_bytes_before
        ),
        "engram_row_sources": sorted(
            {
                type(stage.checkpoint.file_for(name)).__name__
                for name in stage.checkpoint.index
                if name.endswith(".engram.embed.weight")
            }
        ),
    }
    memory = [None] * dist.get_world_size()
    dist.all_gather_object(memory, local_memory)
    if rank != 0:
        return None
    outputs = torch.tensor(completion).T.tolist()
    visible_outputs = [
        output[: output.index(tokenizer.eos_token_id)]
        if tokenizer.eos_token_id in output
        else output
        for output in outputs
    ]
    text = [tokenizer.decode(output) for output in visible_outputs]
    decode_seconds = times[-1] - times[0]
    response_end_times = [
        times[min(len(output), tokens - 1)] for output in visible_outputs
    ]
    response_decode_seconds = max(response_end_times) - times[0]
    useful_decode_tokens = sum(max(0, len(output) - 1) for output in visible_outputs)
    return {
        "name": case["name"],
        "input_tokens_per_request": length,
        "concurrency": batch,
        "prefill_chunk_tokens": chunk_size,
        "output_tokens_per_request": tokens,
        "first_token_seconds": times[0],
        "prefill_effective_tokens_per_second": batch * length / times[0],
        "decode_tokens_per_second_total": (
            batch * (tokens - 1) / decode_seconds if decode_seconds > 0 else None
        ),
        "decode_tokens_per_second_per_request": (
            (tokens - 1) / decode_seconds if decode_seconds > 0 else None
        ),
        "token_elapsed_seconds": times,
        "prompt_sha256": [
            hashlib.sha256(json.dumps(row).encode()).hexdigest() for row in rows
        ],
        "generated_token_ids": outputs,
        "generated_text": text,
        "generated_tokens_before_eos": [len(output) for output in visible_outputs],
        "response_completion_seconds": response_end_times,
        "response_decode_tokens_per_second_total": (
            useful_decode_tokens / response_decode_seconds
            if response_decode_seconds > 0
            else None
        ),
        "response_decode_timing": (
            "useful tokens after first token divided by time until all responses "
            "reach EOS or the configured output limit"
        ),
        "decode_eos_policy": "fixed decode steps; EOS and subsequent steps included",
        "expected_retrieval_keys": markers,
        "retrieval_key_present": [
            marker in answer for marker, answer in zip(markers, text)
        ],
        "gpu_memory": memory,
        "full_model_graph": False,
        "diagnostic_profile_enabled": bool(case.get("profile_decode", False)),
        "native_fp4_decode_graphs": options.cuda_graph,
        "complete_expert_decode_graphs": options.cuda_graph
        and any(
            getattr(module, "_winintel_expert_graph", False)
            for module in stage.modules()
        ),
        "grouped_decode_routes": any(
            getattr(module, "_winintel_grouped_decode_routes", False)
            for module in stage.modules()
        ),
        "execution": "distinct requests in a synchronous batch; no HTTP scheduler",
    }


def capacity(stage, transport, batch, length):
    """Estimate cache admission before touching any rank's live buffers."""
    import math

    from request_cache import CACHE_NAMES, specifications

    old, needed, seen = 0, batch * length * 8, set()
    for layer in stage.layers:
        for child in layer.attn.modules():
            for name, value in child.named_buffers(recurse=False):
                if name not in CACHE_NAMES and name != "freqs_cis":
                    continue
                if value.data_ptr() not in seen:
                    old += value.numel() * value.element_size()
                    seen.add(value.data_ptr())
    for _, name, template, dtype, ratio in specifications(stage):
        shape = list(template)
        shape[0] = batch
        if name in ("compress_kv_cache", "k_cache"):
            shape[1] = length // ratio
        needed += math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if stage.hash_state.cache is not None:
        old += stage.hash_state.cache.numel() * stage.hash_state.cache.element_size()
    # reset_request shares two RoPE tables across attention/indexer modules.
    kinds = {bool(layer.attn.compress_ratio) for layer in stage.layers}
    needed += len(kinds) * length * stage.args.rope_head_dim * 4
    for cache in transport.received_cache.values():
        old += cache.numel() * cache.element_size()
    sources = [i for i in stage.args.kv_source_layers if i < stage.start]
    if sources:
        ratio = stage.args.compress_ratios[sources[-1]]
        needed += (
            batch
            * (length // ratio)
            * (stage.args.head_dim + stage.args.index_head_dim)
            * 2
        )
    free, _ = torch.cuda.mem_get_info(stage.device)
    reclaimable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    # Leave room for native operator graphs, temporary quantization and paging.
    margin = 2 * 1024**3
    return {
        "rank": dist.get_rank(),
        "estimated_cache_bytes": needed,
        "workspace_margin_bytes": margin,
        "available_after_cache_release_bytes": free + reclaimable + old,
        "admitted": needed + margin <= free + reclaimable + old,
        "estimate_only": True,
    }


def run_suite(stage, transport, tokenizer, options, load_seconds):
    import os
    from types import MethodType

    from request_cache import install as install_request_cache
    from routing import install

    install_request_cache(stage)
    for layer in stage.layers:
        install(stage.official, layer.ffn)
    native_hc = os.environ.get("WININTEL_NATIVE_HC", "0") == "1"
    suite = json.loads(Path(options.suite).read_text())
    cases = suite["cases"]
    switches = {
        "fp8_graph": "WININTEL_FP8_GRAPH",
        "expert_shared_decode": "WININTEL_EXPERT_SHARED_DECODE",
    }
    defaults = {key: os.environ.get(env, "1") for key, env in switches.items()}
    report = {
        "author": "数萌AI服务器 (winintel)",
        "backend": "WININTEL_SM70",
        "precision": "native FP4/FP8 weights; FP16 compute; FP32 residuals",
        "transport": "NCCL InfiniBand required by launch_node.sh",
        "tp": options.tp,
        "pipeline_layers": options.stage_sizes,
        "model_load_seconds": load_seconds,
        "cases": [],
    }
    for case in cases:
        for key, env in switches.items():
            if key in case and not isinstance(case[key], bool):
                raise ValueError(f"{key} must be a boolean")
            os.environ[env] = str(int(case[key])) if key in case else defaults[key]
        if case.get("native_attention"):
            raise ValueError("Retired attention backend is not supported")
        requested_hc = case.get("native_hc", native_hc)
        if requested_hc != native_hc:
            if requested_hc:
                from native_hc import install as install_native_hc

                for layer in stage.layers:
                    install_native_hc(stage.official, layer)
            else:
                for layer in stage.layers:
                    layer.hc_mixes = MethodType(stage.official.Block.hc_mixes, layer)
            native_hc = requested_hc
        admission = capacity(
            stage,
            transport,
            case["concurrency"],
            case["input_tokens"] + case.get("new_tokens", options.new_tokens),
        )
        admissions = [None] * dist.get_world_size()
        dist.all_gather_object(admissions, admission)
        allowed = [None]
        if dist.get_rank() == 0:
            print(f"CASE_START {json.dumps(case)}", flush=True)
            reason = speed_rejection(case, report["cases"])
            if not all(item["admitted"] for item in admissions):
                reason = "Estimated cache plus workspace exceeds GPU capacity"
            if case["input_tokens"] >= 1000000:
                for threshold in (8192, 262144):
                    qualified = [
                        item
                        for item in report["cases"]
                        if item.get("input_tokens_per_request") == threshold
                        and item.get("status") == "completed"
                        and all(item.get("retrieval_key_present", []))
                    ]
                    if not any(item["concurrency"] > 1 for item in qualified):
                        reason = "Prior 8K/256K concurrent retrieval not validated"
            allowed[0] = reason
        dist.broadcast_object_list(allowed, src=0)
        if allowed[0]:
            result = {
                "name": case["name"],
                "status": "not_run",
                "reason": allowed[0],
                "input_tokens_per_request": case["input_tokens"],
                "concurrency": case["concurrency"],
                "capacity_estimates": admissions,
            }
        else:
            runner = run_case
            if suite.get("http_sessions", False):
                from http_sessions import run_case as runner
            try:
                result = runner(stage, transport, tokenizer, options, case)
            except CacheCapacityError as error:
                result = {
                    "name": case["name"],
                    "status": "not_run",
                    "reason": f"Request cache allocation failed: {error}",
                    "input_tokens_per_request": case["input_tokens"],
                    "concurrency": case["concurrency"],
                }
            if dist.get_rank() == 0:
                result.setdefault("status", "completed")
                result["capacity_estimates"] = admissions
        if dist.get_rank() == 0:
            result["native_hyper_connections"] = native_hc
            result["attention_backend"] = "WININTEL_SM70"
            result["decode_optimizations"] = {
                key: os.environ[env] == "1" for key, env in switches.items()
            }
            report["cases"].append(result)
            Path(options.output).write_text(
                json.dumps(report, ensure_ascii=False, indent=2)
            )
            print(f"CASE_DONE {json.dumps(result, ensure_ascii=False)}", flush=True)
        dist.barrier()


if __name__ == "__main__":
    import argparse

    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Validate benchmark request tokens")
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", required=True)
    options = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(options.model, local_files_only=True)
    for case in json.loads(Path(options.suite).read_text())["cases"]:
        rows, _ = prompts(tokenizer, case)
        print(
            f"{case['name']}: {len(rows)} distinct requests, "
            f"{len(rows[0])} input tokens each"
        )
