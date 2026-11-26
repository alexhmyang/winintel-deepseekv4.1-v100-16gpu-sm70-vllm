# Author: 数萌AI服务器 (winintel)
"""Generate with native checkpoint weights over an independent TP/PP runtime."""

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from distributed import TensorParallel, Transport, groups
from loading import Stage, model_args
from reference import load_official
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt", default="Write a Python function to add two numbers."
    )
    parser.add_argument("--prompt-file")
    parser.add_argument("--suite", help="JSON benchmark cases; load the model once")
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--prefill-chunk", type=int, default=128)
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable native matrix and complete expert graphs during decode",
    )
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--layers", default="10,10,11,9")
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.float16)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", timeout=timedelta(hours=6))
    sizes = [int(size) for size in options.layers.split(",")]
    options.stage_sizes = sizes
    tp_group, pp_group = groups(options.tp, len(sizes))
    stage_id, tp_rank = divmod(rank, options.tp)
    official = load_official()
    official.world_size, official.rank = options.tp, tp_rank
    official.default_dtype = torch.float8_e4m3fn
    official.dist = TensorParallel(tp_group)
    args = model_args(
        official,
        options.model,
        1 if options.suite else options.batch_size,
        min(options.max_seq_len, 8192) if options.suite else options.max_seq_len,
    )
    if sum(sizes) != args.n_layers:
        raise ValueError("Layer partition must cover the complete backbone")
    tokenizer = AutoTokenizer.from_pretrained(options.model, local_files_only=True)
    started = time.monotonic()
    stage = Stage(
        official,
        args,
        options.model,
        sum(sizes[:stage_id]),
        sum(sizes[: stage_id + 1]),
        stage_id,
        len(sizes),
        tp_rank,
        options.tp,
        tp_group,
        tokenizer,
        device=torch.device("cuda", local_rank),
    ).eval()
    dist.barrier()
    loaded = time.monotonic()
    if rank == 0:
        print(f"MODEL_READY load_seconds={loaded - started:.3f}", flush=True)
    if options.suite:
        from benchmark import run_suite

        run_suite(
            stage,
            Transport(pp_group, stage.device),
            tokenizer,
            options,
            loaded - started,
        )
        dist.destroy_process_group()
        return
    text = (
        Path(options.prompt_file).read_text() if options.prompt_file else options.prompt
    )
    prompt = "<｜begin▁of▁sentence｜><｜User｜>" + text + "<｜Assistant｜></think>"
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    if len(encoded) + options.new_tokens > options.max_seq_len:
        raise ValueError("Prompt plus completion exceeds configured context")
    chunk = torch.tensor(encoded, device="cuda").repeat(options.batch_size, 1)
    if options.prefill_chunk < 1:
        raise ValueError("Prefill chunk must be positive")
    transport = Transport(pp_group, chunk.device)
    completion = []
    compact_greedy = os.environ.get("WININTEL_COMPACT_GREEDY", "1") == "1"
    times = []
    position = 0
    decode_rank = (len(sizes) - 1) * options.tp
    prompt_tokens = chunk
    for begin in range(0, len(encoded), options.prefill_chunk):
        chunk = prompt_tokens[:, begin : begin + options.prefill_chunk]
        if begin + chunk.size(1) == len(encoded):
            break
        boundary = None if stage_id == 0 else transport.receive(rank - options.tp)
        result = stage(
            chunk,
            position,
            boundary,
            greedy=compact_greedy,
            greedy_graph=options.cuda_graph,
        )
        if stage_id < len(sizes) - 1:
            transport.send(result, rank + options.tp)
        position += chunk.size(1)
        if rank == 0:
            print(f"PREFILL tokens={position}/{len(encoded)}", flush=True)
    for step in range(options.new_tokens):
        from qwen_adapter import enable_cuda_graph

        enable_cuda_graph(options.cuda_graph and step > 0)
        boundary = None if stage_id == 0 else transport.receive(rank - options.tp)
        result = stage(
            chunk,
            position,
            boundary,
            greedy=compact_greedy,
            greedy_graph=options.cuda_graph,
        )
        if stage_id < len(sizes) - 1:
            transport.send(result, rank + options.tp)
            next_token = torch.empty(
                options.batch_size, dtype=torch.int64, device=chunk.device
            )
        else:
            if not compact_greedy and not torch.isfinite(result).all().item():
                raise RuntimeError("Nonfinite pretrained logits")
            next_token = (
                result.flatten() if compact_greedy else result.argmax(-1).flatten()
            )
        dist.broadcast(next_token, src=decode_rank)
        torch.cuda.synchronize()
        now = time.monotonic()
        times.append(now - loaded)
        completion.append(next_token.cpu().tolist())
        position += chunk.size(1)
        chunk = next_token.unsqueeze(1)
        if rank == 0:
            print(
                f"TOKEN step={step} ids={completion[-1]} elapsed={times[-1]:.3f}",
                flush=True,
            )
    if rank == 0:
        outputs = torch.tensor(completion).T.tolist()
        report = {
            "author": "数萌AI服务器 (winintel)",
            "backend": "WININTEL_SM70",
            "precision": "FP4 WMMA FP16; FP8 PyTorch FP32; FP32 residuals",
            "tp": options.tp,
            "pipeline_layers": sizes,
            "input_tokens_per_request": len(encoded),
            "concurrency": options.batch_size,
            "prefill_chunk_tokens": options.prefill_chunk,
            "output_tokens_per_request": len(completion),
            "model_load_seconds": loaded - started,
            "first_token_seconds": times[0],
            "token_elapsed_seconds": times,
            "generated_token_ids": outputs,
            "generated_text": [tokenizer.decode(output) for output in outputs],
            "max_gpu_allocated_bytes_rank0": torch.cuda.max_memory_allocated(),
            "full_model_graph": False,
            "native_fp4_decode_graphs": options.cuda_graph,
        }
        Path(options.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
        print(json.dumps(report, ensure_ascii=False), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
