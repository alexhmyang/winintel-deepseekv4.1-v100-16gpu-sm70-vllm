# winintel-deepseekv4.1-v100-16gpu-sm70-vllm

**Keep 16 V100 GPUs useful for the next generation of large models.**

[中文](README.md) · [Deployment](docs/deployment.md) ·
[Measurements](docs/validation.md) · [Provenance](docs/provenance.md) ·
[winintel AI](https://winintel.com)

Author: **数萌AI服务器 (winintel)** · service@winintel.com

An open-source inference research project for **DeepSeek V4.1 Flash on
16 × V100 32GB GPUs, Volta SM70**. It reuses the existing winintel SM70 FP4
WMMA operator and independently implements quantization adapters, checkpoint
loading, tensor and pipeline parallelism, decode graphs, and measurement tools.

**v0.1.0 is an independent source candidate.** Published separately from the
winintel-vllm adaptation work, this repository uses a PyTorch reference execution
path. It does not require an installed vLLM engine and is not a complete vLLM
serving distribution. Its current focus is numerical compatibility and operator
optimization; full-model throughput remains an active development target.

## Highlights

| Feature | Implementation and benefit |
| --- | --- |
| Existing SM70 operator reuse | Original FP4 WMMA CUDA source, packed expert weights, MXFP4 / E8M0 scale adaptation |
| FP8 decoded-weight cache | Reuses admitted FP32 matrices, with a default 256 MiB budget per GPU and invalidation on weight/scale changes |
| Small-batch decode graphs | Captures FP8 input quantization, activation decoding, GEMM, and complete expert computation |
| Shared expert inputs | Reuses gate/up quantization and FP16 activation decoding while preserving pruning, SwiGLU, routing weights, and output quantization order |
| Exact TP greedy selection | Exchanges score/token candidates, preserving global tie rules and nonfinite-value rejection |
| Distributed execution | TP4 × PP4, stage-local loading, demand-paged Engram reads, and per-GPU cache/graph measurements |
| Auditable source | Pinned official model revision, retained licenses, original operator hash, file manifest, and reproducible benchmarks |

## Measured operator improvements

These are **component measurements** on V100-SXM2-32GB, not full-model token/s.
The attention and expert graph comparisons use the previous input-reuse version
as their baseline.

| Measurement | Result | Scope |
| --- | --- | --- |
| FP8 matrix cache hits | 1.22–4.69× | Synthetic quantized matrices; excludes first decoding and non-admitted weights |
| FP8 attention decode graphs | 2.08–3.88× | Real checkpoint wq_a / wq_b / wkv; 1 / 2 / 16 rows |
| Complete FP4 expert graph | 4.5% less time for two requests | 0.8612 → 0.8222 ms, improving on the previous graph |
| Two-GPU, two-request greedy selection | 1.392× | 0.4112 → 0.2954 ms; excludes the final vocabulary projection |
| Real-weight numerical comparison | 15 elementwise-identical cases | Current graph/input-decoding changes; atol=rtol=0 |
| Release V100 regression | 61 passed in 16.82 seconds | Final runtime source; cache, graph, routing, and quantization behavior |

See [validation details](docs/validation.md) for conditions and full results.
The experimental large-batch FP4 prefill path remains disabled by default:
its complete-FFN relative error was approximately 0.6%.

## Full-model status

| Item | Current status |
| --- | --- |
| Tested full-model hardware | Two nodes, eight V100 32GB GPUs each, connected by InfiniBand |
| Native checkpoint | FP8 ordinary weights and MXFP4 experts; weights are distributed separately |
| Candidate context | Tested at 8K with two requests; this run configured a 16K maximum |
| 8K C2 fixed-step decode | 0.3782 token/s per request, below the project's 20 token/s threshold |
| Graph coverage | Local linear/expert graphs; no complete-model graph |
| Speculation / input | Token-by-token generation, no DSpark5; full-model validation uses text |
| Four-node layout | Template included; requires acceptance on actual four-node hardware |
| Serving | Generation and benchmark CLI; HTTP sessions are experimental, not a production OpenAI API |

The fixed 48-step result includes steps after EOS. Both requests produced nine
tokens through EOS; EOS-trimmed total response throughput was 0.6229 token/s,
with both retrieval markers found. Comprehensive model-quality evaluation remains
a separate acceptance task. Long-context results from other engines are not
results of this candidate.

## Getting started

Use **Linux, CUDA 12.8, Python 3.12, and an SM70-compatible PyTorch 2.10 build**.
Prepare local model files and a compatible Torch wheel, then follow the
[deployment guide](docs/deployment.md). Dependencies use a domestic mirror;
inference and tokenization read local files.

```bash
git clone https://github.com/alexhmyang/winintel-deepseekv4.1-v100-16gpu-sm70-vllm.git
cd winintel-deepseekv4.1-v100-16gpu-sm70-vllm
uv venv --python 3.12
uv pip install --python .venv/bin/python /path/to/sm70-compatible-torch.whl
uv pip install --python .venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
uv run --offline --no-project --python .venv/bin/python python verify_release.py
export WININTEL_REFERENCE_ENV="$PWD/.venv"
export CUDA_HOME=/usr/local/cuda-12.8
export WININTEL_NATIVE_SM70=1
bash run_reference.sh -m pytest tests/test_kernel.py -q
```

Deploy identical source, dependencies, and local model files on each node.
Copy the [two-node environment template](configs/two-node.env.example) and set
the actual HCA, network interface, and master address.
Run on both nodes, using rank 0 on the master and rank 1 on the second node:

```bash
set -a
source configs/cluster.env
set +a
bash launch_node.sh 0 --model "$WININTEL_MODEL" \
  --tp 4 --layers 10,10,11,9 --cuda-graph \
  --max-seq-len 16384 --prefill-chunk 128 \
  --prompt "Return only the result of 17 + 25." --new-tokens 16 \
  --output "$PWD/results.json"
```

Environment templates, short/8K C2 suites, graph A/B controls, and the four-node
layout are included. The master writes results. Full-model execution requires
available GPUs; do not launch it alongside an existing model occupying their memory.

## Source map

| Path | Purpose |
| --- | --- |
| `upstream/` | Pinned official model, existing SM70 FP4 source, and their licenses |
| `kernel.py`, `native_ops.py`, `qwen_adapter.py` | Reference numerics, FP8 cache, and FP4 WMMA adaptation |
| `expert_graph.py`, `greedy.py` | Expert graphs and exact TP candidate selection |
| `checkpoint.py`, `loading.py`, `distributed.py` | Checkpoint storage, sharding, Engram, and TP/PP |
| `generate.py`, `benchmark.py`, `benchmarks/` | Generation and full-model/component benchmarks |
| `configs/`, `tests/`, `docs/` | Reproduction settings, behavioral regression tests, and evidence |
| `provenance.json`, `manifest.json` | Source attribution and SHA256 file manifest |

## Open source and next steps

Priorities are full-model host overhead, complete decode graphs, FP8 residency,
and pipeline waiting. Meet the short-context C2 performance threshold before
expanding context or concurrency. Contributions with CUDA improvements,
four-stage timelines, model evaluations, or reproducible hardware results are welcome.

winintel additions are [Apache-2.0](LICENSE); official DeepSeek model code is
[MIT](upstream/LICENSE). Original notices are retained. We do not claim authorship
of the entire model or framework. See [NOTICE](NOTICE) and
[provenance](docs/provenance.md). AI assistance was used for development,
validation, and documentation.

**winintel AI** — [winintel.com](https://winintel.com) ·
[Bilibili](https://space.bilibili.com/429959998) ·
[YouTube](https://www.youtube.com/@winintelcom) · service@winintel.com
