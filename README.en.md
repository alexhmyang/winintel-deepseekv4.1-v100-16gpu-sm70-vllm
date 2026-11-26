# winintel-deepseekv4.1-v100-16gpu-sm70-vllm

**Put affordable V100 GPUs to work on complete models in the 400+ GB class.**

[中文](README.md) · [Deployment](docs/deployment.md) ·
[Measurements](docs/validation.md) · [Provenance](docs/provenance.md) ·
[winintel AI](https://winintel.com)

Author: **数萌AI服务器 (winintel)** · service@winintel.com

This project aims to let businesses deploy **complete models in the 400+ GB
class, such as DeepSeek V4.1 Flash**, on affordable existing **V100 GPUs**.
The goal is reasoning, coding, and knowledge capabilities suitable for commercial
tasks, with a lower hardware barrier to local deployment.

The core layout is **two servers, each with eight V100 32GB GPUs, connected by
100 Gb/s InfiniBand**: 16 GPUs with 512 GB of aggregate physical VRAM.
"Complete model" means retaining the official model architecture and experts.
Weights use native FP8 + MXFP4 compression; this is not FP16 full precision.
Activations, caches, and communication buffers also need space within that VRAM.

The project reuses the existing winintel SM70 FP4 WMMA operator and independently
implements quantization adapters, checkpoint loading, tensor and pipeline
parallelism, decode graphs, and measurement tools, combining affordable GPUs
with fast interconnects. **Commercial capability is the goal; the candidate's
performance and quality acceptance status are documented below.**

**v0.1.0 is an independent source candidate.** Published separately from the
winintel-vllm adaptation work, this repository uses a PyTorch reference execution
path. It does not require an installed vLLM engine and is not a complete vLLM
serving distribution. Its current focus is numerical compatibility and operator
optimization; full-model throughput remains an active development target.

## Highlights

| Feature | Implementation and benefit |
| --- | --- |
| Affordable complete-model deployment | Targets DeepSeek V4.1 Flash in the 400+ GB class, using 16 V100 32GB GPUs and native compressed weights to retain the complete architecture |
| Two-node 100 Gb/s IB interconnect | RDMA and low-latency transfers reduce cross-node communication overhead and provide a fast path for TP / PP |
| Existing SM70 operator reuse | Original FP4 WMMA CUDA source, packed expert weights, MXFP4 / E8M0 scale adaptation |
| FP8 decoded-weight cache | Reuses admitted FP32 matrices, with a default 256 MiB budget per GPU and invalidation on weight/scale changes |
| Small-batch decode graphs | Captures FP8 input quantization, activation decoding, GEMM, and complete expert computation |
| Shared expert inputs | Reuses gate/up quantization and FP16 activation decoding while preserving pruning, SwiGLU, routing weights, and output quantization order |
| Exact TP greedy selection | Exchanges score/token candidates, preserving global tie rules and nonfinite-value rejection |
| Distributed execution | TP4 × PP4, stage-local loading, demand-paged Engram reads, and per-GPU cache/graph measurements |
| Auditable source | Pinned official model revision, retained licenses, original operator hash, file manifest, and reproducible benchmarks |

## Why connect the two servers with IB?

A model distributed across two servers depends on cross-node transfers for tensor
parallel communication and activation transfers between pipeline stages.
**100 Gb/s IB provides high link bandwidth, low latency, and RDMA**, reducing CPU
involvement in data transfers and interference with computation. This helps
existing V100 clusters handle complete large models.

| Link | Nominal bandwidth | Nominal bandwidth ratio of 100 Gb/s IB |
| --- | --- | --- |
| Common Gigabit Ethernet | 1 Gb/s | 100× |
| Common 10 Gigabit Ethernet | 10 Gb/s | 10× |
| 25 Gigabit Ethernet | 25 Gb/s | 4× |
| This project's IB link | 100 Gb/s | 1× |

These are link-rate comparisons, **not full-model inference speedups**. Actual
benefits depend on communication volume, parallel layout, and compute bottlenecks.
This repository has no full-model IB-versus-Ethernet A/B measurement.
100 Gigabit Ethernet offers comparable nominal bandwidth and can also support
RDMA through RoCE. During deployment, confirm IB ports are **Active / LinkUp**,
select the actual HCA, and set `NCCL_IB_DISABLE=0` and `NCCL_NET=IB`.
See the [two-node configuration](configs/two-node.env.example).

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
