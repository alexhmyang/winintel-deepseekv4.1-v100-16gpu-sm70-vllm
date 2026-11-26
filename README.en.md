<div align="center">

# winintel-deepseekv4.1-v100-16gpu-sm70-vllm

### 🚀 winintel AI · Explore complete DeepSeek models on V100

**💎 Put affordable V100 GPUs to work on complete models in the 400+ GB class.**

**16 V100 GPUs · Two-node 100 Gb/s IB · FP4 WMMA · CUDA Graph**

[🌐 Website](https://winintel.com) · [中文](README.md) ·
[🛠️ Deployment](docs/deployment.md) · [📊 Technical validation](docs/validation.md)

[📦 Download v0.1.0](https://github.com/alexhmyang/winintel-deepseekv4.1-v100-16gpu-sm70-vllm/releases/tag/v0.1.0) ·
[⭐ Star this project](https://github.com/alexhmyang/winintel-deepseekv4.1-v100-16gpu-sm70-vllm) ·
[🔔 Watch updates](https://github.com/alexhmyang/winintel-deepseekv4.1-v100-16gpu-sm70-vllm/subscription)

[🎬 Bilibili](https://space.bilibili.com/429959998) ·
[▶️ YouTube](https://www.youtube.com/@winintelcom)

**v0.1.0 source preview · DeepSeek V4.1 Flash · Volta / SM70**

</div>

**Turn existing V100 servers into a local platform for exploring complete
DeepSeek models.** winintel AI combines affordable GPUs, compressed weights,
and fast IB interconnects for **complete models in the 400+ GB class, such as
DeepSeek V4.1 Flash**. The goal is reasoning, coding, and knowledge capabilities
for commercial tasks, making local large-model deployment accessible to more
businesses and developers.

Build around **two servers with eight V100 32GB GPUs each**, connected by
**100 Gb/s InfiniBand**, and follow our guide to start your trial.
The 16 GPUs provide **512 GB of aggregate physical VRAM**. The model retains
its complete architecture and experts with official native **FP8 + MXFP4**
compressed weights.

- 💎 **More value from existing hardware:** SM70 adaptations for affordable V100 GPUs lower the hardware barrier to exploring complete models.
- 🧠 **Complete-model approach:** Retains the official architecture and experts for local exploration of DeepSeek reasoning, coding, and knowledge tasks.
- ⚡ **Fast two-server communication:** 100 Gb/s IB, low latency, and RDMA provide a fast data path between servers.
- 🏎️ **Optimized for V100:** FP4 WMMA, FP8 decoded-weight caching, and decode CUDA graphs reduce repeated work and GPU scheduling overhead.
- 🔒 **Keep data local:** Inference and tokenization use local model files on infrastructure you control.
- 🛠️ **Ready to explore:** Source packages, two-node settings, deployment instructions, and test tools are included.

## Highlights

| Feature | Implementation and benefit |
| --- | --- |
| Affordable complete-model deployment | Targets DeepSeek V4.1 Flash in the 400+ GB class, using 16 V100 32GB GPUs and native compressed weights to retain the complete architecture |
| Two-node 100 Gb/s IB interconnect | RDMA and low-latency transfers reduce cross-node communication overhead and provide a fast path for TP / PP |
| Existing SM70 operator reuse | Original FP4 WMMA CUDA source, packed expert weights, MXFP4 / E8M0 scale adaptation |
| FP8 decoded-weight cache | Reuses decoded matrices to reduce repeated weight conversion, with a configurable cache budget |
| Small-batch decode graphs | Captures FP8 input quantization, activation decoding, GEMM, and complete expert computation |
| Shared expert inputs | Shares input processing across expert computation steps, reducing repeated work while preserving numerical order |
| Exact TP greedy selection | Exchanges compact token candidates to reduce communication payloads while retaining selection rules |
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

The table compares **nominal link bandwidth**. Fast IB transfers and RDMA
provide the communication foundation for TP / PP across the two servers.
Follow the [two-node configuration](configs/two-node.env.example) and
[deployment guide](docs/deployment.md) to configure the HCA and NCCL IB path.

## Measured V100 operator improvements

These are **operator and communication component measurements on
V100-SXM2-32GB**. Attention and expert graphs are compared with the previous
input-reuse implementation.

| Measurement | Result | Scope |
| --- | --- | --- |
| FP8 matrix cache hits | 1.22–4.69× | Synthetic quantized matrices with cache hits |
| FP8 attention decode graphs | 2.08–3.88× | Real checkpoint wq_a / wq_b / wkv; 1 / 2 / 16 rows |
| Complete FP4 expert graph | 4.5% less time for two requests | 0.8612 → 0.8222 ms, improving on the previous graph |
| Two-GPU, two-request greedy selection | 1.392× | Candidate exchange and selection: 0.4112 → 0.2954 ms |
| Real-weight numerical comparison | 15 elementwise-identical cases | Current graph/input-decoding changes; atol=rtol=0 |
| Release V100 regression | 61 passed in 16.82 seconds | Final runtime source; cache, graph, routing, and quantization behavior |

See [validation details](docs/validation.md) for measurement conditions,
full-model execution records, and version information.

## Download and start your trial

Download the source package from
[GitHub Releases](https://github.com/alexhmyang/winintel-deepseekv4.1-v100-16gpu-sm70-vllm/releases/tag/v0.1.0),
or use the Git command below for the latest source.
This version includes a **standalone PyTorch execution path, CLI text generation,
and benchmark tools**. Prepare model weights separately and follow the
[deployment guide](docs/deployment.md).

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
Run on both nodes: use rank `0` below on the master, and change
`launch_node.sh 0` to `launch_node.sh 1` on the second server.

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

Environment templates, short/8K C2 suites, and graph A/B controls are included.
Read the generated results on the master and try your own text tasks.

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

## Help more V100 owners get started

Download, try the project, and share your hardware configuration, deployment
experience, and reproducible measurements.

- ⭐ **Star the project:** Help more V100 owners discover complete-model deployment on existing hardware.
- 🔔 **Watch → Custom → Releases:** Receive version updates.
- 🐛 [Share feedback](https://github.com/alexhmyang/winintel-deepseekv4.1-v100-16gpu-sm70-vllm/issues) with your hardware, configuration, and logs.
- 🎬 Follow Bilibili and YouTube for deployment demonstrations and technical videos.

winintel additions are [Apache-2.0](LICENSE); official DeepSeek model code is
[MIT](upstream/LICENSE). Component provenance and copyright notices are retained.
See [NOTICE](NOTICE) and
[provenance](docs/provenance.md). AI assistance was used for development,
validation, and documentation.

**winintel AI** — [winintel.com](https://winintel.com) ·
[Bilibili](https://space.bilibili.com/429959998) ·
[YouTube](https://www.youtube.com/@winintelcom) · service@winintel.com
