# Deployment / 部署与复现

作者：数萌AI服务器 (winintel)

## Prerequisites

| Item | Requirement |
| --- | --- |
| OS | Linux x86_64; native Windows full-model execution is not qualified |
| GPU | 16 × V100 **32GB**; two nodes × eight GPUs is the tested candidate layout |
| Network | Active/LinkUp InfiniBand; set the real HCA and bootstrap interface on each node |
| Software | Python 3.12, CUDA 12.8 toolkit, C++ compiler, uv, SM70-compatible Torch 2.10 |
| Storage | Complete local checkpoint, tokenizer, config, native quantized tensors and Engram files; sufficient RAM/NVMe for demand paging |
| Capacity | V100 16GB is not interchangeable with the tested 32GB layout |

Generic Torch wheels may omit Volta support. Install a compatible wheel and
check the build before testing. The package does not include a Torch binary.
Model configuration must match the pinned official reference. Text generation
is the qualified experimental input path; inclusion of official vision source
does not establish multimodal inference qualification.

## Environment

Follow the clone and uv steps in either README. The dependency file specifies
compatible ranges, not a fully reproduced clean-machine lockfile. Do not replace
an existing environment blindly. Check the actual GPU and FP4 storage support:

```bash
export WININTEL_REFERENCE_ENV="$PWD/.venv"
export CUDA_HOME=/usr/local/cuda-12.8
export WININTEL_NATIVE_SM70=1
bash run_reference.sh -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list()); assert hasattr(torch, "float4_e2m1fn_x2"); assert torch.cuda.get_device_capability(0) == (7, 0)'
nvcc --version
bash run_reference.sh -m pytest tests/test_kernel.py -q
```

The FP4 extension is compiled from `upstream/qwen_fp4.cu` for `sm_70`.
Compilation is part of initial setup. An optional local prebuilt extension is
accepted only with `WININTEL_FP4_PREBUILT_SHA256`; no such binary is distributed.
The normal path uses the isolated venv directly. `WININTEL_REFERENCE_SYSROOT`
is an optional loader compatibility setting for prepared older Linux hosts.

## Two-node generation

```bash
cp configs/two-node.env.example configs/cluster.env
# Edit configs/cluster.env on each node: model, venv, master, active HCA/interface.
set -a
source configs/cluster.env
set +a
bash launch_node.sh 0 --model "$WININTEL_MODEL" --tp 4 \
  --layers 10,10,11,9 --cuda-graph --max-seq-len 16384 \
  --prefill-chunk 128 --new-tokens 16 \
  --prompt "Return only the result of 17 + 25." --output "$PWD/results.json"
```

Run the same command with node rank **1** on the second node. Both nodes must
join the same rendezvous and use identical source/settings. Run processes in
your service supervisor or persistent terminal. Results are written by rank 0.
The launcher forces `NCCL_IB_DISABLE=0` and `NCCL_NET=IB`; confirm `NET/IB`
in the NCCL logs. A missing HCA is a deployment error, not a reason to silently
switch the full-model run to sockets.

## Reproduce operator and full-model measurements

```bash
bash run_reference.sh benchmarks/benchmark_winintel_v41_graph_reuse.py \
  --source . --model "$WININTEL_MODEL" --output graph-reuse.json
bash run_reference.sh -m torch.distributed.run --nproc-per-node=2 \
  benchmarks/benchmark_winintel_v41_dispatch.py \
  --source . --distributed --output greedy.json
bash launch_node.sh 0 --model "$WININTEL_MODEL" --tp 4 \
  --layers 10,10,11,9 --cuda-graph --max-seq-len 16384 \
  --prefill-chunk 128 --suite configs/short-and-8k-c2.json \
  --output "$PWD/results.json"
```

For the suite, start node rank 1 as well. Module tests need only the selected
GPU(s); a complete model needs all 16 available GPUs. The supplied suite preserves
the recorded fixed baseline-then-candidate order and omits the diagnostic profile
case. Alternate order and repeat measurements before treating A/B gains as stable.
The benchmark stops progression to higher concurrency if decode falls below
20 token/s per request.

| Switch | Default | Purpose |
| --- | --- | --- |
| `WININTEL_FP8_CACHE_MIB` | 256 | Per-GPU decoded FP8 weight budget; 0 disables cache |
| `WININTEL_FP8_GRAPH` | 1 | Cached FP8 small-batch graph |
| `WININTEL_EXPERT_SHARED_QUANT` | 1 | Shared gate/up input quantization |
| `WININTEL_EXPERT_SHARED_DECODE` | 1 | Shared FP16 activation decoding |
| `WININTEL_COMPACT_GREEDY` | 1 | Exact candidate exchange; 0 uses full logits |
| `WININTEL_FP4_PREFILL_ROWS` | 0 | Experimental prefill split remains off |
| `--cuda-graph` / `--no-cuda-graph` | enabled | Local decode graph control; not a complete-model graph |

## Four-node preparation

Use `configs/four-node.env.example`; run node ranks 0, 1, 2, 3.
Keep `--tp 4 --layers 10,10,11,9` for this candidate:

| Node | Global ranks | Stage layers |
| --- | --- | --- |
| 0 | 0–3 | 0–9 |
| 1 | 4–7 | 10–19 |
| 2 | 8–11 | 20–30 |
| 3 | 12–15 | 31–39 |

This is a migration template, not a completed four-node deployment. Check
NVLink/P2P inside each stage and IB across nodes, model/source hashes, available
RAM, NVMe, and swap. The four-node layout adds cross-node stage boundaries;
it is not assumed faster before measurement.

The candidate has been exercised with a 16K maximum configuration. Increasing
`--max-seq-len` alone does not establish qualified 128K/256K execution. First
resolve short-context C2 throughput, then qualify longer context and higher
concurrency. No GPU power cap is applied by these scripts.
