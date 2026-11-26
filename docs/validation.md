# Measurements and validation / 性能与验证

作者：数萌AI服务器 (winintel) · 2026-11-26

## Real-checkpoint component measurements

| Component | Rows | N × K | Previous ms | Current ms | Speedup | Numerics |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| FP4 complete expert | 1 | 2304 × 5120 | 0.8247 | 0.7940 | 1.039× | Elementwise identical |
| FP4 complete expert | 2 | 2304 × 5120 | 0.8612 | 0.8222 | 1.047× | Elementwise identical |
| FP4 complete expert | 16 | 2304 × 5120 | 0.9627 | 0.9296 | 1.036× | Elementwise identical |
| FP8 complete shared expert | 1 | 2304 × 5120 | 1.7101 | 1.7135 | 0.998× | Elementwise identical |
| FP8 complete shared expert | 2 | 2304 × 5120 | 1.7535 | 1.7536 | 1.000× | Elementwise identical |
| FP8 complete shared expert | 16 | 2304 × 5120 | 1.8728 | 1.8746 | 0.999× | Elementwise identical |
| FP8 attention wq_a | 1 | 1280 × 5120 | 0.9537 | 0.2576 | 3.702× | Elementwise identical |
| FP8 attention wq_a | 2 | 1280 × 5120 | 0.9407 | 0.2800 | 3.359× | Elementwise identical |
| FP8 attention wq_a | 16 | 1280 × 5120 | 0.9613 | 0.3013 | 3.190× | Elementwise identical |
| FP8 attention wq_b | 1 | 32768 × 1280 | 1.0766 | 0.4336 | 2.483× | Elementwise identical |
| FP8 attention wq_b | 2 | 32768 × 1280 | 1.0858 | 0.4157 | 2.612× | Elementwise identical |
| FP8 attention wq_b | 16 | 32768 × 1280 | 1.2197 | 0.5857 | 2.082× | Elementwise identical |
| FP8 attention wkv | 1 | 512 × 5120 | 0.9521 | 0.2455 | 3.878× | Elementwise identical |
| FP8 attention wkv | 2 | 512 × 5120 | 0.9558 | 0.2642 | 3.618× | Elementwise identical |
| FP8 attention wkv | 16 | 512 × 5120 | 0.9623 | 0.2754 | 3.494× | Elementwise identical |

| Condition | Setting |
| --- | --- |
| Hardware | Tesla V100-SXM2-32GB |
| Weights | Real checkpoint, layer 0; complete gate/up/down expert FFN |
| Inputs | Seed 82; random activations and FP32 routing weights |
| Baseline | Previous shared input quantization; current shared decoding and cached FP8 graph disabled |
| Candidate | Current shared activation decoding and FP8 graph enabled |
| Timing | Three warmups; both A/B orders; 15 measurements per order; median host-to-GPU-synchronized time |
| Numerical checks | atol=rtol=0; changed inputs/routing and preserved old output checked |
| Context | Existing service resident and idle; no power cap changes |

FP8 shared experts did not improve in this round; their original complete
expert graph is retained. The 15 cases establish agreement with the preceding
implementation, not comprehensive model quality or original full-precision accuracy.
Raw module measurements: [graph-reuse-results.json](graph-reuse-results.json).

## Cache and candidate exchange

| Component | Workload | Baseline ms | Candidate ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| FP8 cache hit | M=1, N=1792, K=5120 | 2.3163 | 0.4941 | 4.687× |
| FP8 cache hit | M=16, N=1792, K=5120 | 2.3492 | 0.5162 | 4.551× |
| FP8 cache hit | M=128, N=1792, K=5120 | 2.3424 | 0.6375 | 3.674× |
| FP8 cache hit | M=512, N=1792, K=5120 | 2.3987 | 1.1033 | 2.174× |
| FP8 cache hit | M=2048, N=1792, K=5120 | 3.8143 | 3.1146 | 1.225× |
| Greedy selection | TP2, one request | 0.4142 | 0.2965 | 1.397× |
| Greedy selection | TP2, two requests | 0.4112 | 0.2954 | 1.392× |
| Greedy selection | TP2, 16 requests | 0.4075 | 0.3115 | 1.308× |

Cache measurements use synthetic quantized matrices and warmed cache hits;
FP8 relative error against reference matmul was approximately 0.00021.
Greedy measurements use a 131072-token vocabulary and exclude final projection.
At two requests the per-GPU selection payload shrinks from 524288 to 48 bytes.
NCCL exchange remains outside the local selection graph.

## Candidate full-model experiment

| Case | Input/request | C | TTFT s | Effective total prefill tok/s | Fixed-step decode/request tok/s | Fixed-step total tok/s | EOS-trimmed response total tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short baseline | 256 | 2 | 38.94 | 13.15 | 0.3226 | 0.6452 | 0.5465 |
| Short candidate | 256 | 2 | 9.83 | 52.09 | 0.4712 | 0.9423 | 0.7503 |
| 8K baseline | 8192 | 2 | 210.44 | 77.86 | 0.3307 | 0.6614 | 0.5570 |
| 8K candidate | 8192 | 2 | 198.80 | 82.41 | 0.3782 | 0.7564 | 0.6229 |

Two nodes × eight V100 32GB, TP4/PP4, layers 10/10/11/9, 128-token prefill
chunks, maximum configuration 16384, default 256 MiB FP8 cache per GPU.
Each GPU admitted eight FP8 matrices totaling 255 MiB and eight FP8 graphs.
Full-model graph=false; DSpark5 disabled. Loading took 567.44 seconds.

Fixed-step decode covers 48 steps including post-EOS steps. Both requests reached
EOS after nine generated tokens. Retrieval was 2/2 in all four non-profile cases.
Effective prefill includes scheduling and first-token overhead. A/B order was
fixed within one loading session; warmup/cache histories differ. The 14.4% 8K
fixed-step gain is an observation from this run, not a stable throughput claim.
The candidate fails the 20 token/s threshold; higher concurrency was not pursued.
Other engines' 128K/256K results are excluded from this source release.

## Reproduction

```bash
bash run_reference.sh -m pytest tests/test_kernel.py -q
bash run_reference.sh benchmarks/benchmark_winintel_v41_graph_reuse.py \
  --source . --model "$WININTEL_MODEL" --output graph-reuse.json
bash launch_node.sh 0 --model "$WININTEL_MODEL" --tp 4 \
  --layers 10,10,11,9 --cuda-graph --max-seq-len 16384 \
  --prefill-chunk 128 --suite configs/short-and-8k-c2.json \
  --output "$PWD/results.json"
```

Start node rank 1 for the full-model suite. The current release regression and
source-origin checks are recorded in [release-checks.md](release-checks.md).
