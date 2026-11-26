# DeepSeek V4.1 Flash · IB 与千兆网卡部署实测 / IB versus Gigabit Ethernet

作者 / Author：数萌AI服务器 (winintel)

**256K 单请求：IB 首字时间 63.47 秒，千兆网卡 213.28 秒，首字等待缩短 70.2%。**

这组网络 A/B 测试使用独立源码版发布前的第三方推理引擎部署基线，
用于展示同一组 V100 服务器上的网络选择价值。测试日期为 2026-11-22；
本次发布整理已有测量记录。当前独立源码版的验证见 [validation.md](validation.md)。

## 测试条件

| 项目 | 配置 |
| --- | --- |
| 模型 | DeepSeek V4.1 Flash |
| 服务器 / GPU | 2 台 × 8 张 V100-SXM2 32GB，共 16 卡 |
| 并行 | TP4 × PP4，CUDA Graph |
| 对照控制 | 网络切换时保持模型、精度、KV、分批参数及请求数据一致 |
| IB 路径 | NCCL IB；物理端口 100 Gb/s、Active |
| 普通网卡路径 | 实际 1 Gb/s 以太网接口，NCCL Socket / TCP |
| 当前硬件规格 | 每台两个 100 Gb/s IB 端口，合计 200 Gb/s 标称链路带宽 |
| 计时 | 客户端首字时间、Decode 阶段和完整请求计时 |
| 请求 | 独立会话；短输入 1,024 输出 token/请求，8K / 256K 为 256 输出 token/请求 |
| 长输入轮次 | 每个场景、每种网络一轮；所有下表配对请求均完成 |

**200 Gb/s 为当前双口硬件合计规格。** 此网络 A/B 记录比较 IB 与千兆 TCP，
并非双轨聚合带宽测试。原报告记录 GPU Direct RDMA 未启用；
独立单轨 RDMA 写测试约 51.7 Gb/s，属于单独的链路测试。

## 全部可配对场景的首字对比

首字采用各轮最慢请求时间。加速比 = 千兆首字时间 ÷ IB 首字时间。

| 输入 | 并发 | IB 首字（秒） | 千兆首字（秒） | 首字加速比 |
| --- | ---: | ---: | ---: | ---: |
| 短输入 | 1 | 0.140 | 1.276 | 9.12× |
| 短输入 | 2 | 0.187 | 0.292 | 1.56× |
| 短输入 | 4 | 0.604 | 0.637 | 1.05× |
| 短输入 | 8 | 0.540 | 0.767 | 1.42× |
| 短输入 | 16 | 0.694 | 1.070 | 1.54× |
| 8K | 1 | 2.228 | 7.011 | 3.15× |
| 8K | 2 | 4.321 | 13.988 | 3.24× |
| 8K | 4 | 8.465 | 27.736 | 3.28× |
| 8K | 8 | 16.741 | 55.314 | 3.30× |
| 8K | 16 | 33.193 | 110.590 | 3.33× |
| 256K | 1 | 63.473 | 213.285 | 3.36× |
| 256K | 2 | 124.011 | 425.349 | 3.43× |
| 256K | 3 | 186.680 | 636.130 | 3.41× |
| 256K | 4 | 249.220 | 848.353 | 3.40× |

长输入场景的差异持续出现在多个并发档位；TP4 组内主要走机内互联，
IB 承担跨主机流水线阶段的激活和控制消息传输。

## Decode 阶段对比

以下为首字后生成阶段的合计吞吐，按该阶段的时间区间并集计时。

| 输入 / 并发 | IB 合计（token/s） | 千兆合计（token/s） | IB / 千兆 |
| --- | ---: | ---: | ---: |
| 短输入 / 1 | 185.37 | 144.79 | 1.28× |
| 短输入 / 16 | 892.26 | 813.82 | 1.10× |
| 8K / 1 | 101.92 | 86.26 | 1.18× |
| 8K / 2 | 85.02 | 48.00 | 1.77× |
| 8K / 4 | 87.04 | 39.97 | 2.18× |
| 8K / 8 | 93.38 | 36.80 | 2.54× |
| 256K / 1 | 70.79 | 33.51 | 2.11× |
| 256K / 2 | 7.04 | 2.31 | 3.05× |
| 256K / 3 | 5.17 | 1.70 | 3.05× |
| 256K / 4 | 4.67 | 1.52 | 3.07× |

## 数据与统计口径

[下载原始导出 CSV](ib-ethernet-measurements.csv)，包含 30 条记录、
14 组网络配对及额外的 IB 256K 高并发记录。

| CSV 字段 | 含义 |
| --- | --- |
| `max_ttft_s` | 本轮最慢请求从开始到首字的秒数 |
| `aggregate_prefill_effective_tps` | 导出表中的客户端有效输入吞吐，包含首字等待 |
| `aggregate_decode_busy_tps` | 首字后输出 token 合计 / Decode 时间区间并集 |
| `mean_request_decode_tps` | 逐会话 Decode 速度的均值 |
| `output_throughput` | 全程输出吞吐，包含输入处理与排队 |
| `cross_host_tx_gib` | 两机物理端口发送量之和；IB 计数按每单位 4 字节换算 |

CSV 保留已有导出值。其短输入有效 Prefill 字段使用旧输入计数口径；
本报告不使用该字段作对比。千兆 8K / 16 并发在中途 SSH 连接中断后自行完成，
旧导出中的阶段吞吐与流量零值是缺失字段占位，本报告只使用该行已追回的首字时间。
两机端口流量包含后台通信。

各场景测量次数有限，期间有短时独立算子验证，未建立完全独占 GPU 的条件。
短输入的小幅差异应结合这些条件解读。结果适用于本次 1GbE / NCCL Socket
与 IB 部署对照；更高速率以太网和 RoCE 应按对应配置测量。
本表是历史部署基线记录，当前独立源码版性能请查看其专门验证文档。

| 本地源记录 | SHA256 |
| --- | --- |
| 原始网络速度 CSV | `05ff19c6afaaea32fceb96878e9ecc829daaae047b0bc8bb98a2b59fba2bdfad` |
| 完整速度与网卡对照 HTML 报告 | `fcfd80a84190010ffd1e8041e721d92eedaf5487a887780b841a24cbe458110c` |

CSV 为原始导出内容的公开副本；发布文件自身的 SHA256 见仓库 `manifest.json`。
原始报告保存在项目归档中。本报告使用 AI 辅助整理。

## English

**At 256K with one request, IB reduced first-token waiting from 213.28 s to
63.47 s: 3.36× faster, or 70.2% less waiting.**

These historical A/B measurements used the third-party inference engine deployed
before this independent source release, on two servers with eight V100 32GB GPUs
each, TP4 × PP4 and CUDA Graph. The model, precision, KV settings, batching
parameters, and request data were held constant across network changes.
The comparison is NCCL IB versus physical 1GbE using NCCL Socket / TCP.

The tables above show every paired first-token result and selected aggregate
decode results. Columns read: input, concurrency, IB, Gigabit Ethernet, and
Ethernet/IB latency ratio (or IB/Ethernet throughput ratio). First-token values
are the slowest request in each round; decode throughput uses the union of
post-first-token intervals. Each long-input configuration has one round per
network. All paired requests completed.

The current hardware has two active 100G IB ports per server, providing 200 Gb/s
aggregate nominal link bandwidth. The historical A/B experiment did not measure
dual-rail aggregation. GPU Direct RDMA was disabled in that deployment; the
separate single-rail RDMA write test measured about 51.7 Gb/s.

The [CSV export](ib-ethernet-measurements.csv) preserves all 30 records.
Short-input effective-prefill values use an older input-count calculation;
they are not used in these comparisons. Zero stage-throughput/traffic fields
in the Ethernet 8K / 16-request row are missing-data placeholders; the recovered
first-token result is used here. Some brief operator validation overlapped the
experiment, so small differences should be interpreted with that condition.

These results describe this historical deployment and its 1GbE comparison.
See [current source validation](validation.md) for this repository's runtime
measurements. Source archive fingerprints are listed above; published file hashes
are in `manifest.json`. AI assistance was used to prepare this report.
