<div align="center">

# winintel-deepseekv4.1-v100-16gpu-sm70-vllm

### 🚀 winintel AI · 用 V100 开启完整 DeepSeek 大模型体验

**💎 让价格亲民的 V100，承载 400 多 GB 级的满血大模型。**

**16 张 V100 · 双机 100 Gb/s IB · FP4 WMMA · CUDA Graph**

[🌐 官网](https://winintel.com) · [English](README.en.md) ·
[🛠️ 部署指南](docs/deployment.md) · [📊 技术验证](docs/validation.md)

[📦 下载 v0.1.0](https://github.com/alexhmyang/winintel-deepseekv4.1-v100-16gpu-sm70-vllm/releases/tag/v0.1.0) ·
[⭐ Star 支持项目](https://github.com/alexhmyang/winintel-deepseekv4.1-v100-16gpu-sm70-vllm) ·
[🔔 关注更新](https://github.com/alexhmyang/winintel-deepseekv4.1-v100-16gpu-sm70-vllm/subscription)

[🎬 Bilibili](https://space.bilibili.com/429959998) ·
[▶️ YouTube](https://www.youtube.com/@winintelcom)

**v0.1.0 源码试用版 · DeepSeek V4.1 Flash · Volta / SM70**

</div>

**把存量 V100 服务器，变成探索完整 DeepSeek 大模型的本地 AI 平台。**
winintel AI 面向 **DeepSeek V4.1 Flash 这样的 400 多 GB 级完整大模型**，
将低成本 GPU、压缩权重与高速 IB 互联结合，以商用任务所需的推理、编程和知识能力
为目标，让更多企业和开发者参与本地大模型部署。

只需围绕 **2 台服务器 × 每台 8 张 V100 32GB** 构建集群，通过
**100 Gb/s InfiniBand** 连接两台机器，即可按本项目指南开展试用。
16 张卡合计提供 **512 GB 物理显存**；模型保留完整架构与专家，
使用官方原生 **FP8 + MXFP4** 压缩权重。

- 💎 **盘活已有硬件：** 为价格亲民的 V100 提供 SM70 大模型适配，降低探索完整模型的硬件门槛。
- 🧠 **完整模型路线：** 保留官方模型架构与专家，围绕 DeepSeek 的推理、编程和知识能力开展本地应用探索。
- ⚡ **高速双机互联：** 100 Gb/s IB、低延迟与 RDMA，为跨服务器计算提供高速数据通道。
- 🏎️ **专门面向 V100 优化：** FP4 WMMA、FP8 权重解码缓存和 Decode CUDA Graph，减少重复计算与 GPU 调度开销。
- 🔒 **数据留在本地：** 推理与分词读取本地模型文件，便于在自有服务器上掌握数据与部署环境。
- 🛠️ **便于下载试用：** 源码包、双机配置、部署指南与测试工具一并提供。

## 项目亮点

| 亮点 | 实现与价值 |
| --- | --- |
| 低成本完整模型部署 | 面向 400 多 GB 级 DeepSeek V4.1 Flash，利用 16 张 V100 32GB 和原生压缩权重承载完整模型架构 |
| 双机 100 Gb/s IB 互联 | RDMA 与低延迟传输减少跨机通信开销，为 TP / PP 并行提供高速数据通道 |
| 复用成熟 SM70 算子 | FP4 专家使用原有 WMMA CUDA 源码，保持压缩权重；适配 MXFP4 / E8M0 分组尺度 |
| FP8 权重解码缓存 | 复用已解码矩阵，减少重复权重转换；缓存预算可配置 |
| 小批 Decode CUDA Graph | 捕获 FP8 输入量化、解码、GEMM 和完整专家计算，减少重复发射 |
| 专家输入复用 | 多个专家计算步骤共享输入处理，减少重复工作并保持数值顺序 |
| 精确 TP 贪心选词 | 交换精简的选词候选，缩小多卡通信载荷并保留选词规则 |
| 多机执行与观测 | TP4 × PP4、分阶段加载、Engram 按需读取；记录每卡缓存、Graph、TTFT 和 Decode |
| 可审计开源 | 固定官方模型版本，保留许可、原算子 SHA256、完整文件清单及复测工具 |

## 为什么双机要用 IB

模型分布在两台服务器上，跨机传输会影响张量并行通信与流水线阶段间的激活传递。
**100 Gb/s IB 的价值是更高的链路带宽、低延迟和 RDMA**：数据传输减少 CPU
参与，降低通信对计算的干扰，让存量 V100 集群更适合运行完整大模型。

| 链路 | 标称带宽 | 100 Gb/s IB 的标称带宽比 |
| --- | --- | --- |
| 常见千兆以太网 | 1 Gb/s | 100× |
| 常见万兆以太网 | 10 Gb/s | 10× |
| 25G 以太网 | 25 Gb/s | 4× |
| 本项目 IB 链路 | 100 Gb/s | 1× |

表中比较的是**标称链路带宽**。IB 的高速传输与 RDMA 配合 TP / PP 并行，
为两台服务器上的 GPU 协作提供通信基础。
按 [双机配置](configs/two-node.env.example) 和 [部署指南](docs/deployment.md)
完成 HCA 与 NCCL 设置，即可使用 IB 通信路径。

## V100 算子优化实测

以下为 **V100-SXM2-32GB 上的算子与通信模块实测**。
FP8 attention 与 FP4 专家 Graph 对比上一版输入量化复用实现。

| 测试 | 结果 | 范围 |
| --- | --- | --- |
| FP8 缓存命中矩阵 | 1.22～4.69× | 合成量化矩阵，缓存命中场景 |
| FP8 attention Decode Graph | 2.08～3.88× | 真实 checkpoint 的 wq_a / wq_b / wkv；1 / 2 / 16 行 |
| FP4 完整专家 Graph | 双请求耗时降低 4.5% | 0.8612 → 0.8222 ms，在上一版 Graph 上继续改进 |
| 两卡双请求贪心选词 | 1.392× | 候选交换与选词：0.4112 → 0.2954 ms |
| 真实权重数值对照 | 15 组逐元素一致 | 本轮 Graph 与激活解码复用；atol=rtol=0 |
| 发布版 V100 回归 | 61 项通过，16.82 秒 | 最终运行源码；缓存、Graph、路由和量化行为 |

完整测量条件、整模型运行记录和版本说明见 [验证记录](docs/validation.md)。

## 下载与开始试用

从 [GitHub Releases](https://github.com/alexhmyang/winintel-deepseekv4.1-v100-16gpu-sm70-vllm/releases/tag/v0.1.0)
下载源码包，或使用下面的 Git 命令获取最新源码。
本版提供 **PyTorch 驱动的独立执行路径、CLI 文本生成和基准工具**。
模型权重单独准备；安装步骤见 [部署指南](docs/deployment.md)。

使用 **Linux、CUDA 12.8、Python 3.12、支持 SM70 的 PyTorch 2.10**。
先准备模型文件和兼容 Torch wheel，再按 [部署指南](docs/deployment.md) 安装。
依赖下载使用国内镜像；模型推理和分词使用本地文件。

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

所有节点部署相同目录、依赖及本地模型。复制
[双机环境模板](configs/two-node.env.example)，填写真实 HCA、网卡和主节点地址。
在两台机器分别执行：主节点使用下面的 rank `0`，第二台将 `launch_node.sh 0`
改为 `launch_node.sh 1`。

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

环境模板、短请求与 8K 双请求 suite、Graph A/B 开关均已随源码保存。
运行后在主节点查看生成结果，使用自己的文本任务体验模型。

## 代码导览

| 目录 / 文件 | 用途 |
| --- | --- |
| `upstream/` | 固定版本的官方模型定义、原 SM70 FP4 算子及各自许可证 |
| `kernel.py`、`native_ops.py`、`qwen_adapter.py` | 参考数值路径、FP8 缓存和 FP4 WMMA 适配 |
| `expert_graph.py`、`greedy.py` | 专家 Graph 和精确 TP 候选交换 |
| `checkpoint.py`、`loading.py`、`distributed.py` | 原生权重读取、分片、Engram 和 TP/PP |
| `generate.py`、`benchmark.py`、`benchmarks/` | 生成、整模型基准及模块测量 |
| `configs/`、`tests/`、`docs/` | 可复现配置、行为回归和验证说明 |
| `provenance.json`、`manifest.json` | 代码来源和 SHA256 文件清单 |

## 一起让更多 V100 发挥价值

欢迎下载试用，分享硬件配置、部署体验和可复现的测量结果。

- ⭐ **Star 项目：** 帮助更多 V100 用户发现这条完整大模型部署路线。
- 🔔 **Watch → Custom → Releases：** 接收版本更新通知。
- 🐛 [提交反馈](https://github.com/alexhmyang/winintel-deepseekv4.1-v100-16gpu-sm70-vllm/issues)：附上硬件、配置和运行日志，方便共同改进。
- 🎬 关注 Bilibili 与 YouTube，获取部署演示和技术分享。

winintel 新增代码采用 [Apache-2.0](LICENSE)；DeepSeek 官方模型代码采用
[MIT](upstream/LICENSE)。保留各组件来源与版权。
详见 [NOTICE](NOTICE) 和 [来源说明](docs/provenance.md)。
本项目使用了 AI 辅助开发、验证和文档整理。

**winintel AI · 数萌AI服务器** — [winintel.com](https://winintel.com) ·
[Bilibili](https://space.bilibili.com/429959998) ·
[YouTube](https://www.youtube.com/@winintelcom) · service@winintel.com
