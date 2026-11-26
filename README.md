# winintel-deepseekv4.1-v100-16gpu-sm70-vllm

**让 16 张 V100 继续承担新一代大模型推理。**

[English](README.en.md) · [部署指南](docs/deployment.md) ·
[性能与验证](docs/validation.md) · [来源与许可](docs/provenance.md) ·
[winintel AI](https://winintel.com)

作者：**数萌AI服务器 (winintel)** · service@winintel.com

面向 **DeepSeek V4.1 Flash、16 × V100 32GB、Volta SM70** 的开源推理研究项目。
我们复用已有 winintel SM70 FP4 WMMA 算子，独立实现量化适配、权重加载、
张量与流水线并行、Decode Graph 和测量工具，为存量 GPU 提供可复查的优化基础。

**v0.1.0 是独立源码候选版。** 本仓库从 winintel-vllm 的适配工作中单独发布，
采用 PyTorch 驱动的参考执行路径；无需安装 vLLM 引擎，也不是完整 vLLM 服务发行版。
当前重点是数值兼容和算子优化，整模型吞吐仍需继续优化。

## 项目亮点

| 亮点 | 实现与价值 |
| --- | --- |
| 复用成熟 SM70 算子 | FP4 专家使用原有 WMMA CUDA 源码，保持压缩权重；适配 MXFP4 / E8M0 分组尺度 |
| FP8 权重解码缓存 | 将准入矩阵解码为 FP32 并复用；默认每卡 256 MiB 预算，权重或尺度变化后失效 |
| 小批 Decode CUDA Graph | 捕获 FP8 输入量化、解码、GEMM 和完整专家计算，减少重复发射 |
| 专家输入复用 | gate/up 共享输入量化与 FP16 激活解码，保持裁剪、SwiGLU、路由权重和输出量化顺序 |
| 精确 TP 贪心选词 | 交换分数和 token ID 候选，保留全局同分规则及非有限值检查，缩小通信载荷 |
| 多机执行与观测 | TP4 × PP4、分阶段加载、Engram 按需读取；记录每卡缓存、Graph、TTFT 和 Decode |
| 可审计开源 | 固定官方模型版本，保留许可、原算子 SHA256、完整文件清单及复测工具 |

## 已取得的模块成绩

以下是 V100-SXM2-32GB 上的**模块测量**，不能换算为整模型 token/s。
FP8 attention 与 FP4 专家 Graph 的基线为上一版输入量化复用实现。

| 测试 | 结果 | 范围 |
| --- | --- | --- |
| FP8 缓存命中矩阵 | 1.22～4.69× | 合成量化矩阵；首次解码及未准入矩阵不享有该收益 |
| FP8 attention Decode Graph | 2.08～3.88× | 真实 checkpoint 的 wq_a / wq_b / wkv；1 / 2 / 16 行 |
| FP4 完整专家 Graph | 双请求耗时降低 4.5% | 0.8612 → 0.8222 ms，在上一版 Graph 上继续改进 |
| 两卡双请求贪心选词 | 1.392× | 0.4112 → 0.2954 ms；不含最终词表投影 |
| 真实权重数值对照 | 15 组逐元素一致 | 本轮 Graph 与激活解码复用；atol=rtol=0 |
| 发布版 V100 回归 | 61 项通过，16.82 秒 | 最终运行源码；缓存、Graph、路由和量化行为 |

完整数据及测量条件见 [验证记录](docs/validation.md)。
有约 0.6% 完整 FFN 相对误差的大批 FP4 Prefill 实验分流保持默认关闭。

## 当前整模型边界

| 项目 | 当前状态 |
| --- | --- |
| 整模型实验硬件 | 2 台 × 8 张 V100 32GB，IB 互联，共 16 卡 |
| 原生 checkpoint | FP8 普通权重 + MXFP4 专家权重；本包不含模型 |
| 候选测试上下文 / 配置 | 已测 8K 双请求；本轮最大上下文配置为 16K |
| 8K C2 固定步 Decode | 每路 0.3782 token/s，低于项目 20 token/s 门槛 |
| Graph 范围 | 线性层和专家局部 Graph；整模型 Graph 尚未实现 |
| 推测解码 / 输入 | 当前逐 token 生成；未启用 DSpark5；整模型验证为文本输入 |
| 四机四卡 | 提供布局模板；需要在实际四节点硬件上验收 |
| 服务入口 | CLI 生成及基准；HTTP 会话模块用于实验，不提供生产 OpenAI 服务 |

固定 48 步结果包含 EOS 后的步骤；本次两路到 EOS 均为 9 个 token，
EOS 截止响应总吞吐为 0.6229 token/s，检索 2/2 命中。
完整模型质量评测仍是后续验收项目。其他引擎的长上下文服务成绩不属于本仓库成绩。

## 开始使用

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
在两台机器分别执行（主节点 rank=0，第二台 rank=1）：

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

环境模板、短请求与 8K 双请求 suite、Graph A/B 开关和四机布局均已随源码保存。
主节点输出结果；不要在现有模型已占满显存的 GPU 上启动第二个整模型。

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

## 开源与后续方向

优先解决整模型主机开销、完整 Decode Graph、FP8 驻留和 PP 等待；
短上下文 C2 达到性能门槛后再开展长上下文及更高并发验证。
欢迎提供 CUDA 优化、四阶段时间线、模型评测及可复现的硬件实验。

winintel 新增代码采用 [Apache-2.0](LICENSE)；DeepSeek 官方模型代码采用
[MIT](upstream/LICENSE)。保留原代码来源及版权，不宣称整个模型和框架均为原创。
详见 [NOTICE](NOTICE) 和 [来源说明](docs/provenance.md)。
本项目使用了 AI 辅助开发、验证和文档整理。

**winintel AI · 数萌AI服务器** — [winintel.com](https://winintel.com) ·
[Bilibili](https://space.bilibili.com/429959998) ·
[YouTube](https://www.youtube.com/@winintelcom) · service@winintel.com
