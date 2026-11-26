# Source provenance / 源码来源

作者：数萌AI服务器 (winintel)

This repository publishes a scoped reference runtime, rather than the contents
or history of an existing engine checkout.

| Component | Origin | License / verification |
| --- | --- | --- |
| Runtime adapters, checkpoint/TP/PP, cache, graphs, tests | winintel implementation | Apache-2.0; file hashes in `manifest.json` |
| `upstream/model.py`, `engram.py`, `vision.py`, `image_processor.py` | Official DeepSeek reference | MIT; pinned revision and hashes in `provenance.json` |
| `upstream/qwen_fp4.cu` | Existing local winintel SM70 Qwen FP4 operator | Apache-2.0; unchanged SHA256 below |
| Torch / CUDA / dependencies | Separately installed | Their respective licenses; no binary redistribution here |
| Model weights | Separately prepared local checkpoint | Model-specific license; not included |

Official reference: `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`.
Existing FP4 source SHA256:
`721d8d0c36d572ce8c8f864f002edc7ab4cf16270c53cfb78a2f28d62cec14ce`.

发布检查包括：逐文件内容、路径与依赖扫描，官方模型及原算子哈希核验，
已知既有引擎实现片段对照，以及敏感信息和二进制检查。
不通过简单改名或删除署名来改变代码来源。
保留引擎部署脚本、引擎源码、wheel、预编译库、登录配置和模型权重不在发布范围。

A scan establishes the checked files and comparison corpus; it is not a
legal guarantee of originality. The official model and external libraries
retain their authorship. AI assistance was used in this work.

Verify an untouched source checkout or extracted release with:

```bash
uv run --offline --no-project --python .venv/bin/python python verify_release.py
```

`manifest.json` lists every distributed file except itself. Hash changes require
an explicit manifest update before a new release. Git metadata and documented
local runtime artifacts are excluded by the verifier.
