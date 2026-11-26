# Changelog

作者：数萌AI服务器 (winintel)

## v0.1.0 — 2026-11-26

Initial independent source candidate for DeepSeek V4.1 on 16 V100 32GB GPUs.

- Reuse the original SM70 FP4 WMMA source with MXFP4/E8M0 scale adaptation.
- Include bounded FP8 decoded-weight caching and small-batch decode graphs.
- Reuse expert gate/up input quantization and FP16 activation decoding.
- Preserve exact TP greedy selection with compact candidate exchange.
- Publish checkpoint loading, Engram paging, TP4/PP4 execution, and benchmarks.
- Add Chinese/English READMEs, two/four-node templates, provenance and manifest.
- Validate the release runtime: 61 V100 regression tests passed.

Full-model throughput remains below the project threshold. Complete-model
graphs, speculative decoding, production serving, and four-node acceptance
remain development work. See docs/validation.md for measured boundaries.
