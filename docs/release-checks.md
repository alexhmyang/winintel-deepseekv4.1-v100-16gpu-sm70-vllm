# v0.1.0 release checks

作者：数萌AI服务器 (winintel) · 2026-11-26

| Check | Method | Result |
| --- | --- | --- |
| V100 regression | Isolated release directory; `bash run_reference.sh -m pytest tests/test_kernel.py -q` | **61 passed in 16.82 s** |
| Shell launchers | `bash -n run_reference.sh` and `bash -n launch_node.sh` | Passed |
| Official reference | Four file SHA256 hashes against pinned revision | Passed |
| Original WMMA source | Vendored source and existing local Qwen release SHA256 | Identical, unchanged |
| Known legacy-engine source comparison | 29 custom source files against 35 model/kernel files (1037293 bytes); 12 consecutive normalized non-comment/nonblank lines | No matching fragments |
| Release content | Every payload path/content; foreign engine dependencies, tokens/private keys, private endpoints, binaries, Python syntax | Passed |
| Configuration preservation | Four real existing launch/config files backed up separately; sanitized candidate templates and suite included | Saved |
| Existing service | HTTP health before and after release regression | 200 / 200 |
| Real-weight module evidence | Prior 15-case checkpoint graph comparison | Elementwise identical |

The source comparison uses a defined known corpus and exact normalized fragments.
It does not establish absence of every possible renamed or independently similar
implementation. The official model reference is intentionally included with its
original MIT license; the project does not claim whole-engine original authorship.

| Release regression environment | Observed value |
| --- | --- |
| GPU | Tesla V100-SXM2-32GB |
| Torch | 2.10.0+cu128 |
| CUDA | 12.8 |
| Transformers | 4.57.6 |
| NumPy | 2.5.5 |
| pytest | 9.2.0 |
| Python | Isolated Python 3.12 environment |
| FP4 extension | Existing hash-verified winintel operator, reused for regression |

Raw release regression output is in [release-regression.txt](release-regression.txt).
No production service was replaced for this source publication. Full-model
measurements and their limitations are in [validation.md](validation.md).
