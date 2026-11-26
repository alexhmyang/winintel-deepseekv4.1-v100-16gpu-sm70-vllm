# Author: 数萌AI服务器 (winintel)
"""Validate one real FP4 expert and optional original Engram rows on a V100."""

import argparse
import json
from pathlib import Path

import torch
from checkpoint import Checkpoint, PagedEngramEmbedding
from kernel import act_quant, fp4_gemm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--expert", default="layers.0.ffn.experts.0.w1")
    parser.add_argument("--engram-prefix")
    parser.add_argument("--original-mirror-shard")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = Checkpoint(args.model)
    weight = checkpoint.prepare_fp4(args.expert)
    torch.manual_seed(41)
    x = torch.randn(3, weight.shape[1] * 2, device="cuda") * 0.1
    a, scales = act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    prepared = weight._winintel_fp4
    del weight._winintel_fp4
    expected = fp4_gemm(a, scales, weight, weight.scale, act_block_size=32)
    weight._winintel_fp4 = prepared
    actual = fp4_gemm(a, scales, weight, weight.scale, act_block_size=32)
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)
    error = (actual.float() - expected.float()).abs()
    relative = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(
        expected.float()
    ).clamp_min(1e-20)
    result = {
        "author": "数萌AI服务器 (winintel)",
        "expert": args.expert,
        "packed_shape": list(weight.shape),
        "activation_shape": list(x.shape),
        "max_absolute_error": error.max().item(),
        "relative_l2_error": relative.item(),
        "finite": torch.isfinite(actual).all().item(),
        "checkpoint_expert_native_vs_reference": "passed",
        "full_checkpoint_generation_qualified": False,
        "engram": None,
    }
    if args.engram_prefix:
        if args.original_mirror_shard:
            checkpoint.add_original_mirror_shard(
                args.original_mirror_shard,
                [args.engram_prefix + ".weight", args.engram_prefix + ".scale"],
            )
        embedding = PagedEngramEmbedding(checkpoint, args.engram_prefix)
        rows = torch.tensor(
            [0, 1, embedding.num_embeddings // 2, embedding.num_embeddings - 1, 0]
        )
        values = embedding(rows)
        assert torch.isfinite(values).all()
        torch.testing.assert_close(values[0], values[-1], rtol=0, atol=0)
        result["engram"] = {
            "prefix": args.engram_prefix,
            "table_rows": embedding.num_embeddings,
            "rows": rows.tolist(),
            "finite": True,
            "original_fp8_storage": True,
            "full_table_validation": False,
        }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
