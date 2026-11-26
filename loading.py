# Author: 数萌AI服务器 (winintel)
"""Load only the native checkpoint tensors owned by one pipeline/TP rank."""

import json
from pathlib import Path

import torch
from checkpoint import Checkpoint, PagedEngramEmbedding
from kernel import _scale_float
from torch import nn


def model_args(official, root, batch_size, sequence_length):
    config = json.loads((Path(root) / "config.json").read_text())
    text = config["text_config"]
    rename = {
        "hidden_size": "dim",
        "moe_intermediate_size": "moe_inter_dim",
        "num_hidden_layers": "n_layers",
        "num_attention_heads": "n_heads",
        "qk_rope_head_dim": "rope_head_dim",
        "rms_norm_eps": "norm_eps",
        "num_experts_per_tok": "n_activated_experts",
        "scoring_func": "score_func",
        "routed_scaling_factor": "route_scale",
        "sliding_window": "window_size",
        "kv_source_layer_ids": "kv_source_layers",
        "index_source_layer_ids": "index_source_layers",
        "candidate_source_layer_id": "candidate_source_layer",
        "engram_pad_token_id": "engram_pad_id",
    }
    fields = official.ModelArgs.__dataclass_fields__
    values = {}
    for key, value in text.items():
        key = rename.get(key, key)
        if key in fields:
            values[key] = value
    rope = text["rope_scaling"]
    values.update(
        original_seq_len=rope["original_max_position_embeddings"],
        rope_factor=rope["factor"],
        beta_fast=rope["beta_fast"],
        beta_slow=rope["beta_slow"],
        max_batch_size=batch_size,
        max_seq_len=sequence_length,
        dtype="fp8",
        expert_dtype="fp4",
        temperature=0,
        vision_n_layers=0,
        n_mtp_layers=0,
        dspark_block_size=0,
    )
    return official.ModelArgs(**values)


def dequant_fp8_weight(weight, scales):
    block = 32
    rows = torch.arange(weight.size(0)) // block
    return weight.float() * _scale_float(scales)[rows].repeat_interleave(block, -1)


def load_parameters(official, module, prefix, checkpoint, device, rank, tp_size):
    """Select column/row/vocab shards and preserve native quantized storage."""
    for path, child in module.named_modules():
        name = prefix + ("." + path if path else "")
        column = isinstance(
            child,
            (
                official.ColumnParallelLinear,
                official.ParallelEmbedding,
                official.ParallelHead,
            ),
        )
        row = isinstance(child, official.RowParallelLinear)
        for key, expected in list(child.named_parameters(recurse=False)):
            tensor_name = name + "." + key
            source = checkpoint.file_for(tensor_name)
            shard_axis = None
            if column and key in ("weight", "scale"):
                shard_axis = 0
            elif row and key in ("weight", "scale"):
                shard_axis = 1
            elif isinstance(child, official.Attention) and key == "attn_sink":
                shard_axis = 0
            if shard_axis is None or tp_size == 1:
                value = source.read_tensor(tensor_name)
            else:
                value = source.read_partition(tensor_name, rank, tp_size, shard_axis)
                value = value.view(source.spec(tensor_name)[1])
            if expected.dtype == torch.float4_e2m1fn_x2:
                value = value.view(expected.dtype)
            elif expected.dtype not in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                if value.dtype == torch.float8_e4m3fn:
                    scales = checkpoint.tensor(name + ".scale")
                    if shard_axis is not None and tp_size > 1:
                        scales = (
                            checkpoint.file_for(name + ".scale")
                            .read_partition(name + ".scale", rank, tp_size, shard_axis)
                            .view(torch.float8_e8m0fnu)
                        )
                    value = dequant_fp8_weight(value, scales)
                dtype = (
                    torch.float16
                    if expected.dtype == torch.bfloat16
                    else expected.dtype
                )
                value = value.to(dtype)
            if value.shape != expected.shape:
                raise ValueError(
                    f"{tensor_name}: checkpoint {value.shape} != {expected.shape}"
                )
            child._parameters[key] = nn.Parameter(value.to(device), requires_grad=False)
        if isinstance(child, official.Linear) and child.scale is not None:
            child.weight.scale = child.scale
            if child.weight.dtype == torch.float4_e2m1fn_x2:
                child.weight._winintel_native = True
            elif child.weight.dtype == torch.float8_e4m3fn:
                from native_ops import enabled, install_fp8_linear, prepare_fp8

                if enabled():
                    prepare_fp8(child.weight, child.scale)
                    install_fp8_linear(official)


def materialize_buffers(official, module, args, device):
    for child in module.modules():
        for name, value in list(child.named_buffers(recurse=False)):
            if name == "freqs_cis":
                ratio = child.compress_ratio
                value = official.precompute_freqs_cis.__wrapped__(
                    args.rope_head_dim,
                    args.max_seq_len,
                    args.original_seq_len if ratio else 0,
                    args.compress_rope_theta if ratio else args.rope_theta,
                    args.rope_factor,
                    args.beta_fast,
                    args.beta_slow,
                ).to(device)
            else:
                dtype = torch.float16 if value.dtype == torch.bfloat16 else value.dtype
                value = torch.full(
                    value.shape,
                    -torch.inf if name == "score_state" else 0,
                    dtype=dtype,
                    device=device,
                )
            child._buffers[name] = value


class Stage(nn.Module):
    """Own one contiguous group of backbone layers and its local TP tensors."""

    def __init__(
        self,
        official,
        args,
        root,
        start,
        stop,
        stage,
        stages,
        rank,
        tp_size,
        group,
        tokenizer,
        device="cuda",
    ):
        super().__init__()
        self.official, self.args = official, args
        self.start, self.stop, self.stage, self.stages = start, stop, stage, stages
        self.device = torch.device(device)
        self.checkpoint = Checkpoint(root)
        layout = official.EngramLayout.from_args(args)
        # Hashing is deterministic and inexpensive compared with table storage.
        self.hash_state = official.NgramHashState(args, layout, tokenizer).to(device)
        self.layers = nn.ModuleList()
        self.embed, self.norm, self.head = None, None, None
        with torch.device("meta"):
            if stage == 0:
                self.embed = official.ParallelEmbedding(args.vocab_size, args.dim)
            if stage == stages - 1:
                self.norm = official.RMSNorm(args.dim, args.norm_eps)
                self.head = official.ParallelHead(
                    args.vocab_size, args.dim, args.norm_eps, args.hc_eps
                )
            for layer_id in range(start, stop):
                self.layers.append(official.Block(layer_id, args, layout))
        for layer in self.layers:
            prefix = f"layers.{layer.layer_id}"
            if layer.engram is not None:
                embed_prefix = prefix + ".engram.embed"
                table_index = layout.layer_ids.index(layer.layer_id)
                shard = f"model-{47 + table_index:05d}-of-00048.safetensors"
                names = [embed_prefix + ".weight", embed_prefix + ".scale"]
                self.checkpoint.add_original_mirror_shard(shard, names)
                layer.engram.embed = PagedEngramEmbedding(
                    self.checkpoint, embed_prefix, device, rank, tp_size, group
                )
            load_parameters(
                official, layer, prefix, self.checkpoint, device, rank, tp_size
            )
            materialize_buffers(official, layer, args, device)
            from expert_graph import install as install_expert_graph
            from precision import install as install_precision
            from prefill import install

            install(official, layer.attn)
            install_precision(official, layer)
            from native_ops import enabled

            if enabled():
                from native_moe import install as install_native_moe

                install_native_moe(official, layer.ffn)
                import os

                if os.environ.get("WININTEL_NATIVE_HC", "0") == "1":
                    from native_hc import install as install_native_hc

                    install_native_hc(official, layer)
            else:
                for expert in layer.ffn.experts:
                    if expert is not None:
                        install_expert_graph(expert)
            install_expert_graph(layer.ffn.shared_experts)
            print(f"LOADED stage={stage} tp={rank} layer={layer.layer_id}", flush=True)
        for name in ("embed", "norm", "head"):
            module = getattr(self, name)
            if module is not None:
                load_parameters(
                    official, module, name, self.checkpoint, device, rank, tp_size
                )

    def reset_request(self, batch_size, sequence_length):
        """Allocate only this case's caches, retaining loaded checkpoint weights."""
        from request_cache import reset

        reset(self, batch_size, sequence_length)

    @torch.inference_mode()
    def forward(
        self, tokens, start_pos, boundary=None, greedy=False, greedy_graph=True
    ):
        from pipeline import capture_boundary

        official = self.official
        if self.stage == 0:
            official.shared_attn = official.SharedAttentionRuntime()
            hidden = (
                self.embed(tokens)
                .float()
                .unsqueeze(2)
                .repeat(1, 1, self.args.hc_mult, 1)
            )
            pre_mix = official.make_identity_pre_mix(hidden, self.args.hc_mult)
        else:
            boundary.install_attention(official.shared_attn)
            hidden, pre_mix = boundary.hidden, boundary.pre_mix
        hashes = self.hash_state(tokens, start_pos)
        for layer in self.layers:
            if layer.engram is not None:
                hidden = layer.engram(
                    hidden, hashes[:, :, layer.engram.layer_hash_index, :]
                )
            hidden, pre_mix = layer(hidden, start_pos, pre_mix, None)
        if self.stage == self.stages - 1:
            hidden = self.layers[-1].hc_pre(hidden, pre_mix)
            normalized = self.norm(hidden)
            if greedy:
                from greedy import next_token

                return next_token(
                    self.head, normalized, official.dist, graph=greedy_graph
                )
            return self.head(normalized)
        return capture_boundary(
            official, self.args, self.stop, hidden, pre_mix, None, [], start_pos
        )
