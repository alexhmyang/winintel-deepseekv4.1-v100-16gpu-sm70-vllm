# Author: 数萌AI服务器 (winintel)
"""Bounded text prefill preserving the reference's causal compression groups."""

from types import MethodType

import torch


def install(official, attention):
    """Extend one loaded attention module without changing pinned source files."""

    def window(self, x, frequencies, position):
        batch, length, _ = x.shape
        width = self.window_size
        current = self.kv_norm(self.wkv(x))
        official.apply_rotary_emb(current[..., -self.rope_head_dim :], frequencies)
        official.act_quant(
            current,
            official.fp8_block_size,
            official.scale_fmt,
            official.scale_dtype,
            True,
        )
        previous = min(position, width - 1)
        positions = torch.arange(position - previous, position, device=x.device)
        old = self.window_kv_cache[:batch, positions % width]
        values = torch.cat((old, current), dim=1)
        queries = torch.arange(length, device=x.device) + previous
        indices = queries[:, None] - width + 1 + torch.arange(width, device=x.device)
        indices = indices.masked_fill(indices < 0, -1)
        indices = indices.expand(batch, -1, -1).int()
        tail = min(length, width)
        slots = (
            torch.arange(position + length - tail, position + length, device=x.device)
            % width
        )
        self.window_kv_cache[:batch, slots] = current[:, -tail:]
        return values, indices

    def compress(self, x, position):
        batch, length, _ = x.shape
        ratio = self.compress_ratio
        if ratio == 1:
            return self.norm(self.wkv(x))
        values, scores = self.wkv(x.float()), self.wgate(x.float())
        pending = position % ratio
        if pending:
            values = torch.cat((self.kv_state[:batch, :pending], values), dim=1)
            scores = torch.cat((self.score_state[:batch, :pending], scores), dim=1)
        completed = (pending + length) // ratio * ratio
        remainder = pending + length - completed
        if remainder:
            self.kv_state[:batch, :remainder] = values[:, completed:]
            self.score_state[:batch, :remainder] = scores[:, completed:]
        if not completed:
            return None
        values = values[:, :completed].unflatten(1, (-1, ratio))
        scores = scores[:, :completed].unflatten(1, (-1, ratio))
        return self.norm((values * scores.softmax(2)).sum(2).to(x.dtype))

    def index(self, x, qr, latent, position, offset):
        batch, length, _ = x.shape
        ratio, rd = self.compress_ratio, self.rope_head_dim
        end = position + length
        first = position // ratio
        if self.owns_k and latent is not None:
            key = self.k_norm(self.wk(latent))
            frequencies = self.freqs_cis[
                first * ratio : (first + key.size(1)) * ratio : ratio
            ]
            official.apply_rotary_emb(key[..., -rd:], frequencies)
            official.fp4_act_quant(key, official.fp4_block_size, True)
            self.k_cache[:batch, first : first + key.size(1)] = key
        if self.owns_k:
            official.shared_attn.index_k = self.k_cache
        queries = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        official.apply_rotary_emb(queries[..., -rd:], self.freqs_cis[position:end])
        official.fp4_act_quant(queries, official.fp4_block_size, True)
        keys = official.shared_attn.index_k[:batch, : end // ratio]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        selected, candidates = [], []
        workspace = batch * self.n_local_heads * keys.size(1) * queries.element_size()
        query_chunk = max(1, min(16, 128 * 1024**2 // max(1, workspace)))
        for begin in range(0, length, query_chunk):
            stop = min(begin + query_chunk, length)
            scores = torch.einsum("bshd,btd->bsht", queries[:, begin:stop], keys)
            scores = (scores.relu_() * weights[:, begin:stop, :, None]).sum(2)
            if official.world_size > 1:
                official.dist.all_reduce(scores)
            visible = (
                torch.arange(position + begin + 1, position + stop + 1, device=x.device)
                // ratio
            )[:, None]
            scores.masked_fill_(
                torch.arange(keys.size(1), device=x.device) >= visible, -torch.inf
            )
            if self.is_candidate_source:
                candidates.append(
                    official.select_candidate_blocks(
                        scores,
                        visible,
                        self.candidate_topk_blocks,
                        self.candidate_block_size,
                    )
                )
            elif self.uses_candidates:
                scores.masked_fill_(
                    ~official.shared_attn.candidates[:, begin:stop], -torch.inf
                )
            count = min(self.index_topk, keys.size(1))
            indices = scores.topk(count, dim=-1, sorted=False).indices.sort(-1).values
            selected.append(torch.where(indices < visible, indices + offset, -1))
        if candidates:
            official.shared_attn.candidates = torch.cat(candidates, dim=1)
        return torch.cat(selected, dim=1).int()

    def compressed_kv(self, x, qr, position, offset):
        batch, length, _ = x.shape
        ratio = self.compress_ratio
        valid = (position + length) // ratio
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, position)
            official.shared_attn.compress_kv = self.compress_kv_cache
        indices = self._compress_topk_idxs(x, qr, latent, position, offset, valid)
        if latent is not None:
            first = position // ratio
            frequencies = self.freqs_cis[
                first * ratio : (first + latent.size(1)) * ratio : ratio
            ]
            official.apply_rotary_emb(latent[..., -self.rope_head_dim :], frequencies)
            official.fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
            self.compress_kv_cache[:batch, first : first + latent.size(1)] = latent
        return official.shared_attn.compress_kv[:batch, :valid], indices

    def forward(self, x, position):
        """Keep compressed and sliding KV separate instead of copying full caches."""
        batch, length, _ = x.shape
        frequencies = self.freqs_cis[position : position + length]
        rd = self.rope_head_dim
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        official.apply_rotary_emb(q[..., -rd:], frequencies)
        values, indices = self._window_kv(x, frequencies, position)
        if self.compress_ratio:
            compressed, selected = self._compress_kv(x, qr, position, values.size(1))
            values = (values, compressed)
            indices = torch.cat((indices, selected), dim=-1)
        output = official.sparse_attn(
            q, values, self.attn_sink, indices, self.softmax_scale
        )
        official.apply_rotary_emb(output[..., -rd:], frequencies, True)
        output = output.view(batch, length, self.n_local_groups, -1)
        official.act_quant(
            output,
            official.fp8_block_size,
            official.scale_fmt,
            official.scale_dtype,
            True,
        )
        weight = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        output = torch.einsum("bsgd,grd->bsgr", output, weight)
        return self.wo_b(output.flatten(2))

    attention.forward = MethodType(forward, attention)
    attention._window_kv = MethodType(window, attention)
    attention._compress_kv = MethodType(compressed_kv, attention)
    if attention.compressor is not None:
        attention.compressor.forward = MethodType(compress, attention.compressor)
    if attention.indexer is not None:
        attention.indexer.forward = MethodType(index, attention.indexer)
