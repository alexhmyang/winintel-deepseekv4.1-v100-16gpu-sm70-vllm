# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Exact greedy selection with one compact candidate exchange per TP group."""

import os

import torch
import torch.nn.functional as F


def _packet(local_logits, parallel):
    score, index = local_logits.float().max(-1)
    token = index + parallel.get_rank() * local_logits.shape[-1]
    valid = torch.isfinite(local_logits).all(-1)
    return torch.stack(
        (score.contiguous().view(torch.int32).long(), token, valid.long()), -1
    ).contiguous()


def _finish(packets):
    candidates = torch.stack(packets)
    valid = candidates[..., 2].all()
    scores = candidates[..., 0].to(torch.int32).contiguous().view(torch.float32)
    ids = candidates[..., 1]
    winners = scores == scores.amax(0, keepdim=True)
    return ids.masked_fill(~winners, torch.iinfo(torch.int64).max).amin(0), valid


def _exchange(packet, packets, parallel):
    if parallel.get_world_size() > 1:
        parallel.all_gather(packets, packet)
    else:
        packets[0].copy_(packet)


def _reduce(local_logits, parallel):
    packet = _packet(local_logits, parallel)
    packets = [torch.empty_like(packet) for _ in range(parallel.get_world_size())]
    _exchange(packet, packets, parallel)
    return _finish(packets)


def select(local_logits, parallel, graph=True):
    """Match concatenated argmax, including first-token ties and finite checks."""
    use_graph = (
        graph
        and local_logits.is_cuda
        and os.environ.get("WININTEL_GREEDY_GRAPH", "1") == "1"
    )
    if use_graph:
        key = (local_logits.shape, local_logits.dtype, local_logits.device)
        state = getattr(parallel, "_winintel_greedy_graph", None)
        if state is None or state["key"] != key:
            activation = local_logits.clone()
            stream = torch.cuda.Stream(device=local_logits.device)
            stream.wait_stream(torch.cuda.current_stream(local_logits.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _packet(activation, parallel)
            torch.cuda.current_stream(local_logits.device).wait_stream(stream)
            local_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(local_graph):
                packet = _packet(activation, parallel)
            local_graph.replay()
            packets = [
                torch.empty_like(packet) for _ in range(parallel.get_world_size())
            ]
            _exchange(packet, packets, parallel)
            stream.wait_stream(torch.cuda.current_stream(local_logits.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _finish(packets)
            torch.cuda.current_stream(local_logits.device).wait_stream(stream)
            merge_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(merge_graph):
                output, valid = _finish(packets)
            state = dict(
                key=key,
                activation=activation,
                local_graph=local_graph,
                merge_graph=merge_graph,
                packet=packet,
                packets=packets,
                output=output,
                valid=valid,
            )
            parallel._winintel_greedy_graph = state
        else:
            state["activation"].copy_(local_logits)
        state["local_graph"].replay()
        _exchange(state["packet"], state["packets"], parallel)
        state["merge_graph"].replay()
        output, valid = state["output"], state["valid"]
    else:
        output, valid = _reduce(local_logits, parallel)
    if not valid.item():
        raise RuntimeError("Nonfinite pretrained logits")
    return output.clone() if use_graph else output


def next_token(head, hidden, parallel, graph=True):
    """Keep the official last-position FP32 projection and reduce only candidates."""
    return select(F.linear(hidden[:, -1].float(), head.weight), parallel, graph=graph)
