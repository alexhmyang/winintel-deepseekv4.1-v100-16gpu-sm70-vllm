# Author: 数萌AI服务器 (winintel)
"""NCCL tensor-parallel groups and explicit pipeline boundary transport."""

import torch
import torch.distributed as dist
from pipeline import Boundary


class TensorParallel:
    """Bind official reference collectives to this stage's TP group."""

    def __init__(self, group):
        self.group = group

    def all_reduce(self, tensor):
        return dist.all_reduce(tensor, group=self.group)

    def all_gather(self, outputs, tensor):
        return dist.all_gather(outputs, tensor, group=self.group)

    def get_world_size(self):
        return dist.get_world_size(self.group)

    def get_rank(self):
        return dist.get_rank(self.group)

    def is_initialized(self):
        return dist.is_initialized()


def groups(tp_size, stages):
    world = dist.get_world_size()
    if world != tp_size * stages:
        raise ValueError("World size must equal TP size times pipeline stages")
    rank = dist.get_rank()
    tp_group, pp_group = None, None
    for stage in range(stages):
        ranks = list(range(stage * tp_size, (stage + 1) * tp_size))
        group = dist.new_group(ranks, backend="nccl")
        if rank in ranks:
            tp_group = group
    for lane in range(tp_size):
        ranks = [stage * tp_size + lane for stage in range(stages)]
        group = dist.new_group(ranks, backend="nccl")
        if rank in ranks:
            pp_group = group
    return tp_group, pp_group


class Transport:
    """Send shapes and owned tensor payloads over one NCCL pipeline lane."""

    fields = (
        "hidden",
        "pre_mix",
        "compress_kv",
        "index_k",
        "topk_idxs",
        "candidates",
        "engram_hashes",
    )
    dtypes = (
        torch.float16,
        torch.float32,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
    )

    def __init__(self, group, device):
        self.group, self.device = group, device
        self.reset_request()

    def reset_request(self, cache_capacity=None):
        self.cache_capacity = cache_capacity
        self.sent_cache = {}
        self.received_cache = {}

    def send(self, boundary, destination):
        if boundary.main_hiddens:
            raise ValueError("Draft target transport is not enabled")
        header = torch.full((len(self.fields), 8), -1, dtype=torch.int64)
        payloads = []
        for index, name in enumerate(self.fields):
            value = getattr(boundary, name)
            if value is None:
                continue
            if value.ndim > 4 or value.dtype not in self.dtypes:
                raise ValueError("Unsupported pipeline tensor shape or dtype")
            header[index, :2] = torch.tensor(
                [value.ndim, self.dtypes.index(value.dtype)]
            )
            header[index, 2 : 2 + value.ndim] = torch.tensor(value.shape)
            begin = 0
            if name in ("compress_kv", "index_k") and self.cache_capacity:
                identity = (value.data_ptr(), tuple(value.shape[::2]), value.dtype)
                previous = self.sent_cache.get(name)
                if previous and previous[0] == identity:
                    begin = min(previous[1], value.size(1))
                self.sent_cache[name] = (identity, value.size(1))
                base = value._base
                capacity = self.cache_capacity
                if base is not None and base.ndim == 3:
                    capacity = min(capacity, base.size(1))
                header[index, 6:] = torch.tensor([begin, capacity])
                value = value[:, begin:]
            payloads.append(value.contiguous().view(torch.uint8))
        dist.send(header.to(self.device), dst=destination, group=self.group)
        for payload in payloads:
            if payload.numel():
                dist.send(payload, dst=destination, group=self.group)

    def receive(self, source):
        header = torch.empty(
            (len(self.fields), 8), dtype=torch.int64, device=self.device
        )
        dist.recv(header, src=source, group=self.group)
        values = []
        for name, description in zip(self.fields, header.cpu().tolist()):
            ndim, code = description[:2]
            if ndim == -1:
                values.append(None)
                continue
            if not 0 <= ndim <= 4 or not 0 <= code < len(self.dtypes):
                raise ValueError("Invalid pipeline tensor metadata")
            shape = description[2 : 2 + ndim]
            if any(size < 0 for size in shape):
                raise ValueError("Invalid pipeline tensor extent")
            begin, capacity = description[6:]
            dtype = self.dtypes[code]
            if capacity != -1:
                if (
                    name not in ("compress_kv", "index_k")
                    or ndim != 3
                    or not 0 <= begin <= shape[1] <= capacity
                    or not self.cache_capacity
                    or capacity > self.cache_capacity
                ):
                    raise ValueError("Invalid incremental cache metadata")
                cache = self.received_cache.get(name)
                if cache is None:
                    if begin:
                        raise ValueError("First cache transfer must include its prefix")
                    cache = torch.empty(
                        shape[0], capacity, shape[2], dtype=dtype, device=self.device
                    )
                    self.received_cache[name] = cache
                payload = torch.empty(
                    shape[0],
                    shape[1] - begin,
                    shape[2],
                    dtype=dtype,
                    device=self.device,
                )
                if payload.numel():
                    dist.recv(payload.view(torch.uint8), src=source, group=self.group)
                    cache[:, begin : shape[1]].copy_(payload)
                value = cache[:, : shape[1]]
            else:
                value = torch.empty(shape, dtype=dtype, device=self.device)
                if value.numel():
                    dist.recv(value.view(torch.uint8), src=source, group=self.group)
            values.append(value)
        if values[0] is None or values[1] is None:
            raise ValueError("Pipeline boundary must include residual and mix")
        return Boundary(*values, ())
