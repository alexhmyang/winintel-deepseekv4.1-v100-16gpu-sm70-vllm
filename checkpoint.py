# Author: 数萌AI服务器 (winintel)
"""Read native checkpoint tensors and Engram rows without loading whole tables."""

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request

import requests
import torch
from torch import nn
from urllib3.exceptions import HTTPError as TransportError

_http_state = threading.local()


class _RangeResponse:
    def __init__(self, response):
        self.response = response
        self.status = response.status_code
        self.headers = response.headers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.response.close()
        return False

    def read(self, size):
        try:
            return self.response.raw.read(size)
        except (requests.RequestException, TransportError, OSError) as error:
            raise URLError("Mirror range body was interrupted") from error


def urlopen(request, timeout):
    """Reuse each worker's connections and refresh expired download redirects."""
    if not hasattr(_http_state, "session"):
        _http_state.session = requests.Session()
        _http_state.redirects = {}
    original = request.full_url
    cached = _http_state.redirects.get(original, original)
    targets = (cached, original) if cached != original else (original,)
    for target in targets:
        try:
            response = _http_state.session.get(
                target,
                headers=dict(request.header_items()),
                timeout=timeout,
                stream=True,
            )
        except requests.RequestException as error:
            raise URLError("Mirror range connection failed") from error
        if response.status_code in (401, 403) and target != original:
            response.close()
            _http_state.redirects.pop(original, None)
            continue
        if response.status_code >= 400:
            response.close()
            raise HTTPError(
                original,
                response.status_code,
                "Mirror request failed",
                response.headers,
                None,
            )
        _http_state.redirects[original] = response.url
        return _RangeResponse(response)
    raise URLError("Mirror download redirect could not be refreshed")


def storage_dtype(name):
    names = {
        "U8": torch.uint8,
        "I8": torch.int8,
        "I32": torch.int32,
        "I64": torch.int64,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E8M0": torch.float8_e8m0fnu,
    }
    if name not in names:
        raise ValueError(f"Unsupported checkpoint storage dtype: {name}")
    return names[name]


class TensorFile:
    """Use bounded file reads; memory use depends on selected rows."""

    def __init__(self, path):
        self.path = Path(path)
        self.file_size = self.path.stat().st_size
        with self.path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated safetensors header")
            size = struct.unpack("<Q", prefix)[0]
            if size > min(64 * 1024**2, self.file_size - 8):
                raise ValueError("Invalid safetensors header length")
            self.header = json.loads(stream.read(size))
        self.offset = 8 + size
        self.bytes_read = 0

    def _read(self, offset, size):
        with self.path.open("rb") as stream:
            stream.seek(offset)
            data = bytearray(stream.read(size))
        if len(data) != size:
            raise ValueError("Checkpoint changed during a bounded read")
        self.bytes_read += len(data)
        return data

    def spec(self, name):
        entry = self.header[name]
        dtype = storage_dtype(entry["dtype"])
        shape = entry["shape"]
        start, end = entry["data_offsets"]
        width = torch.empty((), dtype=dtype).element_size()
        if (
            any(not isinstance(n, int) or n < 0 for n in shape)
            or start < 0
            or end - start != math.prod(shape) * width
            or self.offset + end > self.file_size
        ):
            raise ValueError(f"Invalid tensor extent: {name}")
        return shape, dtype, start, end

    def read_rows(self, name, indices):
        shape, dtype, start, _ = self.spec(name)
        if len(shape) != 2:
            raise ValueError("Row reads require a matrix")
        indices = torch.as_tensor(indices, device="cpu")
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Row indices must be integers")
        if indices.numel() and (
            indices.min().item() < 0 or indices.max().item() >= shape[0]
        ):
            raise IndexError("Checkpoint row index is outside the table")
        unique, inverse = torch.unique(indices.flatten().long(), return_inverse=True)
        row_bytes = shape[1] * torch.empty((), dtype=dtype).element_size()
        values = torch.empty((unique.numel(), shape[1]), dtype=dtype)
        for position, row in enumerate(unique.tolist()):
            data = self._read(self.offset + start + row * row_bytes, row_bytes)
            values[position].view(torch.uint8).copy_(
                torch.frombuffer(data, dtype=torch.uint8)
            )
        # Byte indexing works even for dtypes without CPU indexing kernels.
        raw = values.view(torch.uint8)[inverse]
        return raw.contiguous().view(dtype).reshape(*indices.shape, shape[1])

    def read_tensor(self, name, max_bytes=256 * 1024**2):
        shape, dtype, start, end = self.spec(name)
        if end - start > max_bytes:
            raise ValueError("Tensor exceeds bounded read; use read_rows")
        data = self._read(self.offset + start, end - start)
        if not data:
            return torch.empty(shape, dtype=dtype)
        return torch.frombuffer(data, dtype=torch.uint8).view(dtype).reshape(shape)

    def read_partition(self, name, rank, world_size, axis=0):
        """Read a contiguous row partition without allocating the whole tensor."""
        shape, dtype, start, _ = self.spec(name)
        if axis not in (0, 1) or axis >= len(shape):
            raise ValueError("Partition axis must be a tensor dimension")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid tensor partition rank")
        if shape[axis] % world_size:
            raise ValueError("Tensor dimension must divide evenly across ranks")
        width = shape[axis] // world_size
        if axis == 1:
            value = self.read_tensor(name)
            # Byte views work for shell dtypes such as packed FP4.
            element_bytes = value.element_size()
            return value.view(torch.uint8)[
                :, rank * width * element_bytes : (rank + 1) * width * element_bytes
            ].clone()
        row_bytes = math.prod(shape[1:]) * torch.empty((), dtype=dtype).element_size()
        data = self._read(
            self.offset + start + rank * width * row_bytes, width * row_bytes
        )
        return (
            torch.frombuffer(data, dtype=torch.uint8)
            .view(dtype)
            .reshape(width, *shape[1:])
        )


class MirrorTensorFile(TensorFile):
    """Read pinned original rows from the domestic mirror for validation.

    A server ignoring Range is rejected before consuming its body. Remote row
    reads are a development fallback, not a low-latency serving strategy.
    """

    def __init__(self, filename, timeout=30):
        if not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", filename):
            raise ValueError("Supply an original model shard filename")
        self.url = (
            "https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/"
            "1bae0e1088993b9b82de08f59de5be5a34e42170/" + filename
        )
        self.path = Path(filename)
        self.timeout = timeout
        self.file_size = None
        self.bytes_read = 0
        directory = Path(
            os.environ.get(
                "WININTEL_CHECKPOINT_CACHE", Path.home() / ".cache/winintel-checkpoint"
            )
        )
        directory.mkdir(parents=True, exist_ok=True)
        self.cache_path = directory / (
            hashlib.sha256(self.url.encode()).hexdigest() + ".sqlite"
        )
        with sqlite3.connect(self.cache_path, timeout=60) as database:
            database.execute("PRAGMA journal_mode=WAL")
            database.execute(
                "CREATE TABLE IF NOT EXISTS ranges "
                "(offset INTEGER, size INTEGER, total INTEGER, data BLOB, "
                "PRIMARY KEY(offset,size))"
            )
        size = struct.unpack("<Q", self._read(0, 8))[0]
        if size > min(64 * 1024**2, self.file_size - 8):
            raise ValueError("Invalid remote safetensors header length")
        self.header = json.loads(self._read(8, size))
        self.offset = 8 + size
        self.row_cache = OrderedDict()
        self.row_cache_limit = 100000
        self.pool = ThreadPoolExecutor(max_workers=16)

    def read_rows(self, name, indices):
        shape, dtype, start, _ = self.spec(name)
        indices = torch.as_tensor(indices, device="cpu")
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Row indices must be integers")
        if indices.numel() and (
            indices.min().item() < 0 or indices.max().item() >= shape[0]
        ):
            raise IndexError("Checkpoint row index is outside the table")
        unique, inverse = torch.unique(indices.flatten().long(), return_inverse=True)
        row_bytes = shape[1] * torch.empty((), dtype=dtype).element_size()
        result = {}
        missing = []
        for row in unique.tolist():
            key = (name, row)
            if key in self.row_cache:
                result[row] = self.row_cache[key]
                self.row_cache.move_to_end(key)
            else:
                missing.append(row)

        def fetch(row):
            return row, self._read(self.offset + start + row * row_bytes, row_bytes)

        for row, data in self.pool.map(fetch, missing):
            value = torch.frombuffer(data, dtype=torch.uint8).clone()
            result[row] = value
            self.row_cache[(name, row)] = value
            while len(self.row_cache) > self.row_cache_limit:
                self.row_cache.popitem(last=False)
        if not result:
            return torch.empty((*indices.shape, shape[1]), dtype=dtype)
        raw = torch.stack([result[row] for row in unique.tolist()])[inverse]
        return raw.contiguous().view(dtype).reshape(*indices.shape, shape[1])

    def _read(self, offset, size):
        if not size:
            return bytearray()
        cache_path = getattr(self, "cache_path", None)
        if cache_path:
            with sqlite3.connect(cache_path, timeout=60) as database:
                cached = database.execute(
                    "SELECT total,data FROM ranges WHERE offset=? AND size=?",
                    (offset, size),
                ).fetchone()
            if cached is not None:
                total, data = cached
                if len(data) != size or (
                    self.file_size is not None and total != self.file_size
                ):
                    raise ValueError("Cached checkpoint range is invalid")
                self.file_size = total
                return bytearray(data)
        end = offset + size - 1
        request = Request(
            self.url,
            headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"},
        )
        for attempt in range(5):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)",
                        response.headers.get("Content-Range", ""),
                    )
                    if response.status != 206 or match is None:
                        raise ValueError(
                            "Mirror must honor the bounded HTTP Range request"
                        )
                    first, last, total = map(int, match.groups())
                    if (
                        first != offset
                        or last != end
                        or (self.file_size is not None and total != self.file_size)
                    ):
                        raise ValueError(
                            "Mirror returned a different checkpoint byte range"
                        )
                    self.file_size = total
                    data = bytearray(response.read(size + 1))
                break
            except (URLError, TimeoutError) as error:
                if attempt == 4 or (
                    isinstance(error, HTTPError)
                    and error.code < 500
                    and error.code not in (408, 429)
                ):
                    raise
                print(
                    f"MIRROR_RETRY offset={offset} bytes={size} attempt={attempt + 1}",
                    flush=True,
                )
        if len(data) != size:
            raise ValueError("Mirror returned an unexpected payload length")
        self.bytes_read += len(data)
        if cache_path:
            with sqlite3.connect(cache_path, timeout=60) as database:
                database.execute(
                    "INSERT OR REPLACE INTO ranges VALUES(?,?,?,?)",
                    (offset, size, self.file_size, data),
                )
        return data


class CachedTensorFile(TensorFile):
    """Reuse the bounded original-row cache for local files and NFS mounts."""

    def __init__(self, path):
        super().__init__(path)
        self.row_cache = OrderedDict()
        self.row_cache_limit = 100000
        self.pool = ThreadPoolExecutor(max_workers=16)

    read_rows = MirrorTensorFile.read_rows


class Checkpoint:
    """Resolve native tensor names through the model's explicit shard index."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.index = json.loads(
            (self.root / "model.safetensors.index.json").read_text("utf-8")
        )["weight_map"]
        self.files = {}

    def file_for(self, name):
        path = (self.root / self.index[name]).resolve()
        if path.parent != self.root:
            raise ValueError("Checkpoint shard must be inside its model directory")
        if path not in self.files:
            self.files[path] = TensorFile(path)
        return self.files[path]

    def tensor(self, name, max_bytes=256 * 1024**2):
        return self.file_for(name).read_tensor(name, max_bytes)

    def rows(self, name, indices):
        return self.file_for(name).read_rows(name, indices)

    def add_original_mirror_shard(self, filename, names=None):
        """Supply original Engram tensors from a local directory or the mirror."""
        if not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", filename):
            raise ValueError("Supply an original model shard filename")
        directory = os.environ.get("WININTEL_ENGRAM_DIRECTORY")
        local = Path(directory) / filename if directory else self.root / filename
        if directory or local.is_file():
            source = CachedTensorFile(local)
        else:
            source = MirrorTensorFile(filename)
        if names is None:
            names = set(source.header).difference({"__metadata__"})
        for name in names:
            source.spec(name)
            if name in self.index:
                raise ValueError(f"Mirror shard would replace a local tensor: {name}")
        path = self.root / filename
        if path in self.files:
            raise ValueError("Mirror shard would replace an open local file")
        self.files[path] = source
        for name in names:
            self.index[name] = filename
        return source

    def prepare_fp4(self, prefix, device="cuda"):
        """Read one real expert matrix, keeping both weight and scale packed."""
        from qwen_adapter import prepare_weight

        weight = self.tensor(prefix + ".weight")
        scales = self.tensor(prefix + ".scale")
        if weight.dtype not in (torch.int8, torch.uint8):
            raise ValueError("Native FP4 checkpoint must contain packed bytes")
        if scales.dtype != torch.float8_e8m0fnu:
            raise ValueError("Native FP4 checkpoint must contain E8M0 scales")
        weight = weight.view(torch.float4_e2m1fn_x2).to(device)
        scales = scales.to(device)
        weight.scale = scales
        prepare_weight(weight, scales)
        return weight


class PagedEngramEmbedding(nn.Module):
    """Read original FP8 rows on demand and reduce rank-local contributions.

    File I/O must run outside CUDA Graph capture. Rows retain original FP8
    storage precision and are decoded only when requested.
    """

    def __init__(
        self, checkpoint, prefix, device="cuda", rank=0, world_size=1, group=None
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.device = torch.device(device)
        shape, dtype, _, _ = checkpoint.file_for(prefix + ".weight").spec(
            prefix + ".weight"
        )
        scale_shape, scale_dtype, _, _ = checkpoint.file_for(prefix + ".scale").spec(
            prefix + ".scale"
        )
        if (
            len(shape) != 2
            or dtype != torch.float8_e4m3fn
            or scale_dtype != torch.float8_e8m0fnu
            or shape[1] % 32
            or scale_shape != [shape[0], shape[1] // 32]
        ):
            raise ValueError("Engram requires original FP8 rows and per-32 E8M0 scales")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid Engram rank")
        self.num_embeddings, self.dim = shape
        self.world_size = world_size
        self.group = group
        per_rank = (shape[0] + world_size - 1) // world_size
        self.begin = rank * per_rank
        self.end = min(self.begin + per_rank, shape[0])

    @torch.inference_mode()
    def forward(self, indices):
        from kernel import _scale_float

        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Page Engram rows before CUDA Graph capture")
        rows = indices.cpu()
        if rows.dtype not in (torch.int32, torch.int64):
            raise ValueError("Engram indices must be integers")
        if rows.numel() and (
            rows.min().item() < 0 or rows.max().item() >= self.num_embeddings
        ):
            raise IndexError("Engram hash id is outside the original table")
        local = (rows >= self.begin) & (rows < self.end)
        output = torch.zeros((*rows.shape, self.dim), dtype=torch.float32)
        if local.any():
            selected = rows[local]
            weight = self.checkpoint.rows(self.prefix + ".weight", selected)
            scales = self.checkpoint.rows(self.prefix + ".scale", selected)
            values = weight.float().reshape(-1, self.dim // 32, 32)
            values *= _scale_float(scales).unsqueeze(-1)
            output[local] = values.flatten(-2)
        output = output.to(device=self.device, dtype=torch.bfloat16)
        if self.world_size > 1:
            import torch.distributed as dist

            if (
                not dist.is_initialized()
                or dist.get_world_size(self.group) != self.world_size
            ):
                raise RuntimeError("Engram process group does not match row sharding")
            dist.all_reduce(output, group=self.group)
        return output
