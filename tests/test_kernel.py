# Author: 数萌AI服务器 (winintel)
"""Guard official storage semantics and observable reference calculations."""

import functools
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "winintel_reference_kernel", Path(__file__).parents[1] / "kernel.py"
)
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)


def test_serving_routes_installation_is_idempotent():
    """Repeated installation must retain the already grouped decode forward."""
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from routing import install

    calls = []

    def packed_forward(x, image_mask=None):
        calls.append((x, image_mask))
        return x * 3

    moe = SimpleNamespace(
        forward=packed_forward,
        _winintel_grouped_decode_routes=True,
    )
    install(SimpleNamespace(), moe)
    x = torch.randn(8, 256)
    mask = torch.ones(8, dtype=torch.bool)
    torch.testing.assert_close(moe.forward(x, mask), x * 3)
    assert calls == [(x, mask)]


@pytest.mark.parametrize("invalid", [False, True])
def test_compact_greedy_preserves_global_ties_and_remote_finite_checks(invalid):
    """A remote tied maximum picks the first global ID; any bad rank rejects."""
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from greedy import select

    logits = [
        torch.tensor([[1.0, 7.0, 7.0], [0.0, -2.0, 3.0]]),
        torch.tensor([[7.0, 0.0, 1.0], [5.0, 5.0, 0.0]]),
    ]
    if invalid:
        logits[1][0, 1] = float("nan")

    def gather(outputs, packet):
        for rank, local in enumerate(logits):
            score, index = local.max(-1)
            outputs[rank].copy_(
                torch.stack(
                    (
                        score.view(torch.int32).long(),
                        index + rank * 3,
                        torch.isfinite(local).all(-1).long(),
                    ),
                    -1,
                )
            )

    parallel = SimpleNamespace(
        get_world_size=lambda: 2, get_rank=lambda: 0, all_gather=gather
    )
    if invalid:
        with pytest.raises(RuntimeError, match="Nonfinite"):
            select(logits[0], parallel)
    else:
        torch.testing.assert_close(
            select(logits[0], parallel), torch.cat(logits, -1).argmax(-1)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
def test_fp8_cache_redecodes_changed_weights_and_obeys_memory_limit(monkeypatch):
    """In-place checkpoint updates must invalidate cached values without growth."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    import native_ops

    native_ops._decoded_weights.clear()
    monkeypatch.setenv("WININTEL_FP8_CACHE_MIB", "1")
    weight = torch.ones(256, 256, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.ones(8, 8, device="cuda")
    a = torch.ones(2, 256, device="cuda").to(torch.float8_e4m3fn)
    a_s = torch.ones(2, 8, device="cuda")
    native_ops.prepare_fp8(weight, scales)
    first = kernel.fp8_gemm(a, a_s, weight, scales, block_size=32)
    weight.copy_(torch.full((256, 256), 2.0, device="cuda"))
    scales.mul_(0.5)
    torch.testing.assert_close(
        kernel.fp8_gemm(a, a_s, weight, scales, block_size=32), first
    )
    for _ in range(6):
        other = weight.clone()
        native_ops.prepare_fp8(other, scales)
        kernel.fp8_gemm(a, a_s, other, scales, block_size=32)
    assert (
        sum(v[3].numel() * 4 for v in native_ops._decoded_weights.values()) <= 1024**2
    )
    assert id(weight) in native_ops._decoded_weights
    native_ops._decoded_weights.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
def test_greedy_head_graph_preserves_last_position_and_changed_logits():
    """Replay must select current head scores and reject a later nonfinite input."""
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from greedy import next_token
    from reference import load_official

    official = load_official()
    head = official.ParallelHead(256, 128).cuda()
    parallel = SimpleNamespace(get_world_size=lambda: 1, get_rank=lambda: 0)
    with torch.no_grad():
        head.weight.normal_(0, 0.02)
        for rows in (1, 2, 2):
            hidden = torch.randn(rows, 3, 128, device="cuda")
            expected = head(hidden).argmax(-1)
            torch.testing.assert_close(next_token(head, hidden, parallel), expected)
        hidden[0, -1, 0] = float("nan")
        with pytest.raises(RuntimeError, match="Nonfinite"):
            next_token(head, hidden, parallel)


@pytest.mark.parametrize("rows", [1, 16])
def test_fp8_adapter_preserves_quantized_input_and_rejects_replaced_scales(rows):
    """PyTorch execution must honor official FP8 rounding and scale ownership."""
    import os

    if not torch.cuda.is_available() or os.getenv("WININTEL_NATIVE_SM70") != "1":
        pytest.skip("Independent CUDA execution must be explicitly configured")
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from native_ops import prepare_fp8

    weight = (torch.randn(1792, 5120, device="cuda", dtype=torch.float32) * 32).to(
        torch.float8_e4m3fn
    )
    scales = torch.randint(115, 121, (56, 160), device="cuda", dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.float16)
    a, a_s = kernel.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    expected = kernel.fp8_gemm(a, a_s, weight, scales, block_size=32)
    prepare_fp8(weight, scales)
    actual = kernel.fp8_gemm(a, a_s, weight, scales, block_size=32)
    error = (actual.float() - expected.float()).norm()
    assert error / expected.float().norm().clamp_min(1e-12) < 0.001
    with pytest.raises(ValueError, match="Reprepare native FP8"):
        kernel.fp8_gemm(a, a_s, weight, scales.clone(), block_size=32)


@pytest.mark.parametrize("rows,iterations", [(1, 1), (16, 20)])
def test_native_hc_matches_fp32_reference_after_graph_input_changes(rows, iterations):
    """Sinkhorn adaptation must preserve residual range and changing graph inputs."""
    import os
    from types import SimpleNamespace

    if not torch.cuda.is_available() or os.getenv("WININTEL_NATIVE_SM70") != "1":
        pytest.skip("Independent CUDA execution must be explicitly configured")
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from native_hc import install
    from reference import load_official

    official = load_official()
    block = SimpleNamespace(
        hc_mult=4,
        hc_attn_fn=torch.randn(24, 20480, device="cuda", dtype=torch.float32) * 0.005,
        norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=iterations,
    )
    scale = torch.tensor([0.3, 0.8, 0.4], device="cuda", dtype=torch.float32)
    base = torch.randn(24, device="cuda", dtype=torch.float32)
    x = torch.randn(rows, 1, 4, 5120, device="cuda", dtype=torch.float32)
    install(official, block)

    def reference():
        return official.Block.hc_mixes(block, x, block.hc_attn_fn, scale, base)

    def native():
        return block.hc_mixes(x, block.hc_attn_fn, scale, base)

    with torch.inference_mode():
        for actual, expected in zip(native(), reference()):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            native()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = native()
        x.mul_(100000).add_(713)
        scale.mul_(0.7)
        base.add_(0.2)
        graph.replay()
        for actual, expected in zip(output, reference()):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            assert torch.isfinite(actual).all()


def test_independent_http_streams_preserve_each_sessions_tokens():
    """Separate sockets must receive only their own ordered generated tokens."""
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from http_sessions import StreamingServer, collect_stream

    tokenizer = SimpleNamespace(decode=lambda tokens: str(tokens[0]))
    server = StreamingServer(tokenizer)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(collect_stream, server.address, [index, 3], 2)
                for index in (1, 2)
            ]
            sessions = [server.pending.get(timeout=3) for _ in range(2)]
            server.token(sessions, [session.rows[0] * 10 for session in sessions], 0)
            server.token(
                sessions, [session.rows[0] * 10 + 1 for session in sessions], 1
            )
            server.finish(sessions)
            clients = [future.result(timeout=3) for future in futures]
        assert clients[0]["token_ids"] == [10, 11]
        assert clients[1]["token_ids"] == [20, 21]
        assert clients[0]["request_id"] != clients[1]["request_id"]
        assert all(len(client["token_elapsed_seconds"]) == 2 for client in clients)
    finally:
        server.close()


@pytest.mark.parametrize("limit", [0.0, 0.5])
def test_grouped_decode_routes_preserve_expert_addition_order(limit):
    """Host grouping must preserve route slots, clamps and inactive local experts."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from reference import load_official
    from routing import install

    official = load_official()
    args = official.ModelArgs(
        dim=128,
        moe_inter_dim=64,
        n_routed_experts=4,
        n_activated_experts=2,
        dtype="bf16",
        expert_dtype=None,
        swiglu_limit=limit,
    )
    torch.manual_seed(83)
    moe = official.MoE(0, args).float()
    with torch.no_grad():
        for parameter in moe.parameters():
            parameter.normal_(0, 0.2)
    original = moe.forward
    install(official, moe)
    with torch.inference_mode():
        for tokens in (1, 2, 16, 17):
            x = torch.randn(1, tokens, 128)
            for start, stop in ((0, 4), (0, 2), (4, 4)):
                moe.experts_start_idx, moe.experts_end_idx = start, stop
                torch.testing.assert_close(moe(x), original(x), atol=0, rtol=0)


def test_slow_decode_stops_higher_concurrency_only():
    """A slow completed case blocks escalation but permits same-batch A/B."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from benchmark import speed_rejection

    result = {
        "name": "8k-two",
        "status": "completed",
        "concurrency": 2,
        "decode_tokens_per_second_per_request": 19.9,
    }
    assert speed_rejection({"concurrency": 4}, [result])
    assert speed_rejection({"concurrency": 2}, [result]) is None
    result["decode_tokens_per_second_per_request"] = 20
    assert speed_rejection({"concurrency": 4}, [result]) is None
    result["status"] = "not_run"
    result["decode_tokens_per_second_per_request"] = 0
    assert speed_rejection({"concurrency": 4}, [result]) is None


def test_long_context_capacity_rejection_keeps_live_caches(monkeypatch):
    """An oversized batch must be refused before any live tensor is replaced."""
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from benchmark import capacity

    attention = torch.nn.Module()
    attention.compress_ratio = 2
    cache = torch.zeros(1, 16, 32)
    attention.register_buffer("compress_kv_cache", cache)
    args = SimpleNamespace(
        rope_head_dim=8,
        kv_source_layers=(0,),
        compress_ratios=(2,),
        head_dim=32,
        index_head_dim=16,
    )
    stage = SimpleNamespace(
        layers=[SimpleNamespace(attn=attention)],
        args=args,
        hash_state=SimpleNamespace(cache=torch.zeros(1, 32, dtype=torch.int64)),
        start=1,
        device=torch.device("cpu"),
    )
    transport = SimpleNamespace(received_cache={})
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (3 * 1024**3, 0))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    assert capacity(stage, transport, 1, 8192)["admitted"]
    rejected = capacity(stage, transport, 16, 1000064)
    assert not rejected["admitted"]
    assert rejected["estimate_only"]
    assert attention.compress_kv_cache is cache


def test_request_cache_can_resize_after_an_allocation_failure(monkeypatch):
    """A failed allocation must retain cache shapes for the next smaller request."""
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from request_cache import reset

    attention = torch.nn.Module()
    attention.compress_ratio, attention.indexer = 2, None
    attention.register_buffer("compress_kv_cache", torch.zeros(1, 16, 32))
    args = SimpleNamespace(
        rope_head_dim=8,
        original_seq_len=32,
        compress_rope_theta=10000,
        rope_theta=10000,
        rope_factor=1,
        beta_fast=32,
        beta_slow=1,
    )
    official = SimpleNamespace(
        SharedAttentionRuntime=SimpleNamespace,
        precompute_freqs_cis=SimpleNamespace(
            __wrapped__=lambda width, length, *unused: torch.zeros(length, width // 2)
        ),
    )
    stage = SimpleNamespace(
        layers=[SimpleNamespace(attn=attention)],
        official=official,
        args=args,
        hash_state=SimpleNamespace(cache=torch.zeros(1, 32, dtype=torch.int64)),
        device=torch.device("cpu"),
    )
    original = torch.full

    def fail(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("synthetic cache allocation failure")

    monkeypatch.setattr(torch, "full", fail)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        reset(stage, 4, 128)
    assert attention.compress_kv_cache is None
    assert stage.hash_state.cache is None
    monkeypatch.setattr(torch, "full", original)
    reset(stage, 2, 48)
    assert attention.compress_kv_cache.shape == (2, 24, 32)
    assert stage.hash_state.cache.shape == (2, 48)
    assert not attention.compress_kv_cache.any()


def test_checkpoint_row_reads_preserve_fp8_bytes_and_request_order(tmp_path):
    """Engram paging must preserve duplicates while reading only distinct rows."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from checkpoint import Checkpoint, PagedEngramEmbedding

    weight = torch.arange(8 * 32).reshape(8, 32).remainder(16).float()
    weight = weight.to(torch.float8_e4m3fn)
    scale = torch.arange(123, 131, dtype=torch.uint8).reshape(8, 1)
    scale = scale.view(torch.float8_e8m0fnu)
    tensors = {"table.weight": weight, "table.scale": scale}
    header, data = {}, bytearray()
    for name, tensor in tensors.items():
        raw = bytes(tensor.view(torch.uint8).flatten().tolist())
        header[name] = {
            "dtype": "F8_E4M3" if name.endswith("weight") else "F8_E8M0",
            "shape": list(tensor.shape),
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data.extend(raw)
    encoded = json.dumps(header).encode()
    (tmp_path / "rows.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + data
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "rows.safetensors")})
    )
    checkpoint = Checkpoint(tmp_path)
    embedding = PagedEngramEmbedding(checkpoint, "table", device="cpu")
    ids = torch.tensor([[7, 0, 7], [3, 0, 3]])
    actual = embedding(ids)
    expected = weight.float()[ids] * kernel._scale_float(scale)[ids]
    torch.testing.assert_close(actual, expected.bfloat16())
    assert checkpoint.file_for("table.weight").bytes_read == 3 * (32 + 1)
    with pytest.raises(IndexError, match="outside"):
        embedding(torch.tensor([8]))
    with pytest.raises(ValueError, match="integers"):
        embedding(torch.tensor([0.0]))


def test_checkpoint_refuses_truncated_payload_and_unbounded_tensor_read(tmp_path):
    """Malformed metadata must fail before a large tensor allocation."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from checkpoint import TensorFile

    header = {"weight": {"dtype": "F32", "shape": [2, 32], "data_offsets": [0, 256]}}
    encoded = json.dumps(header).encode()
    path = tmp_path / "broken.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(128))
    with pytest.raises(ValueError, match="extent"):
        TensorFile(path).read_tensor("weight")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(256))
    with pytest.raises(ValueError, match="bounded read"):
        TensorFile(path).read_tensor("weight", max_bytes=128)


@pytest.mark.parametrize("status,content_range", [(200, ""), (206, "bytes 10-17/1000")])
def test_mirror_refuses_full_file_or_wrong_range_without_reading(
    monkeypatch, status, content_range
):
    """A mirror must not turn a tiny Engram row request into a huge download."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    import checkpoint

    class Response:
        headers = {"Content-Range": content_range}

        def __enter__(self):
            self.status = status
            return self

        def __exit__(self, *args):
            return False

        def read(self, *args):
            pytest.fail("Rejected range response must not consume the payload")

    monkeypatch.setattr(checkpoint, "urlopen", lambda *a, **k: Response())
    source = checkpoint.MirrorTensorFile.__new__(checkpoint.MirrorTensorFile)
    source.url = "https://hf-mirror.com/example"
    source.timeout = 1
    source.file_size = None
    source.bytes_read = 0
    with pytest.raises(ValueError, match="Range|range"):
        source._read(0, 8)


def test_mirror_retries_timeout_and_reuses_rows_after_reopening(monkeypatch, tmp_path):
    """A transient download must not abort generation or repeat on restart."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    import checkpoint

    header = {"weight": {"dtype": "U8", "shape": [2, 32], "data_offsets": [0, 64]}}
    encoded = json.dumps(header).encode()
    payload = struct.pack("<Q", len(encoded)) + encoded + bytes(range(64))
    calls = []

    class Response:
        status = 206

        def __init__(self, first, last):
            self.data = payload[first : last + 1]
            self.headers = {"Content-Range": f"bytes {first}-{last}/{len(payload)}"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, *args):
            return self.data

    def fetch(request, **kwargs):
        calls.append(request)
        if len(calls) == 1:
            raise TimeoutError("Transient mirror stall")
        first, last = map(int, request.get_header("Range")[6:].split("-"))
        return Response(first, last)

    monkeypatch.setenv("WININTEL_CHECKPOINT_CACHE", str(tmp_path))
    monkeypatch.setattr(checkpoint, "urlopen", fetch)
    source = checkpoint.MirrorTensorFile("model-00047-of-00048.safetensors")
    expected = torch.arange(64, dtype=torch.uint8).reshape(2, 32)[[1, 0, 1]]
    actual = source.read_rows("weight", torch.tensor([1, 0, 1]))
    torch.testing.assert_close(actual, expected)
    assert len(calls) == 5
    monkeypatch.setattr(
        checkpoint,
        "urlopen",
        lambda *a, **k: pytest.fail("Restart must reuse validated disk ranges"),
    )
    reopened = checkpoint.MirrorTensorFile("model-00047-of-00048.safetensors")
    torch.testing.assert_close(
        reopened.read_rows("weight", torch.tensor([1, 0, 1])), expected
    )
    assert reopened.bytes_read == 0


def test_mirror_connection_reuses_redirect_and_refreshes_expired_address(monkeypatch):
    """Later ranges must keep their headers when a signed download URL expires."""
    import io
    import threading
    from urllib.request import Request

    sys.path.insert(0, str(Path(__file__).parents[1]))
    import checkpoint

    original = "https://hf-mirror.com/example"
    download = "https://download.example/shard"
    refreshed = "https://download.example/refreshed"
    calls, responses = [], []

    class Response:
        def __init__(self, status, url):
            self.status_code, self.url = status, url
            self.headers = {"Content-Range": "bytes 0-7/8"}
            self.raw = io.BytesIO(b"original")
            self.closed = False

        def close(self):
            self.closed = True
            self.raw.close()

    replies = iter(
        ((206, download), (403, download), (206, refreshed), (206, refreshed))
    )

    class Session:
        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            response = Response(*next(replies))
            responses.append(response)
            return response

    monkeypatch.setattr(checkpoint, "_http_state", threading.local())
    monkeypatch.setattr(checkpoint.requests, "Session", Session)
    for _ in range(3):
        request = Request(original, headers={"Range": "bytes=0-7"})
        with checkpoint.urlopen(request, 3) as response:
            assert response.status == 206
            assert response.read(9) == b"original"
    assert [url for url, _ in calls] == [original, download, original, refreshed]
    assert all(options["headers"]["Range"] == "bytes=0-7" for _, options in calls)
    assert all(options["timeout"] == 3 and options["stream"] for _, options in calls)
    assert all(response.closed for response in responses)


def test_original_shard_directory_reads_requested_rows_without_network(
    monkeypatch, tmp_path
):
    """A complete original shard must support offline, ordered Engram paging."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    import checkpoint

    filename = "model-00047-of-00048.safetensors"
    header = {"weight": {"dtype": "U8", "shape": [2, 32], "data_offsets": [0, 64]}}
    encoded = json.dumps(header).encode()
    (tmp_path / filename).write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + bytes(range(64))
    )
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    monkeypatch.setenv("WININTEL_ENGRAM_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(
        checkpoint,
        "MirrorTensorFile",
        lambda *a, **k: pytest.fail("Local original rows must not access the mirror"),
    )
    source = checkpoint.Checkpoint(tmp_path)
    source.add_original_mirror_shard(filename, ["weight"])
    ids = torch.tensor([1, 0, 1])
    expected = torch.arange(64, dtype=torch.uint8).reshape(2, 32)[ids]
    torch.testing.assert_close(source.rows("weight", ids), expected)
    assert source.file_for("weight").bytes_read == 64
    order = torch.tensor([1, 0, 2])
    torch.testing.assert_close(source.rows("weight", ids[order]), expected[order])
    assert source.file_for("weight").bytes_read == 64
    source.file_for("weight").row_cache_limit = 1
    source.file_for("weight").row_cache.clear()
    torch.testing.assert_close(source.rows("weight", ids), expected)
    assert len(source.file_for("weight").row_cache) == 1
    torch.testing.assert_close(source.rows("weight", torch.tensor([0])), expected[1:2])
    assert source.file_for("weight").bytes_read == 160
    with pytest.raises(ValueError, match="filename"):
        source.add_original_mirror_shard("../private.safetensors")


def test_fp4_decodes_every_nibble_and_exponent_scale():
    packed = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    scales = (
        torch.arange(119, 135, dtype=torch.uint8)
        .reshape(16, 1)
        .view(torch.float8_e8m0fnu)
    )
    decoded = kernel.dequant_fp4(packed, scales)
    for row in range(16):
        for column in range(32):
            byte = int(packed[row, column // 2])
            nibble = (byte >> (4 * (column % 2))) & 15
            expected = (0, 0.5, 1, 1.5, 2, 3, 4, 6)[nibble & 7]
            expected *= (-1 if nibble & 8 else 1) * 2 ** (row - 8)
            assert decoded[row, column].item() == expected


def test_fp4_midpoint_ties_round_even_and_pack_low_nibble_first():
    values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6] * 4)
    packed, scales = kernel.fp4_act_quant(values.reshape(1, 32))
    assert packed.view(torch.uint8)[0].tolist() == [0x20, 0x42, 0x64, 0x76] * 4
    assert scales.view(torch.uint8).item() == 127


@pytest.mark.parametrize("block", [32, 128])
def test_fp8_gemm_scales_partial_output_block(block):
    a = torch.ones(2, 2 * block, dtype=torch.float32).to(torch.float8_e4m3fn)
    b = torch.ones(block + 3, 2 * block).to(torch.float8_e4m3fn)
    a_s = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    b_s = torch.tensor([[7.0, 11.0], [13.0, 17.0]])
    result = kernel.fp8_gemm(a, a_s, b, b_s, block_size=block)
    assert torch.all(result[0, :block] == block * (2 * 7 + 3 * 11))
    assert torch.all(result[1, block:] == block * (4 * 13 + 5 * 17))


def test_fp4_gemm_has_row_scales_independent_of_activation_block():
    a = torch.ones(2, 128).to(torch.float8_e4m3fn)
    b = torch.full((3, 64), 0x22, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    b_s = torch.tensor(
        [[1.0, 2.0, 4.0, 8.0], [2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]]
    )
    result = kernel.fp4_gemm(a, torch.tensor([[2.0], [3.0]]), b, b_s)
    torch.testing.assert_close(
        result, torch.tensor([[960.0, 512.0, 256.0], [1440.0, 768.0, 384.0]])
    )


def test_zero_quantization_has_finite_nonzero_scales_and_zero_values():
    for quant in (kernel.act_quant, kernel.fp4_act_quant):
        values = torch.zeros(2, 128)
        encoded, scales = quant(values)
        assert torch.all(kernel._scale_float(scales) > 0)
        assert torch.isfinite(kernel._scale_float(scales)).all()
        assert torch.all(encoded.view(torch.uint8) == 0)
        assert quant(values, inplace=True) is values
        assert not values.any()


def test_sparse_attention_matches_dense_sink_across_workspace_chunks():
    torch.manual_seed(8)
    q = torch.randn(2, 19, 3, 8)
    kv = torch.randn(2, 80, 8)
    idx = torch.randint(0, 80, (2, 19, 70))
    idx[:, 0] = -1
    idx[:, 1, :3] = -1
    sink = torch.tensor([-2.0, 0.0, 3.0])
    selected = kv[torch.arange(2)[:, None, None], idx.clamp_min(0)]
    scores = torch.einsum("bshd,bstd->bsht", q, selected) * 0.25
    scores.masked_fill_(idx.unsqueeze(-2) < 0, float("-inf"))
    logits = torch.cat((scores, sink.view(1, 1, 3, 1).expand(2, 19, 3, 1)), -1)
    probabilities = logits.softmax(-1)[..., :-1]
    expected = torch.einsum("bsht,bstd->bshd", probabilities, selected)
    actual = kernel.sparse_attn(q, kv, sink, idx, 0.25)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    split = kernel.sparse_attn(q, (kv[:, :11], kv[:, 11:]), sink, idx, 0.25)
    torch.testing.assert_close(split, expected, atol=1e-6, rtol=1e-5)
    assert not actual[:, 0].any()


def test_sinkhorn_matches_uniform_analytical_solution():
    pre, post, comb = kernel.hc_split_sinkhorn(
        torch.zeros(2, 3, 24), torch.ones(3), torch.zeros(24)
    )
    torch.testing.assert_close(pre, torch.full((2, 3, 4), 0.500001))
    torch.testing.assert_close(post, torch.ones(2, 3, 4))
    torch.testing.assert_close(
        comb, torch.full((2, 3, 4, 4), 0.25), atol=1e-6, rtol=1e-5
    )


def test_large_residual_is_normalized_before_fp16_compute():
    """A finite 70,000 residual must survive until RMS normalization."""
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from precision import install
    from reference import load_official

    official = load_official()
    block = SimpleNamespace(attn_norm=official.RMSNorm(2), ffn_norm=official.RMSNorm(2))
    install(official, block)
    residual = torch.full((1, 1, 2, 2), 60000.0, dtype=torch.float16)
    sublayer = torch.full((1, 1, 2), 10000.0, dtype=torch.float16)
    updated = block.hc_post(
        sublayer, residual, torch.ones(1, 1, 2), torch.eye(2).reshape(1, 1, 2, 2)
    )
    assert updated.dtype == torch.float32
    torch.testing.assert_close(updated, torch.full_like(updated, 70000.0))
    collapsed = updated.mean(2)
    for normalizer in (block.attn_norm, block.ffn_norm):
        normalized = normalizer(collapsed)
        assert normalized.dtype == torch.float16
        torch.testing.assert_close(normalized, torch.ones_like(normalized))


def test_bad_scale_layout_rejected_before_matmul():
    with pytest.raises(ValueError, match="FP8 weight scales"):
        kernel.fp8_gemm(
            torch.zeros(1, 32),
            torch.ones(1, 1),
            torch.zeros(33, 32),
            torch.ones(1, 1),
            block_size=32,
        )


@pytest.mark.parametrize("candidate_selection", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_official_tiny_model_prefill_matches_incremental_decode(
    candidate_selection, device
):
    """Exercise shared compressed KV and the sliding ring with synthetic weights."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from reference import load_official

    official = load_official()
    args = official.ModelArgs(
        max_batch_size=1,
        max_seq_len=32,
        temperature=0,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=256,
        dim=128,
        moe_inter_dim=64,
        n_layers=4,
        n_mtp_layers=0,
        n_heads=4,
        n_routed_experts=4,
        n_activated_experts=2,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=8,
        o_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=(0, 2, 2 if candidate_selection else 1, 0),
        kv_source_layers=(1,) if candidate_selection else (1, 2),
        index_source_layers=(1, 2),
        index_n_heads=4,
        index_head_dim=32,
        index_topk=2 if candidate_selection else 8,
        candidate_source_layer=1 if candidate_selection else -1,
        candidate_topk_blocks=2,
        candidate_block_size=2,
    )
    torch.manual_seed(17)
    model = official.Transformer(args).float().to(device).eval()
    model.head.forward = functools.partial(model.head.forward, full_logits=True)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight"):
                parameter.fill_(1)
            else:
                parameter.normal_(0, 0.02)
    tokens = torch.randint(0, 256, (1, 20), device=device)
    import copy

    from prefill import install

    chunked = copy.deepcopy(model)
    for layer in chunked.layers:
        install(official, layer.attn)
    with torch.inference_mode(), torch.device(tokens.device):
        _, full, _ = model(tokens)
        _, partial, _ = model(tokens[:, :8])
        _, chunk_partial, _ = chunked(tokens[:, :8])
        torch.testing.assert_close(chunk_partial, partial, atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(full[:, :8], partial, atol=1e-4, rtol=1e-3)
        for position in range(8, 20):
            _, partial, _ = model(tokens[:, position : position + 1], position)
            _, chunk_partial, _ = chunked(tokens[:, position : position + 1], position)
            torch.testing.assert_close(chunk_partial, partial, atol=1e-4, rtol=1e-3)
        position = 0
        for length in (3, 6, 2, 9):
            _, actual, _ = chunked(tokens[:, position : position + length], position)
            torch.testing.assert_close(
                actual, full[:, position : position + length], atol=1e-4, rtol=1e-3
            )
            position += length
        # Resident-model cases must resize batch/context and clear old state.
        from types import SimpleNamespace

        from loading import Stage

        wrapper = SimpleNamespace(
            official=official,
            args=args,
            layers=chunked.layers,
            device=tokens.device,
            hash_state=SimpleNamespace(cache=None),
        )
        _, reversed_full, _ = model(tokens.flip(1))
        Stage.reset_request(wrapper, 2, 48)
        _, resized, _ = chunked(torch.cat((tokens, tokens.flip(1))))
        torch.testing.assert_close(
            resized, torch.cat((full, reversed_full)), atol=1e-4, rtol=1e-3
        )
        Stage.reset_request(wrapper, 1, 12)
        _, shorter, _ = chunked(tokens[:, :8])
        torch.testing.assert_close(shorter, full[:, :8], atol=1e-4, rtol=1e-3)
    assert torch.isfinite(full).all() and torch.isfinite(partial).all()
    if not candidate_selection:
        torch.testing.assert_close(full[:, -1], partial[:, -1], atol=1e-4, rtol=1e-3)
    assert not any(name == "vllm" or name.startswith("vllm.") for name in sys.modules)


def test_pipeline_boundaries_preserve_shared_sources_and_decode_logits():
    """KV/index consumers across boundaries must match unpartitioned logits."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from pipeline import text_logits
    from reference import load_official

    official = load_official()
    args = official.ModelArgs(
        max_batch_size=1,
        max_seq_len=32,
        temperature=0,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=256,
        dim=128,
        moe_inter_dim=64,
        n_layers=6,
        n_mtp_layers=0,
        n_heads=4,
        n_routed_experts=4,
        n_activated_experts=2,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=8,
        o_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=(0, 2, 2, 2, 1, 1),
        kv_source_layers=(1, 4),
        index_source_layers=(1, 4),
        index_n_heads=4,
        index_head_dim=32,
        index_topk=8,
        dspark_target_layer_ids=(2, 5),
    )
    torch.manual_seed(31)
    model = official.Transformer(args).float().eval()
    model.head.forward = functools.partial(model.head.forward, full_logits=True)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight"):
                parameter.fill_(1)
            else:
                parameter.normal_(0, 0.02)
    tokens = torch.randint(0, 256, (1, 12))

    def transport(boundary, stage):
        copied = boundary.copy_to("cpu")
        assert copied.hidden.data_ptr() != boundary.hidden.data_ptr()
        if boundary.compress_kv is not None:
            assert copied.compress_kv.shape[1] <= tokens.size(1)
            assert copied.compress_kv.data_ptr() != boundary.compress_kv.data_ptr()
        official.shared_attn = official.SharedAttentionRuntime()
        return copied

    # Keep source caches for each path independent while interleaving forwards.
    import copy

    staged = copy.deepcopy(model)
    for position, chunk in (
        (0, tokens[:, :8]),
        (8, tokens[:, 8:9]),
        (9, tokens[:, 9:10]),
        (10, tokens[:, 10:11]),
    ):
        _, expected, expected_targets = model(chunk, position)
        actual, actual_targets = text_logits(
            official,
            staged,
            args,
            chunk,
            (2, 1, 2, 1),
            start_pos=position,
            transport=transport,
        )
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(
            actual_targets, expected_targets, atol=1e-4, rtol=1e-3
        )


def test_pipeline_cache_transport_appends_only_new_rows(monkeypatch):
    """Long-context steps retain received KV without retransmitting old rows."""
    from collections import deque

    sys.path.insert(0, str(Path(__file__).parents[1]))
    import distributed
    from pipeline import Boundary

    messages, transferred = deque(), []

    def send(value, **kwargs):
        messages.append(value.clone())
        transferred.append(value.numel() * value.element_size())

    def receive(value, **kwargs):
        value.copy_(messages.popleft())

    monkeypatch.setattr(distributed.dist, "send", send)
    monkeypatch.setattr(distributed.dist, "recv", receive)
    sender = distributed.Transport(None, "cpu")
    receiver = distributed.Transport(None, "cpu")
    sender.reset_request(100)
    receiver.reset_request(100)
    kv = torch.randn(2, 50, 32).half()
    keys = torch.randn(2, 50, 16).half()
    hidden = torch.randn(2, 1, 4, 32).half()
    mix = torch.randn(2, 1, 4)
    previous = 0
    for length in (10, 11, 11, 15):
        begin = previous
        boundary = Boundary(
            hidden, mix, kv[:, :length], keys[:, :length], None, None, None, ()
        )
        transferred.clear()
        sender.send(boundary, 1)
        actual = receiver.receive(0)
        torch.testing.assert_close(actual.compress_kv, kv[:, :length])
        torch.testing.assert_close(actual.index_k, keys[:, :length])
        assert actual.compress_kv.data_ptr() != kv.data_ptr()
        assert sum(transferred) == 7 * 8 * 8 + (
            hidden.numel() * 2 + mix.numel() * 4 + 2 * (length - begin) * 48 * 2
        )
        previous = length
    sender.reset_request(100)
    receiver.reset_request(100)
    sender.send(Boundary(hidden, mix, kv[:, :2], None, None, None, None, ()), 1)
    torch.testing.assert_close(receiver.receive(0).compress_kv, kv[:, :2])


def test_meta_model_rotary_buffers_materialize_with_real_values():
    """Cached meta rotary tables must never be reused for checkpoint loading."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from loading import materialize_buffers
    from reference import load_official

    official = load_official()
    args = official.ModelArgs(max_batch_size=1, max_seq_len=32)
    with torch.device("meta"):
        attention = official.Attention(0, args)
    materialize_buffers(official, attention, args, "cpu")
    assert not attention.freqs_cis.is_meta
    assert torch.isfinite(attention.freqs_cis).all()
    torch.testing.assert_close(
        attention.freqs_cis[0], torch.ones_like(attention.freqs_cis[0])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
def test_cuda_graph_replay_uses_changed_inputs():
    """The quantized path and sparse attention must not freeze values at capture."""
    device = "cuda"
    x = torch.ones(2, 128, device=device)
    packed = torch.full((3, 64), 0x22, dtype=torch.uint8, device=device).view(
        torch.float4_e2m1fn_x2
    )
    scales = torch.full((3, 4), 127, dtype=torch.uint8, device=device).view(
        torch.float8_e8m0fnu
    )
    q = torch.randn(1, 4, 2, 32, device=device)
    kv = torch.randn(1, 16, 32, device=device)
    indices = torch.arange(16, device=device).expand(1, 4, 16).int()
    sink = torch.zeros(2, device=device)
    mixes = torch.zeros(1, 4, 24, device=device)
    hc_scale = torch.ones(3, device=device)
    hc_base = torch.zeros(24, device=device)

    def run():
        a, a_s = kernel.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        product = kernel.fp4_gemm(a, a_s, packed, scales, act_block_size=32)
        attention = kernel.sparse_attn(q, kv, sink, indices, 32**-0.5)
        _, _, comb = kernel.hc_split_sinkhorn(mixes, hc_scale, hc_base)
        return product, attention, comb

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    x.fill_(2)
    q.mul_(2)
    graph.replay()
    expected = run()
    for actual, reference in zip(captured, expected):
        torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(captured[0], torch.full_like(captured[0], 256))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
@pytest.mark.parametrize(
    "tokens,rows,columns,block,threshold",
    [
        (1, 32, 32, 32, 128),
        (3, 67, 160, 32, 128),
        (17, 67, 256, 128, 128),
        (17, 67, 256, 128, 1),
    ],
)
def test_reused_qwen_fp4_matches_reference_and_graph(
    tokens, rows, columns, block, threshold, monkeypatch
):
    """Cover partial N/K tiles, split-K and more than sixteen activation rows."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from qwen_adapter import prepare_weight

    monkeypatch.setenv("WININTEL_FP4_PREFILL_ROWS", str(threshold))
    torch.manual_seed(29)
    a = torch.randn(tokens, columns, device="cuda").to(torch.float8_e4m3fn)
    a_s = torch.full(
        (tokens, columns // block), 122, dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e8m0fnu)
    packed = torch.randint(
        0, 256, (rows, columns // 2), dtype=torch.uint8, device="cuda"
    ).view(torch.float4_e2m1fn_x2)
    b_s = torch.randint(
        120, 125, (rows, columns // 32), dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e8m0fnu)
    expected = kernel.fp4_gemm(a, a_s, packed, b_s, act_block_size=block)
    prepared = prepare_weight(packed, b_s)
    assert prepared.packed.data_ptr() == packed.data_ptr()
    assert torch.equal(prepared.scales[:, ::2], prepared.scales[:, 1::2])
    decoded = kernel._activation(a, a_s, block).half()
    for begin in range(0, tokens, 16):
        activation = decoded[begin : begin + 16].contiguous()
        plain = prepared.extension.gemm(activation, prepared.packed, prepared.scales)
        compact = prepared.extension.gemm_lookup(
            activation, prepared.packed, prepared.scale_ids, prepared.scale_table
        )
        torch.testing.assert_close(compact, plain, atol=0, rtol=0)
    assert prepared.scale_ids.element_size() == 2
    with pytest.raises(ValueError, match="Reprepare"):
        kernel.fp4_gemm(a, a_s, packed, b_s.clone(), act_block_size=block)
    actual = kernel.fp4_gemm(a, a_s, packed, b_s, act_block_size=block)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kernel.fp4_gemm(a, a_s, packed, b_s, act_block_size=block)
    a.copy_(torch.ones((tokens, columns), device="cuda"))
    graph.replay()
    decoded_a = kernel._activation(a, a_s, block)
    reference = decoded_a @ kernel.dequant_fp4(packed, b_s).T
    torch.testing.assert_close(captured, reference, atol=1e-4, rtol=1e-3)
    from qwen_adapter import enable_cuda_graph

    enable_cuda_graph(True)
    try:
        first = kernel.fp4_gemm(a, a_s, packed, b_s, act_block_size=block)
        a.copy_(torch.full((tokens, columns), 2.0, device="cuda"))
        second = kernel.fp4_gemm(a, a_s, packed, b_s, act_block_size=block)
        torch.testing.assert_close(first, reference, atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(second, reference * 2, atol=1e-4, rtol=1e-3)
        assert prepared.graph is not None
    finally:
        enable_cuda_graph(False)


@pytest.mark.parametrize("dtype", [torch.float4_e2m1fn_x2, torch.float8_e4m3fn])
@pytest.mark.parametrize("limit", [0.0, 0.5])
def test_shared_expert_quantization_preserves_full_ffn_and_repeated_install(
    dtype, limit
):
    """Quantized gate/up reuse preserves clipping and routed output rounding."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from expert_graph import install
    from reference import load_official

    official = load_official()
    expert = official.Expert(128, 64, dtype=dtype, swiglu_limit=limit)
    with torch.no_grad():
        for layer in (expert.w1, expert.w2, expert.w3):
            if dtype == torch.float4_e2m1fn_x2:
                layer.weight.view(torch.uint8).random_(0, 256)
            else:
                layer.weight.copy_(torch.randn(layer.weight.shape))
            layer.scale.view(torch.uint8).fill_(123)
        x = torch.randn(17, 128)
        weights = torch.rand(17, 1)
        original = expert.forward
        expected = expert(x, weights)
        install(expert)
        forward = expert.forward
        install(expert)
        assert expert.forward is forward
        torch.testing.assert_close(expert(x, weights), expected, atol=0, rtol=0)
        x.mul_(0.5)
        torch.testing.assert_close(expert(x), original(x), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
@pytest.mark.parametrize("limit", [0.0, 0.5])
@pytest.mark.parametrize("shared_quant", ["0", "1"])
@pytest.mark.parametrize("shared_decode", ["0", "1"])
def test_complete_expert_graph_preserves_routing_before_output_quantization(
    limit, shared_quant, shared_decode, monkeypatch
):
    """Replay changed inputs/weights and recapture routed/shared decode shapes."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from expert_graph import install
    from qwen_adapter import cached_weight, enable_cuda_graph
    from reference import load_official

    monkeypatch.setenv("WININTEL_EXPERT_SHARED_QUANT", shared_quant)
    monkeypatch.setenv("WININTEL_EXPERT_SHARED_DECODE", shared_decode)
    official = load_official()
    torch.manual_seed(47)
    with torch.device("cuda"):
        expert = official.Expert(
            128, 64, dtype=torch.float4_e2m1fn_x2, swiglu_limit=limit
        )
    for linear in (expert.w1, expert.w2, expert.w3):
        with torch.no_grad():
            linear.weight.view(torch.uint8).random_(0, 256)
            linear.scale.view(torch.uint8).fill_(123)
        linear.weight._winintel_native = True
    original = expert.forward
    install(expert)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with torch.inference_mode():
            for tokens, routed in ((1, True), (1, True), (3, True), (3, False)):
                x = torch.randn(tokens, 128, device="cuda", dtype=torch.float16)
                weights = (
                    torch.rand(tokens, 1, device="cuda", dtype=torch.float32)
                    if routed
                    else None
                )
                enable_cuda_graph(False)
                expected = original(x, weights)
                torch.testing.assert_close(expert(x, weights), expected, atol=0, rtol=0)
                enable_cuda_graph(True)
                actual = expert(x, weights)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                cached_weight(expert.w1.weight, expert.w1.scale, limit=1)
                assert hasattr(expert.w2.weight, "_winintel_graph_fp4")
                retained = actual.clone()
                x.mul_(0.5)
                if weights is not None:
                    weights.mul_(0.25)
                replay = expert(x, weights)
                enable_cuda_graph(False)
                torch.testing.assert_close(replay, original(x, weights), atol=0, rtol=0)
                torch.testing.assert_close(actual, retained, atol=0, rtol=0)
    finally:
        enable_cuda_graph(False)
        torch.set_default_dtype(previous_dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
@pytest.mark.parametrize("block", [32, 128])
def test_fp8_adapter_partial_tiles_and_changed_graph_inputs(block):
    """Bounded E4M3 decoding preserves tails and changed inputs during replay."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from native_ops import prepare_fp8

    weight = torch.randn(291, 256, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.full(
        ((291 + block - 1) // block, 256 // block), 0.125, device="cuda"
    )
    x = torch.randn(17, 256, device="cuda")
    a, a_s = kernel.act_quant(x, block)
    expected = kernel.fp8_gemm(a, a_s, weight, scales, block_size=block)
    with torch.inference_mode():
        prepare_fp8(weight, scales)
        actual = kernel.fp8_gemm(a, a_s, weight, scales, block_size=block)
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.005)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            kernel.fp8_gemm(a, a_s, weight, scales, block_size=block)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = kernel.fp8_gemm(a, a_s, weight, scales, block_size=block)
        a.copy_(kernel.act_quant(x * 0.5, block)[0])
        a_s.mul_(0.5)
        graph.replay()
        torch.testing.assert_close(
            captured,
            kernel.fp8_gemm(a, a_s, weight, scales, block_size=block),
            atol=0,
            rtol=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
@pytest.mark.parametrize("block", [32, 128])
def test_fp8_decode_graph_tracks_inputs_checkpoint_updates_and_cache_disable(
    block, monkeypatch
):
    """A cached graph must neither freeze new activations nor outlive its weights."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    import native_ops
    from qwen_adapter import enable_cuda_graph

    native_ops._decoded_weights.clear()
    monkeypatch.setenv("WININTEL_FP8_CACHE_MIB", "1")
    weight = torch.randn(291, 256, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.full(
        ((291 + block - 1) // block, 256 // block), 0.125, device="cuda"
    )
    native_ops.prepare_fp8(weight, scales)
    retained = []
    try:
        with torch.no_grad():
            for rows in (1, 2, 2, 16):
                x = torch.randn(rows, 256, device="cuda")
                a, a_s = kernel.act_quant(x, block)
                enable_cuda_graph(False)
                expected = kernel.fp8_gemm(a, a_s, weight, scales, block_size=block)
                enable_cuda_graph(True)
                with torch.inference_mode():
                    actual = kernel.fp8_gemm(a, a_s, weight, scales, block_size=block)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                retained.append((actual, actual.clone()))
                a_s.mul_(0.5)
                alternate = torch.cuda.Stream()
                alternate.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(alternate):
                    changed = kernel.fp8_gemm(a, a_s, weight, scales, block_size=block)
                torch.cuda.current_stream().wait_stream(alternate)
                enable_cuda_graph(False)
                torch.testing.assert_close(
                    changed,
                    kernel.fp8_gemm(a, a_s, weight, scales, block_size=block),
                    atol=0,
                    rtol=0,
                )
                weight.copy_(torch.randn(weight.shape, device="cuda"))
                scales.mul_(0.5)
            monkeypatch.setenv("WININTEL_FP8_CACHE_MIB", "0")
            actual = kernel.fp8_gemm(a, a_s, weight, scales, block_size=block)
            enable_cuda_graph(False)
            torch.testing.assert_close(
                actual,
                kernel.fp8_gemm(a, a_s, weight, scales, block_size=block),
                atol=0,
                rtol=0,
            )
            assert not native_ops._decoded_weights
            for actual, saved in retained:
                torch.testing.assert_close(actual, saved, atol=0, rtol=0)
    finally:
        enable_cuda_graph(False)
        native_ops._decoded_weights.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
def test_fp8_linear_graph_preserves_official_input_rounding_and_checkpoint_updates(
    monkeypatch,
):
    """Raw-input capture must keep FP8 rounding and refreshed checkpoint values."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    import native_ops
    from qwen_adapter import enable_cuda_graph
    from reference import load_official

    monkeypatch.setenv("WININTEL_FP8_CACHE_MIB", "1")
    native_ops._decoded_weights.clear()
    official = load_official()
    original = official.linear
    with torch.device("cuda"):
        layer = official.Linear(128, 64, dtype=torch.float8_e4m3fn)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(layer.weight.shape, device="cuda"))
        layer.scale.view(torch.uint8).fill_(123)
    native_ops.prepare_fp8(layer.weight, layer.scale)
    native_ops.install_fp8_linear(official)
    try:
        with torch.no_grad():
            for rows in (1, 2, 2, 16):
                x = torch.randn(rows, 128, device="cuda", dtype=torch.float16)
                enable_cuda_graph(False)
                expected = original(x, layer.weight)
                enable_cuda_graph(True)
                with torch.inference_mode():
                    actual = layer(x)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                x.mul_(0.5)
                changed = layer(x)
                enable_cuda_graph(False)
                torch.testing.assert_close(
                    changed, original(x, layer.weight), atol=0, rtol=0
                )
                layer.weight.copy_(torch.randn(layer.weight.shape, device="cuda"))
                layer.scale.view(torch.uint8).sub_(1)
            monkeypatch.setenv("WININTEL_FP8_CACHE_MIB", "0")
            enable_cuda_graph(True)
            torch.testing.assert_close(
                layer(x), original(x, layer.weight), atol=0, rtol=0
            )
            assert not native_ops._decoded_weights
    finally:
        enable_cuda_graph(False)
        native_ops._decoded_weights.clear()
