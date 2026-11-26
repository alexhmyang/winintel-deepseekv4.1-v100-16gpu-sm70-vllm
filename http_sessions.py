# Author: 数萌AI服务器 (winintel)
"""Independent HTTP streams queued into equal-length resident model batches."""

import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Session:
    def __init__(self, rows, new_tokens):
        self.id = uuid.uuid4().hex
        self.rows, self.new_tokens = rows, new_tokens
        self.events = queue.Queue()


class StreamingServer:
    """Own independent streams; only the model thread consumes queued requests."""

    def __init__(self, tokenizer, host="127.0.0.1", port=0):
        self.tokenizer = tokenizer
        self.pending = queue.Queue()
        self.deferred = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.path != "/v1/completions":
                    self.send_error(404)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 32 * 1024 * 1024:
                        raise ValueError("Invalid request body size")
                    data = json.loads(self.rfile.read(size))
                    prompt = data["prompt"]
                    rows = (
                        owner.tokenizer.encode(prompt, add_special_tokens=False)
                        if isinstance(prompt, str)
                        else prompt
                    )
                    tokens = data.get("max_tokens", 64)
                    if (
                        not isinstance(rows, list)
                        or not rows
                        or any(type(token) is not int or token < 0 for token in rows)
                        or type(tokens) is not int
                        or tokens < 1
                    ):
                        raise ValueError("Supply a prompt and positive max_tokens")
                except (KeyError, ValueError, TypeError) as error:
                    self.send_error(400, str(error))
                    return
                session = Session(rows, tokens)
                owner.pending.put(session)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    while True:
                        event = session.events.get()
                        content = "[DONE]" if event is None else json.dumps(event)
                        self.wfile.write(f"data: {content}\n\n".encode())
                        self.wfile.flush()
                        if event is None:
                            break
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def address(self):
        return self.server.server_address

    def batch(self, maximum=16, wait_seconds=0.05):
        first = self.deferred.pop(0) if self.deferred else self.pending.get()
        selected = [first]
        deadline = time.monotonic() + wait_seconds
        while len(selected) < maximum:
            try:
                candidate = self.pending.get(
                    timeout=max(0, deadline - time.monotonic())
                )
            except queue.Empty:
                break
            if (
                len(candidate.rows) == len(first.rows)
                and candidate.new_tokens == first.new_tokens
            ):
                selected.append(candidate)
            else:
                self.deferred.append(candidate)
        return selected

    def token(self, sessions, token_ids, step):
        for session, token in zip(sessions, token_ids):
            session.events.put(
                {
                    "id": session.id,
                    "object": "text_completion",
                    "choices": [
                        {
                            "index": 0,
                            "text": self.tokenizer.decode([token]),
                            "token_id": token,
                            "finish_reason": None,
                        }
                    ],
                    "step": step,
                }
            )

    def finish(self, sessions, error=None):
        for session in sessions:
            if error:
                session.events.put({"id": session.id, "error": error})
            session.events.put(None)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def collect_stream(address, rows, tokens):
    """Measure one independent HTTP connection without a first-token timeout."""
    import http.client

    connection = http.client.HTTPConnection(*address, timeout=None)
    started = time.monotonic()
    connection.request(
        "POST",
        "/v1/completions",
        json.dumps({"prompt": rows, "max_tokens": tokens}),
        {"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(response.read().decode())
    events, times = [], []
    for line in response:
        if not line.startswith(b"data: "):
            continue
        payload = line[6:].strip()
        if payload == b"[DONE]":
            break
        event = json.loads(payload)
        if "error" in event:
            raise RuntimeError(event["error"])
        events.append(event)
        times.append(time.monotonic() - started)
    connection.close()
    return {
        "request_id": events[0]["id"] if events else None,
        "token_ids": [event["choices"][0]["token_id"] for event in events],
        "token_elapsed_seconds": times,
        "first_token_seconds": times[0] if times else None,
        "completion_seconds": time.monotonic() - started,
    }


def run_case(stage, transport, tokenizer, options, case):
    """Send benchmark prompts through separate connections to the resident model."""
    from concurrent.futures import ThreadPoolExecutor

    import torch.distributed as dist
    from benchmark import prompts
    from benchmark import run_case as generate

    rank = dist.get_rank()
    server, sessions, pool, futures = None, [], None, []
    payload = [None]
    if rank == 0:
        rows, markers = prompts(tokenizer, case)
        server = StreamingServer(tokenizer)
        pool = ThreadPoolExecutor(max_workers=len(rows))
        futures = [
            pool.submit(
                collect_stream,
                server.address,
                row,
                case.get("new_tokens", options.new_tokens),
            )
            for row in rows
        ]
        # Collect all connections before inference, retaining their actual order.
        while len(sessions) < len(rows):
            sessions.extend(server.batch(maximum=len(rows) - len(sessions)))
        lookup = {tuple(row): marker for row, marker in zip(rows, markers)}
        payload[0] = (
            [session.rows for session in sessions],
            [lookup[tuple(session.rows)] for session in sessions],
        )
    dist.broadcast_object_list(payload, src=0)
    try:
        result = generate(
            stage,
            transport,
            tokenizer,
            options,
            case,
            supplied_prompts=payload[0],
            on_token=(
                (lambda ids, step: server.token(sessions, ids, step))
                if rank == 0
                else None
            ),
        )
        if rank == 0:
            server.finish(sessions)
            clients = [future.result() for future in futures]
            by_id = {client["request_id"]: client for client in clients}
            ordered = [by_id[session.id] for session in sessions]
            if any(
                client["token_ids"] != output
                for client, output in zip(ordered, result["generated_token_ids"])
            ):
                raise RuntimeError("HTTP stream tokens differ from model outputs")
            result["http_sessions"] = ordered
            result["execution"] = (
                "independent HTTP SSE connections queued into an equal-length batch"
            )
        return result
    finally:
        if rank == 0:
            server.finish(sessions, error="Model batch ended")
            pool.shutdown(wait=True)
            server.close()
