# Author: 数萌AI服务器 (winintel)
"""Render measured benchmark results as a readable Markdown report."""

import argparse
import json
from pathlib import Path


def render(data):
    visible = [
        case for case in data["cases"] if not case["name"].startswith("profile-")
    ]
    cases = [case for case in visible if case.get("status") != "not_run"]
    refused = [case for case in visible if case.get("status") == "not_run"]
    sources = {
        source
        for case in cases
        for rank in case.get("gpu_memory", [])
        for source in rank.get("engram_row_sources", [])
    }
    storage = (
        "Engram rows come from local checkpoint files, including any mounted "
        "filesystem. Bounded row reads and cache misses are included in latency."
        if sources and sources <= {"TensorFile", "CachedTensorFile"}
        else "The development Engram path reads original FP8 rows through a "
        "persistent domestic-mirror range cache. Cold row fetches are "
        "included in measured latency."
    )
    lines = [
        "# winintel AI · DeepSeek V4.1 IB benchmark",
        "",
        "Author: 数萌AI服务器 (winintel)",
        "",
        (
            "| Case | Input tokens/request | Concurrent requests | First token (s) | "
            "Prefill (tokens/s) | Fixed-step decode total (tokens/s) | "
            "Decode/request (tokens/s) | Retrieval keys found | "
            "Peak GPU allocation (GiB) |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        memory = (
            max(row["allocated_peak_bytes"] for row in case["gpu_memory"]) / 1024**3
        )
        found = sum(case["retrieval_key_present"])
        decode = case["decode_tokens_per_second_total"]
        per_request = case["decode_tokens_per_second_per_request"]
        lines.append(
            f"| {case['name']} | {case['input_tokens_per_request']:,} | "
            f"{case['concurrency']} | "
            f"{case['first_token_seconds']:.3f} | "
            f"{case['prefill_effective_tokens_per_second']:.2f} | "
            f"{decode:.2f} | {per_request:.2f} | "
            f"{found}/{case['concurrency']} | {memory:.2f} |"
        )
    http = bool(cases) and all(case.get("http_sessions") for case in cases)
    execution = (
        "Concurrent requests use independent HTTP SSE connections queued into "
        "equal-length batches. The client table includes queueing and HTTP delivery."
        if http
        else "Measurements use distinct requests in synchronous batches. "
        "They do not measure independent HTTP sessions or scheduler latency."
    )
    lines.extend(
        [
            "",
            (
                f"Model load: {data['model_load_seconds']:.3f} seconds. "
                f"TP: {data['tp']}; pipeline layers: {data['pipeline_layers']}."
            ),
            "",
            (
                execution + " Effective prefill throughput includes the first step. "
                "Decode throughput excludes the first token."
                " The workload runs a fixed number of decode steps, including "
                "steps after EOS; this is not a count of useful response tokens."
            ),
            "",
            (
                storage + " Native FP4 matrix CUDA Graphs are "
                "enabled; full-model CUDA Graph capture is not qualified. "
                "Retrieval-key presence is a narrow check, not a model accuracy score."
            ),
            "",
        ]
    )
    if refused:
        lines.extend(
            [
                "| Case not run | Reason |",
                "| --- | --- |",
            ]
        )
        lines.extend(f"| {case['name']} | {case['reason']} |" for case in refused)
        lines.extend(
            ["", "Capacity figures are admission estimates, not measurements.", ""]
        )
    if http:
        lines.extend(
            [
                "| Case | HTTP session | First token (s) | Decode (tokens/s) |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for case in cases:
            for index, client in enumerate(case["http_sessions"]):
                times = client["token_elapsed_seconds"]
                speed = (
                    (len(times) - 1) / (times[-1] - times[0])
                    if len(times) > 1 and times[-1] > times[0]
                    else 0
                )
                lines.append(
                    f"| {case['name']} | {index + 1} | "
                    f"{client['first_token_seconds']:.3f} | {speed:.2f} |"
                )
        lines.append("")
    if any("response_decode_tokens_per_second_total" in case for case in cases):
        lines.extend(
            [
                "| Case | Useful decode total (tokens/s) | Tokens before EOS/request |",
                "| --- | ---: | --- |",
            ]
        )
        for case in cases:
            speed = case.get("response_decode_tokens_per_second_total")
            value = f"{speed:.2f}" if speed is not None else "N/A"
            counts = ", ".join(
                str(count) for count in case.get("generated_tokens_before_eos", [])
            )
            lines.append(f"| {case['name']} | {value} | {counts} |")
        lines.extend(
            [
                "",
                (
                    "Useful decode counts exclude EOS and subsequent tokens; time runs "
                    "from the first token until all responses reach EOS or the output "
                    "limit. Profiling warmup cases are omitted."
                ),
                "",
            ]
        )
    lines.extend(["Generated output samples:", ""])
    for case in cases:
        lines.append(f"**{case['name']}**")
        lines.append("")
        for request, text in enumerate(case["generated_text"]):
            lines.append(f"Request {request}:")
            lines.append("")
            text = text.split("<｜end▁of▁sentence｜>")[0]
            lines.extend("> " + row for row in text.splitlines())
            lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    options.output.write_text(
        render(json.loads(options.input.read_text(encoding="utf-8"))),
        encoding="utf-8",
    )
