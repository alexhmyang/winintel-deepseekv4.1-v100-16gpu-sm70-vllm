# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Verify source contents against the published SHA256 manifest."""

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "winintel-independent-v1":
        raise ValueError("Unsupported manifest schema")
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"Unsafe manifest path: {name}")
        path = root / name
        if path.is_symlink():
            raise ValueError(f"Symlink in release: {name}")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"SHA256 mismatch: {name}")
        if path.suffix == ".py":
            ast.parse(content)
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    for name, expected in provenance["model_reference"]["files"].items():
        if manifest["files"].get(name) != expected:
            raise ValueError(f"Official reference mismatch: {name}")
    operator = provenance["reused_local_operator"]
    if manifest["files"].get(operator["vendored_file"]) != operator["sha256"]:
        raise ValueError("Original operator mismatch")
    ignored_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in ignored_dirs for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink: {relative}")
        if not path.is_file():
            continue
        name = relative.as_posix()
        if name in manifest["files"] or name == "manifest.json":
            continue
        if (
            path.name == "cluster.env"
            or path.suffix in {".log", ".pyc"}
            or path.name.startswith(("results", "profile"))
            and path.suffix == ".json"
        ):
            continue
        raise ValueError(f"Unmanifested release file: {name}")
    print(f"PASS: {len(manifest['files'])} source files and provenance verified")


if __name__ == "__main__":
    main()
