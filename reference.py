# Author: 数萌AI服务器 (winintel)
"""Load the pinned official architecture with the winintel reference backend."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def load_official():
    root = Path(__file__).resolve().parent
    model_file = root / "upstream" / "model.py"
    manifest = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["model_reference"]["files"].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"{filename} does not match the pinned source")
    for name in ("kernel", "engram", "vision", "image_processor"):
        existing = sys.modules.get(name)
        expected_path = root / (
            "kernel.py" if name == "kernel" else f"upstream/{name}.py"
        )
        if existing is not None and Path(existing.__file__).resolve() != expected_path:
            raise RuntimeError(f"A different {name} module is already loaded")
    sys.path[:0] = [str(root), str(root / "upstream")]
    spec = importlib.util.spec_from_file_location("winintel_official_model", model_file)
    model = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = model
    spec.loader.exec_module(model)
    if model.act_quant.__module__ != "kernel":
        raise RuntimeError("Official architecture did not bind the local backend")
    return model
