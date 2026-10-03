"""Load an explicitly supplied, hash-sealed serving bundle; never download model data."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def command(root):
    root = root.resolve()
    manifest = json.loads((root / "release.json").read_text())
    files = manifest["files"]
    entry = manifest["entrypoint"]
    if not isinstance(files, dict) or entry not in files:
        raise ValueError("Entry point must be in the sealed file map")
    for name, expected in files.items():
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root):
            raise ValueError("Bundle path escapes runtime directory")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("Runtime bundle hash mismatch")
    return [sys.executable, str(root / entry)]


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        print(json.dumps({"status": "DEPENDENCIES_ONLY_NOT_SERVING_ACCEPTANCE",
                          "versions": {n: importlib.metadata.version(n)
                                       for n in ("torch", "triton", "exllamav3", "safetensors")}}))
    elif sys.argv[1:] == ["--serve"]:
        argv = command(Path("/runtime"))
        os.execv(argv[0], argv)
    else:
        raise SystemExit("Use --check or mount a sealed /runtime bundle and use --serve")
