"""Shared utilities: seeding, config loading, run provenance, hashing."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def file_sha1(path: str | Path, nbytes: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(nbytes):
            h.update(chunk)
    return h.hexdigest()[:16]


def array_hash(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


@dataclass
class RunContext:
    """Captures provenance for a run: seed, git commit, config hash."""

    run_dir: Path
    config: dict[str, Any]
    seed: int

    def save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "seed": self.seed,
            "git_commit": git_commit(),
            "config": self.config,
            "env": {
                "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
        }
        save_json(meta, self.run_dir / "run_meta.json")
        save_json(self.config, self.run_dir / "config.json")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]
