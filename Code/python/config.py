from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoPaths:
    repo_root: Path
    data_raw: Path
    data_derived: Path
    output: Path


@dataclass(frozen=True)
class RunConfig:
    data_type: str = "simulated"
    extension: str = "png"


def load_paths() -> RepoPaths:
    repo_root = Path(__file__).resolve().parents[2]
    data_raw = repo_root / "Data-raw"
    data_derived = repo_root / "Data-derived"
    output = repo_root / "Output"
    data_derived.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    return RepoPaths(
        repo_root=repo_root,
        data_raw=data_raw,
        data_derived=data_derived,
        output=output,
    )
