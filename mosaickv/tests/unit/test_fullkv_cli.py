from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaickv.cli import build_parser
from mosaickv.fullkv_cli import load_workload, select_workload


def _write_workload(path: Path, image: Path, *, count: int = 3) -> None:
    rows = [
        {
            "sample_id": f"sample-{index}",
            "prompt": f"prompt {index}",
            "media": [{"kind": "image", "path": image.name}],
        }
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_workload_loading_and_selection_are_strict_and_deterministic(tmp_path: Path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"fixture")
    workload = tmp_path / "workload.jsonl"
    _write_workload(workload, image)
    rows = load_workload(workload)
    assert len(rows) == 3
    assert rows[0].media[0].paths == (image.resolve(),)
    assert select_workload(rows, seed=7, count=2) == select_workload(
        tuple(reversed(rows)), seed=7, count=2
    )


def test_workload_rejects_unknown_fields(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text(
        json.dumps({"sample_id": "x", "prompt": "q", "media": [], "extra": 1}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fields"):
        load_workload(source)


@pytest.mark.parametrize(
    "command",
    ["fullkv-debug", "fullkv-smoke", "fullkv-run", "fullkv-latency"],
)
def test_fullkv_commands_are_registered(command: str) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            command,
            "--config",
            "config.toml",
            "--workload",
            "workload.jsonl",
            "--run-id",
            "run",
            "--raw-output",
            "raw.jsonl",
            "--aggregate-output",
            "aggregate.json",
            "--log-output",
            "log.json",
            "--manifest",
            "manifest.json",
            "--cache-root",
            "/scratch/cache",
        ]
    )
    assert args.fullkv_mode in {"debug", "smoke20", "dataset", "latency"}
