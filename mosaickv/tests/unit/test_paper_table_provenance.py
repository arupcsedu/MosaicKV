from __future__ import annotations

from pathlib import Path


def _tables(source: str) -> tuple[str, ...]:
    tables: list[str] = []
    offset = 0
    while True:
        start = source.find(r"\begin{table", offset)
        if start < 0:
            return tuple(tables)
        end = source.find(r"\end{table", start)
        if end < 0:
            raise AssertionError("unterminated LaTeX table")
        tables.append(source[start:end])
        offset = end + 1


def test_published_baseline_tables_do_not_use_ambiguous_method_labels() -> None:
    repo = Path(__file__).resolve().parents[3]
    paths = (
        repo / "mosaickv/tables/published_baseline_fidelity.tex",
        repo / "mosaickv/tables/quality_by_model_task.tex",
        repo / "mosaickv/tables/memory_by_model_task.tex",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for table in _tables(source):
            for family in ("LOOK-M", "VL-Cache", "PrefixKV"):
                if family in table:
                    assert "reimpl" in table.casefold() or "official" in table.casefold()
