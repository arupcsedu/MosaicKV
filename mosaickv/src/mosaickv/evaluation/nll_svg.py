"""Dependency-free SVG renderer for paired, per-task cumulative NLL diagnostics."""

from html import escape
from math import isfinite


def cumulative_nll_svg(comparisons):
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    panels = []
    for panel, task in enumerate(("llava_description", "mmvet_prefixkv")):
        curves = []
        for row in comparisons:
            if row["task"] != task or row["reference"] != "full_kv_reference":
                continue
            total, values = 0.0, []
            for delta in row["per_token_deltas"]:
                if not isfinite(delta):
                    raise ValueError("SVG cannot render nonfinite NLL")
                total += delta
                values.append(total)
            curves.append((row, values))
        if not curves:
            raise ValueError("both task panels require paired FullKV comparisons")
        all_values = [0.0] + [v for _, values in curves for v in values]
        low, high = min(all_values), max(all_values)
        span = max(high - low, 1e-9)
        low, high = low - span * 0.08, high + span * 0.08
        n = max(len(values) for _, values in curves)
        left, top, width, height = 75 + panel * 630, 50, 515, 255

        def x(i, left=left, width=width, n=n):
            return left + width * (i - 1) / max(n - 1, 1)

        def y(v, top=top, height=height, high=high, low=low):
            return top + height * (high - v) / (high - low)

        lines = [f'<text x="{left}" y="24" font-size="16">{escape(task)}</text>']
        lines.append(
            f'<rect x="{left}" y="{top}" width="{width}" height="{height}" '
            'fill="white" stroke="black"/>'
        )
        for tick in range(5):
            value = low + (high - low) * tick / 4
            lines.append(
                f'<text x="{left - 7}" y="{y(value):.2f}" text-anchor="end" '
                f'font-size="10">{value:.2f}</text>'
            )
            position = 1 + (n - 1) * tick / 4
            lines.append(
                f'<text x="{x(position):.2f}" y="{top + height + 18}" '
                f'text-anchor="middle" font-size="10">{position:.0f}</text>'
            )
        lines.append(
            f'<line x1="{left}" x2="{left + width}" y1="{y(0):.2f}" y2="{y(0):.2f}" '
            'stroke="grey" stroke-dasharray="3 3"/>'
        )
        for index, (row, values) in enumerate(curves):
            color = colors[index % len(colors)]
            path = " ".join(
                f"{'M' if i == 1 else 'L'}{x(i):.2f},{y(v):.2f}"
                for i, v in enumerate(values, start=1)
            )
            lines.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.5"/>')
            lines.append(
                f'<text x="{left}" y="{350 + 16 * index}" fill="{color}" '
                f'font-size="11">{escape(row["role"])}</text>'
            )
            if row["role"] == "window_topk_control":
                boundary = row["first_generated_eviction_affected_prediction"]
                if boundary is not None:
                    lines.append(
                        f'<line data-boundary="fifo" x1="{x(boundary):.2f}" '
                        f'x2="{x(boundary):.2f}" y1="{top}" y2="{top + height}" '
                        'stroke="black" stroke-dasharray="5 3"/>'
                    )
                    lines.append(
                        f'<text x="{x(boundary) + 4:.2f}" y="{top + 13}" '
                        f'font-size="10">first FIFO effect: {boundary}</text>'
                    )
        lines.append(
            f'<text x="{left + width / 2}" y="337" text-anchor="middle" '
            'font-size="12">Teacher target position (1-based)</text>'
        )
        panels.extend(lines)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="430" viewBox="0 0 1260 430">'
        "<title>Cumulative NLL delta versus FullKV, in nats; two consumed inputs</title>"
        '<rect width="1260" height="430" fill="white"/>'
        '<text x="8" y="420" font-size="11">Y axis: cumulative NLL delta (nats). '
        'Temporal alignment is not causal proof. No statistical claim.</text>'
        + "".join(panels)
        + "</svg>"
    )
