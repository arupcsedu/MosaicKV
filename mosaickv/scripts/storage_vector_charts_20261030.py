"""Dependency-free vector PDF/SVG bars from measured tables, no new packages."""
# ruff: noqa: E501

from html import escape


class Canvas:
    def __init__(self, width=1000, height=420):
        self.width, self.height = width, height
        self.pdf = []
        self.svg = []

    def rect(self, x, y, width, height, color):
        r, g, b = tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
        self.pdf.append(f"{r:.4f} {g:.4f} {b:.4f} rg {x:.3f} {y:.3f} {width:.3f} {height:.3f} re f")
        self.svg.append(
            f'<rect x="{x}" y="{self.height - y - height}" width="{width}" height="{height}" fill="{color}"/>'
        )

    def text(self, x, y, text, size=11):
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        self.pdf.append(f"0.08 0.08 0.08 rg BT /F1 {size} Tf {x} {y} Td ({safe}) Tj ET")
        self.svg.append(
            f'<text x="{x}" y="{self.height - y}" font-family="sans-serif" font-size="{size}">{escape(text)}</text>'
        )

    def save(self, stem):
        stream = "\n".join(self.pdf).encode("ascii")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
        ]
        output = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for i, body in enumerate(objects, 1):
            offsets.append(len(output))
            output.extend(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
        start = len(output)
        output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode())
        output.extend(
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
        )
        stem.with_suffix(".pdf").write_bytes(output)
        stem.with_suffix(".svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}"><rect width="100%" height="100%" fill="white"/>'
            + "".join(self.svg)
            + "</svg>"
        )


def draw(frame, ratios, out):
    methods = (
        "full_kv",
        "prompt_attention_topk",
        "prefixkv_reimpl",
        "vl_cache_reimpl",
        "lookm_reimpl",
        "mosaickv_exact",
    )
    names = ("All-pos.+", "Top-K", "Prefix*", "VL-Cache*", "LOOK-M*", "EviCastKV")
    tasks = sorted(frame.task.unique())
    chart = Canvas()
    chart.text(
        35,
        396,
        "Consumed diagnostics: common source-slot capacity, NOT encoded-byte-matched curves",
        13,
    )
    for panel, task in enumerate(tasks):
        x0 = 55 + panel * 495
        sub = frame[(frame.task == task) & (frame.horizon == 32)].set_index("method")
        ymax = max(float(sub.observed_decode_owned_peak_bytes.max()) / 2**20 * 1.12, 1)
        chart.text(x0, 365, task, 12)
        chart.text(x0, 345, "MiB; observed decode-owned peak, NOT whole-request HBM", 9)
        for tick in range(5):
            y = 100 + tick * 55
            chart.rect(x0, y, 420, 0.4, "#d1d5db")
            chart.text(x0 - 32, y - 3, f"{ymax * tick / 4:.0f}", 9)
        fields = ("initial_bytes", "final_bytes", "observed_decode_owned_peak_bytes")
        colors = ("#93c5fd", "#3b82f6", "#1e3a8a")
        for index, method in enumerate(methods):
            x = x0 + 12 + index * 68
            for j, (field, color) in enumerate(zip(fields, colors, strict=True)):
                value = float(sub.loc[method, field]) / 2**20
                chart.rect(x + j * 14, 100, 12, 220 * value / ymax, color)
            chart.text(x - 3, 78, names[index], 8)
    for j, (label, color) in enumerate(
        zip(
            ("Initial", "Final", "Decode-owned peak"),
            ("#93c5fd", "#3b82f6", "#1e3a8a"),
            strict=True,
        )
    ):
        chart.rect(235 + j * 180, 40, 10, 10, color)
        chart.text(250 + j * 180, 40, label, 10)
    chart.text(
        35,
        16,
        "* Local adaptation. + All-position lossless control, NOT ordinary FullKV memory. Two consumed inputs only.",
        10,
    )
    chart.save(out / "storage_boundaries")
    chart = Canvas()
    chart.text(
        35,
        396,
        "Same-GPU randomized decode trials: prefill, selection and construction excluded",
        13,
    )
    for panel, task in enumerate(tasks):
        x0 = 55 + panel * 495
        selected = {r["method"]: r for r in ratios if r["task"] == task}
        ymax = max(r["decode_p95_seconds"] for r in selected.values()) * 1.2
        chart.text(x0, 365, task, 12)
        chart.text(x0, 345, "32-token decode seconds: p50, upper whisker p95", 10)
        for tick in range(5):
            y = 100 + tick * 55
            chart.rect(x0, y, 420, 0.4, "#d1d5db")
            chart.text(x0 - 32, y - 3, f"{ymax * tick / 4:.1f}", 9)
        for index, method in enumerate(methods[1:]):
            row = selected[method]
            x = x0 + 20 + index * 80
            height = 220 * row["decode_p50_seconds"] / ymax
            top = 100 + 220 * row["decode_p95_seconds"] / ymax
            chart.rect(x, 100, 38, height, "#c36d25" if method == "mosaickv_exact" else "#64748b")
            chart.rect(x + 18, 100 + height, 1, max(top - 100 - height, 0.5), "#111827")
            chart.rect(x + 13, top, 11, 0.7, "#111827")
            chart.text(x, top + 7, f"{row['decode_p50_seconds']:.3f}", 9)
            chart.text(x - 4, 78, names[index + 1], 9)
    chart.text(
        35,
        35,
        "Five warmups + twenty timed blocks; identical inputs/precision/runtime, unequal realized encoded bytes.",
        10,
    )
    chart.text(
        35,
        16,
        "Conditional timing evidence, not statistical benchmark-quality or end-to-end superiority evidence.",
        10,
    )
    chart.save(out / "paired_decode_latency")
