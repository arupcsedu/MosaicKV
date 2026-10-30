import xml.etree.ElementTree as ET

import storage_vector_charts_20261030 as module


def test_dependency_free_vectors_have_valid_lengths_offsets_and_escaped_text(tmp_path):
    chart = module.Canvas(200, 120)
    chart.rect(10, 20, 30, 40, "#c36d25")
    chart.text(10, 90, "kept (K/V) & exact <source>")
    stem = tmp_path / "chart"
    chart.save(stem)
    pdf = stem.with_suffix(".pdf").read_bytes()
    assert pdf.startswith(b"%PDF-1.4") and pdf.endswith(b"%%EOF\n")
    start = int(pdf.split(b"startxref\n")[1].splitlines()[0])
    assert pdf[start:].startswith(b"xref\n")
    assert b"kept \\(K/V\\)" in pdf
    svg = ET.parse(stem.with_suffix(".svg")).getroot()
    assert svg.attrib["width"] == "200"
    assert "kept (K/V) & exact <source>" in "".join(svg.itertext())
