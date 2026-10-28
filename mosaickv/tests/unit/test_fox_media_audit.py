import io
import tarfile
from pathlib import Path

import pytest
from PIL import Image


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import evicastkv_media_audit

    return evicastkv_media_audit


def test_question_type_does_not_parse_answers(monkeypatch):
    m = module(monkeypatch)
    # Invalid/non-JSON data after the metadata object must never be parsed.
    raw = (
        b'{"meta_data":{"question_type":"multi-choice","instruction":"} escaped \\""},'
        b'"data":SECRET_NON_JSON'
    )
    assert m.question_type_only(raw) == "multi-choice"
    with pytest.raises(ValueError):
        m.question_type_only(b'{"data":[],"meta_data":{"question_type":"multi-choice"}}')


def test_pixel_hash_catches_differently_encoded_identical_images(monkeypatch):
    m = module(monkeypatch)
    image = Image.new("RGB", (3, 2), (10, 20, 30))
    a, b = io.BytesIO(), io.BytesIO()
    image.save(a, format="PNG", compress_level=0)
    image.save(b, format="PNG", compress_level=9)
    aa, bb = m.image_identity(a.getvalue()), m.image_identity(b.getvalue())
    assert aa["encoded_sha256"] != bb["encoded_sha256"]
    assert aa["pixel_sha256"] == bb["pixel_sha256"]


def test_requested_archive_symlink_rejected_without_extraction(tmp_path, monkeypatch):
    m = module(monkeypatch)
    archive = tmp_path / "test.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("./IEdit/images/a.jpg")
        info.type = tarfile.SYMTYPE
        info.linkname = "/outside"
        tar.addfile(info)
    with pytest.raises(ValueError, match="regular file"):
        m.scan_archive(archive, {"IEdit/images/a.jpg"})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["test.tar.gz"]
