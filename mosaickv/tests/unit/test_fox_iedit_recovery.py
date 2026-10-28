import io
import json
import tarfile
from pathlib import Path

from PIL import Image


def test_one_archive_asset_audit_never_decodes_responses_or_extracts(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import evicastkv_iedit_recovery as module
    import huggingface_hub

    from mosaickv.evaluation.fox_campaign import sha, write_once

    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, format="PNG")
    archive = tmp_path / "archive.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, raw in (
            (
                "IEdit/IEdit.json",
                b'{"meta_data":{"question_type":"open-ended"},"data":FORBIDDEN_NOT_JSON',
            ),
            ("IEdit/images/0.png", image.getvalue()),
            ("IEdit/combined_1_images/0.png", image.getvalue()),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    scope, root = tmp_path / "scope", tmp_path / "run"
    history = tmp_path / "history.json"
    write_once(
        scope / "catalogs/IEdit.json",
        dict(
            rows=[
                dict(
                    sample_id="synthetic",
                    source_image_paths=["0.png"],
                    combined_1_images="0.png",
                )
            ]
        ),
    )
    write_once(history, dict(identities={}))
    write_once(
        root / "protocol.json",
        dict(
            source_sha="synthetic",
            catalog_sha256=sha(scope / "catalogs/IEdit.json"),
            historical_media_sha256=sha(history),
        ),
    )
    monkeypatch.setattr(module, "clean_source", lambda *a: "synthetic")
    monkeypatch.setattr(module, "SCOPE", scope)
    monkeypatch.setattr(module, "HISTORY", history)
    monkeypatch.setattr(module, "SIZE", archive.stat().st_size)
    monkeypatch.setattr(module, "SHA", sha(archive))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    calls = []

    class Remote:
        def info(self, path):
            return dict(size=archive.stat().st_size)

        def open(self, *args, **kwargs):
            calls.append((args, kwargs))
            return archive.open("rb")

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", Remote)
    module.worker(root)
    report = json.loads((root / "report.json").read_text())
    assert len(calls) == 1
    assert report["complete_media_samples"] == 1
    assert report["official_question_type_verified"]
    assert not report["archive_extracted"] and not report["responses_decoded"]
    assert not report["confirmation_authorized"]
    assert not (tmp_path / "IEdit").exists()
