import json
from xml.etree import ElementTree as ET

import evicastkv_consolidated_history as entry
import pytest

from mosaickv.evaluation.fox_campaign import sha


def setup(tmp_path, monkeypatch, *, source="a" * 40, tests=2674, bad_digest=False):
    document = ET.Element("testsuites")
    suite = ET.SubElement(
        document, "testsuite", tests=str(tests), failures="0", errors="0", skipped="0"
    )
    for i in range(tests):
        ET.SubElement(suite, "testcase", name=f"synthetic_fixture_{i}")
    xml = tmp_path / "cpu-regression.xml"
    ET.ElementTree(document).write(xml)
    record = dict(
        source_sha=source,
        returncode=0,
        cpu_xml=str(xml),
        cpu_xml_sha256="0" * 64 if bad_digest else sha(xml),
    )
    (tmp_path / "cpu_record.json").write_text(json.dumps(record))
    (tmp_path / "plan.json").write_text(json.dumps(dict(source_sha="a" * 40)))
    monkeypatch.setattr(entry.base.common, "inputs", lambda *_: dict(source_sha="a" * 40))
    calls = []
    monkeypatch.setattr(entry.base.launch, "submit", lambda *a, **k: calls.append((a, k)))
    return xml, calls


@pytest.mark.parametrize("kwargs", [dict(source="b" * 40), dict(tests=1), dict(bad_digest=True)])
def test_bad_or_obsolete_cpu_proof_cannot_launch(tmp_path, monkeypatch, kwargs):
    xml, calls = setup(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(ValueError):
        entry.admit(tmp_path, "f" * 64, xml)
    assert calls == []


def test_exact_cpu_proof_permits_only_the_registered_script(tmp_path, monkeypatch):
    xml, calls = setup(tmp_path, monkeypatch)
    entry.admit(tmp_path, "f" * 64, xml)
    assert len(calls) == 1
    assert calls[0][1]["script_name"] == "evicastkv_consolidated_history.sbatch"
    assert calls[0][1]["worker_minutes"] == 30
    assert (tmp_path / "cpu_admission.json").exists()
