import json

import pytest
from evicastkv_logit_composition_audit import status_header


def test_sorted_header_does_not_parse_scored_rows(tmp_path):
    path = tmp_path / "decision.json"
    # Deliberately malformed sample rows cannot be parsed or consulted.
    path.write_text(
        '{\n  "complete": true,\n  "model_level_diagnostic_pass": false,\n'
        '  "rows": [ THIS IS NOT JSON AND MUST NOT BE READ\n'
    )
    assert status_header(path) == dict(complete=True, model_level_diagnostic_pass=False)
    path.write_text(json.dumps(dict(complete=True)))
    with pytest.raises(ValueError):
        status_header(path)
