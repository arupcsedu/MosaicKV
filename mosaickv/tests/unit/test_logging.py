from __future__ import annotations

import io
import json
import logging

from mosaickv.logging import configure_logging, get_logger


def test_structured_json_logging() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream)
    logger = get_logger("mosaickv.test")
    logger.info("hello", extra={"event": "unit_test", "count": 3})

    record = json.loads(stream.getvalue())
    assert record["level"] == "INFO"
    assert record["message"] == "hello"
    assert record["event"] == "unit_test"
    assert record["count"] == 3
    assert record["logger"] == "mosaickv.test"


def test_non_json_extra_is_stringified() -> None:
    stream = io.StringIO()
    configure_logging(logging.INFO, stream)
    get_logger("test").info("path", extra={"value": object()})
    record = json.loads(stream.getvalue())
    assert isinstance(record["value"], str)
