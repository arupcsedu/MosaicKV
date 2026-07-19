"""Structured JSON logging helpers."""

from __future__ import annotations

import json
import logging as std_logging
import sys
from datetime import UTC, datetime
from typing import TextIO

from mosaickv.types import JsonObject, JsonValue

_STANDARD_RECORD_FIELDS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


def _json_safe(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


class JsonFormatter(std_logging.Formatter):
    """Serialize each log record as one JSON object."""

    def format(self, record: std_logging.LogRecord) -> str:
        payload: JsonObject = {
            "timestamp_utc": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS and not key.startswith("_"):
                payload[key] = _json_safe(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def configure_logging(
    level: str | int = "INFO", stream: TextIO | None = None
) -> std_logging.Logger:
    """Configure the isolated MosaicKV logger and return it."""

    logger = std_logging.getLogger("mosaickv")
    logger.handlers.clear()
    handler = std_logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: str) -> std_logging.Logger:
    """Return a child of the structured MosaicKV logger."""

    if name == "mosaickv":
        return std_logging.getLogger(name)
    suffix = name.removeprefix("mosaickv.")
    return std_logging.getLogger(f"mosaickv.{suffix}")


__all__ = ["JsonFormatter", "configure_logging", "get_logger"]
