#!/usr/bin/env python3
"""Structured trace logging for STS2 agent runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip_text(value: str, limit: int = 320) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _summarize(value: Any, depth: int = 0) -> Any:
    if depth >= 3:
        return _clip_text(str(value), 180)
    if isinstance(value, dict):
        summary: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 24:
                summary["..."] = f"{len(value) - index} more keys"
                break
            summary[str(key)] = _summarize(item, depth + 1)
        return summary
    if isinstance(value, list):
        items = [_summarize(item, depth + 1) for item in value[:8]]
        if len(value) > 8:
            items.append(f"... {len(value) - 8} more items")
        return items
    if isinstance(value, str):
        return _clip_text(value)
    return value


class AgentTraceRecorder:
    """Simple JSONL trace recorder with LangSmith-like node events."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.run_id: Optional[str] = None
        self.trace_path: Optional[Path] = None

    def begin_run(self, run_id: str, metadata: Dict[str, Any]) -> None:
        self.run_id = run_id
        run_dir = self.base_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = run_dir / "trace.jsonl"
        self.record(
            node="run_start",
            step=0,
            status="ok",
            inputs={"metadata": metadata},
        )

    def record(
        self,
        node: str,
        step: int,
        status: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        if self.trace_path is None:
            return
        entry = {
            "ts": _utc_now(),
            "run_id": self.run_id,
            "step": step,
            "node": node,
            "status": status,
            "inputs": _summarize(inputs or {}),
            "outputs": _summarize(outputs or {}),
            "metadata": _summarize(metadata or {}),
            "error": error,
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def current_path(self) -> Optional[Path]:
        return self.trace_path
