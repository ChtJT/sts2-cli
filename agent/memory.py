#!/usr/bin/env python3
"""Layered memory storage for the STS2 agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, entry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


@dataclass
class MemorySnapshot:
    current_run: Dict[str, Any]
    facts: Dict[str, Any]
    recent_events: List[Dict[str, Any]]
    recent_reflections: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_run": self.current_run,
            "facts": self.facts,
            "recent_events": self.recent_events,
            "recent_reflections": self.recent_reflections,
        }


class LayeredMemory:
    """Three-layer memory: working, episodic, and reflective."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.working_path = self.base_dir / "working_memory.json"
        self.episodes_path = self.base_dir / "episodes.jsonl"
        self.reflections_path = self.base_dir / "reflections.jsonl"
        self.run_id: Optional[str] = None
        self.run_steps_path: Optional[Path] = None
        self.working = self._load_working()

    def _load_working(self) -> Dict[str, Any]:
        if self.working_path.is_file():
            return json.loads(self.working_path.read_text(encoding="utf-8"))
        return {
            "facts": {},
            "current_run": {},
            "recent_events": [],
            "recent_reflections": [],
            "last_updated": _utc_now(),
        }

    def begin_run(self, run_id: str, metadata: Dict[str, Any]) -> None:
        self.run_id = run_id
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_steps_path = run_dir / "steps.jsonl"
        self.working["current_run"] = {
            "run_id": run_id,
            "started_at": _utc_now(),
            **metadata,
        }
        self.working["last_updated"] = _utc_now()
        _safe_write_json(self.working_path, self.working)

    def snapshot(self, limit: int = 6) -> MemorySnapshot:
        return MemorySnapshot(
            current_run=self.working.get("current_run", {}),
            facts=self.working.get("facts", {}),
            recent_events=self.working.get("recent_events", [])[-limit:],
            recent_reflections=self.working.get("recent_reflections", [])[-3:],
        )

    def remember_fact(self, key: str, value: Any) -> None:
        self.working.setdefault("facts", {})[key] = value
        self.working["last_updated"] = _utc_now()
        _safe_write_json(self.working_path, self.working)

    def observe_state(self, state: Dict[str, Any]) -> None:
        player = state.get("player", {}) if isinstance(state, dict) else {}
        context = state.get("context", {}) if isinstance(state, dict) else {}
        facts = self.working.setdefault("facts", {})
        facts.update(
            {
                "decision": state.get("decision", state.get("type")),
                "act": context.get("act") or state.get("act"),
                "floor": context.get("floor") or state.get("floor"),
                "hp": player.get("hp"),
                "max_hp": player.get("max_hp"),
                "gold": player.get("gold"),
                "deck_size": player.get("deck_size"),
            }
        )
        self.working["last_updated"] = _utc_now()
        _safe_write_json(self.working_path, self.working)

    def record_step(
        self,
        step: int,
        state: Dict[str, Any],
        command: Dict[str, Any],
        response: Optional[Dict[str, Any]],
        provider_name: str,
        rationale: str,
        memory_note: str,
        decision_steps: Iterable[str],
        retrieval_hits: Iterable[Dict[str, Any]],
    ) -> None:
        summary = {
            "step": step,
            "decision": state.get("decision", state.get("type")),
            "action": command.get("action", command.get("cmd")),
            "response": None if response is None else response.get("decision", response.get("type")),
            "floor": state.get("context", {}).get("floor", state.get("floor")),
            "hp": state.get("player", {}).get("hp"),
            "gold": state.get("player", {}).get("gold"),
            "provider": provider_name,
            "rationale": rationale,
            "memory_note": memory_note,
            "decision_steps": list(decision_steps),
        }
        self.working.setdefault("recent_events", []).append(summary)
        self.working["recent_events"] = self.working["recent_events"][-20:]
        self.working["last_updated"] = _utc_now()
        _safe_write_json(self.working_path, self.working)

        event = {
            "ts": _utc_now(),
            "run_id": self.run_id,
            "step": step,
            "state": state,
            "command": command,
            "response": response,
            "provider": provider_name,
            "rationale": rationale,
            "memory_note": memory_note,
            "decision_steps": list(decision_steps),
            "retrieval": list(retrieval_hits),
        }
        _append_jsonl(self.episodes_path, event)
        if self.run_steps_path:
            _append_jsonl(self.run_steps_path, event)

    def reflect(self, kind: str, summary: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "ts": _utc_now(),
            "run_id": self.run_id,
            "kind": kind,
            "summary": summary,
            "metadata": metadata or {},
        }
        self.working.setdefault("recent_reflections", []).append(entry)
        self.working["recent_reflections"] = self.working["recent_reflections"][-10:]
        self.working["last_updated"] = _utc_now()
        _safe_write_json(self.working_path, self.working)
        _append_jsonl(self.reflections_path, entry)
