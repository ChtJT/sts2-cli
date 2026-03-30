#!/usr/bin/env python3
"""Episode retrieval over prior STS2 agent runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


def _extract_name(obj: Any) -> str:
    if isinstance(obj, dict):
        if obj.get("en"):
            return str(obj["en"])
        if obj.get("zh"):
            return str(obj["zh"])
    return str(obj or "")


def _normalize(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _name_set(items: Iterable[Any]) -> Set[str]:
    output: Set[str] = set()
    for item in items:
        name = _normalize(_extract_name(item))
        if name:
            output.add(name)
    return output


def _enemy_names(state: Dict[str, Any]) -> Set[str]:
    return _name_set(enemy.get("name") for enemy in state.get("enemies", []) or [])


def _card_names(state: Dict[str, Any]) -> Set[str]:
    names: Set[str] = set()
    for key in ("cards", "hand"):
        names.update(_name_set(card.get("name") for card in state.get(key, []) or []))
    return names


def _hp_ratio(state: Dict[str, Any]) -> float:
    player = state.get("player", {}) or {}
    hp = player.get("hp")
    max_hp = player.get("max_hp")
    try:
        hp_value = float(hp)
        max_hp_value = float(max_hp)
    except (TypeError, ValueError):
        return 1.0
    if max_hp_value <= 0:
        return 1.0
    return hp_value / max_hp_value


def _floor(state: Dict[str, Any]) -> int:
    context = state.get("context", {}) or {}
    value = context.get("floor", state.get("floor", 0))
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


@dataclass
class EpisodeHit:
    run_id: str
    step: int
    decision: str
    action: str
    score: float
    floor: int
    hp_ratio: float
    rationale: str
    memory_note: str
    response_type: str
    enemy_names: List[str]
    card_names: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step": self.step,
            "decision": self.decision,
            "action": self.action,
            "score": round(self.score, 4),
            "floor": self.floor,
            "hp_ratio": round(self.hp_ratio, 3),
            "rationale": self.rationale,
            "memory_note": self.memory_note,
            "response_type": self.response_type,
            "enemy_names": self.enemy_names,
            "card_names": self.card_names,
        }


@dataclass
class _EpisodeRecord:
    run_id: str
    step: int
    decision: str
    action: str
    floor: int
    hp_ratio: float
    rationale: str
    memory_note: str
    response_type: str
    enemy_names: Set[str]
    card_names: Set[str]
    room_type: str
    starter_cards_hint: int
    outcome_penalty: float


class EpisodicRetriever:
    """Retrieve similar prior agent steps from stored episodes."""

    def __init__(self, state_dir: str) -> None:
        self.state_dir = Path(state_dir)
        self.episodes_path = self.state_dir / "episodes.jsonl"
        self.records = self._load_records()

    def _load_records(self) -> List[_EpisodeRecord]:
        if not self.episodes_path.is_file():
            return []

        records: List[_EpisodeRecord] = []
        with self.episodes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                state = entry.get("state", {}) or {}
                response = entry.get("response", {}) or {}
                player = state.get("player", {}) or {}
                deck_size = player.get("deck_size") or len(player.get("deck", []) or [])
                starter_hint = max(0, min(int(deck_size or 0), 10))
                response_type = str(response.get("decision", response.get("type", "")))
                penalty = 0.0
                if response.get("type") == "error":
                    penalty = 2.0
                records.append(
                    _EpisodeRecord(
                        run_id=str(entry.get("run_id") or ""),
                        step=int(entry.get("step") or 0),
                        decision=str(state.get("decision", state.get("type", ""))),
                        action=str((entry.get("command") or {}).get("action", "")),
                        floor=_floor(state),
                        hp_ratio=_hp_ratio(state),
                        rationale=str(entry.get("rationale") or ""),
                        memory_note=str(entry.get("memory_note") or ""),
                        response_type=response_type,
                        enemy_names=_enemy_names(state),
                        card_names=_card_names(state),
                        room_type=str((state.get("context") or {}).get("room_type") or ""),
                        starter_cards_hint=starter_hint,
                        outcome_penalty=penalty,
                    )
                )
        return records

    def search(
        self,
        state: Dict[str, Any],
        memory: Dict[str, Any],
        limit: int = 3,
        exclude_run_id: Optional[str] = None,
    ) -> List[EpisodeHit]:
        decision = str(state.get("decision", state.get("type", "")))
        enemy_names = _enemy_names(state)
        card_names = _card_names(state)
        floor = _floor(state)
        hp_ratio = _hp_ratio(state)
        room_type = str((state.get("context") or {}).get("room_type") or "")
        starter_cards = int((memory.get("deck_profile") or {}).get("starter_cards", 0))

        results: List[EpisodeHit] = []
        for record in self.records:
            if exclude_run_id and record.run_id == exclude_run_id:
                continue
            if record.decision != decision:
                continue
            score = 5.0

            if room_type and record.room_type == room_type:
                score += 1.0

            floor_gap = abs(record.floor - floor)
            score += max(0.0, 2.0 - min(floor_gap, 8) * 0.25)

            hp_gap = abs(record.hp_ratio - hp_ratio)
            score += max(0.0, 2.0 - hp_gap * 4.0)

            if enemy_names and record.enemy_names:
                overlap = len(enemy_names & record.enemy_names)
                union = len(enemy_names | record.enemy_names)
                score += (overlap / max(union, 1)) * 4.0

            if card_names and record.card_names:
                overlap = len(card_names & record.card_names)
                union = len(card_names | record.card_names)
                score += (overlap / max(union, 1)) * 2.0

            starter_gap = abs(record.starter_cards_hint - starter_cards)
            score += max(0.0, 1.0 - min(starter_gap, 6) * 0.15)
            score -= record.outcome_penalty

            if score <= 4.0:
                continue

            results.append(
                EpisodeHit(
                    run_id=record.run_id,
                    step=record.step,
                    decision=record.decision,
                    action=record.action,
                    score=score,
                    floor=record.floor,
                    hp_ratio=record.hp_ratio,
                    rationale=record.rationale,
                    memory_note=record.memory_note,
                    response_type=record.response_type,
                    enemy_names=sorted(record.enemy_names),
                    card_names=sorted(record.card_names)[:8],
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]
