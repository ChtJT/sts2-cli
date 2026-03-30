#!/usr/bin/env python3
"""Schema objects for STS2 offline RL logging and training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RewardBreakdown:
    terminal: float = 0.0
    room_progress: float = 0.0
    combat_delta: float = 0.0
    hp_delta: float = 0.0
    deck_delta: float = 0.0
    economy_delta: float = 0.0
    action_penalty: float = 0.0
    safety_penalty: float = 0.0
    shaping: float = 0.0
    engine_penalty: float = 0.0

    @property
    def total(self) -> float:
        return round(
            self.terminal
            + self.room_progress
            + self.combat_delta
            + self.hp_delta
            + self.deck_delta
            + self.economy_delta
            + self.action_penalty
            + self.safety_penalty
            + self.shaping
            + self.engine_penalty,
            6,
        )

    def to_dict(self) -> Dict[str, float]:
        data = asdict(self)
        data["total"] = self.total
        return data


@dataclass
class RLTransition:
    ts: str
    run_id: str
    step: int
    provider: str
    character: str
    decision: str
    action: str
    action_key: str
    command: Dict[str, Any]
    available_actions: List[str]
    action_hints: Dict[str, Any]
    chosen_action_features: Dict[str, Any]
    state: Dict[str, Any]
    next_state: Optional[Dict[str, Any]]
    state_features: Dict[str, Any]
    next_state_features: Dict[str, Any]
    reward: float
    reward_breakdown: Dict[str, float]
    done: bool
    terminal_type: str
    rationale: str = ""
    memory_note: str = ""
    decision_steps: List[str] = field(default_factory=list)
    agent_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLTransition":
        return cls(**data)
