#!/usr/bin/env python3
"""Safety policies and guardrails for STS2 agent decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _extract_name(obj: Any) -> str:
    if isinstance(obj, dict):
        if obj.get("en"):
            return str(obj["en"])
        if obj.get("zh"):
            return str(obj["zh"])
    return str(obj or "")


def _hp_ratio(facts: Dict[str, Any]) -> float:
    value = facts.get("hp_ratio")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _incoming_damage(state: Dict[str, Any]) -> int:
    total = 0
    for enemy in state.get("enemies", []):
        for intent in enemy.get("intents", []) or []:
            damage = intent.get("damage")
            hits = intent.get("hits") or 1
            try:
                damage_value = int(damage or 0)
                hits_value = int(hits or 1)
            except (TypeError, ValueError):
                continue
            total += max(damage_value, 0) * max(hits_value, 1)
    return total


@dataclass
class SafetyContext:
    hard_rules: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hard_rules": self.hard_rules,
            "warnings": self.warnings,
            "suggested_actions": self.suggested_actions,
            "risk_flags": self.risk_flags,
        }


@dataclass
class SafetyDecision:
    allowed: bool
    reason: str = ""
    warnings: List[str] = field(default_factory=list)


class AgentSafetyPolicy:
    """Decision-time warnings and narrow hard blockers."""

    def __init__(self, character: str) -> None:
        self.character = character

    def build_context(
        self,
        state: Dict[str, Any],
        memory: Dict[str, Any],
        world_model: Optional[Dict[str, Any]] = None,
    ) -> SafetyContext:
        decision = state.get("decision", state.get("type", "unknown"))
        facts = memory.get("facts", {})
        deck_profile = memory.get("deck_profile", {})
        context = SafetyContext(
            hard_rules=[
                "Choose only legal actions and indices already present in the current state.",
                "Prefer robust lines over greedy lines when HP is unstable.",
            ]
        )

        hp_ratio = _hp_ratio(facts)
        if hp_ratio <= 0.35:
            context.risk_flags.append("critical_hp")
            context.warnings.append("HP is critical; avoid greedy or HP-loss lines unless forced.")

        if decision == "combat_play":
            incoming = _incoming_damage(state)
            playable_cards = [card for card in state.get("hand", []) if card.get("can_play")]
            defensive_cards = [
                _extract_name(card.get("name"))
                for card in playable_cards
                if str(card.get("type") or "") == "Skill"
            ]
            if incoming >= int(facts.get("hp") or 0) and incoming > 0:
                context.risk_flags.append("incoming_lethal_or_near_lethal")
                context.warnings.append(
                    f"Incoming damage is approximately {incoming}; prioritize surviving this turn."
                )
            if defensive_cards:
                context.suggested_actions.append(
                    f"Defensive resources available now: {', '.join(defensive_cards[:3])}."
                )

        if decision == "rest_site":
            recommendation = (memory.get("decision_context") or {}).get("recommended_option_id")
            if recommendation:
                context.suggested_actions.append(f"Rest-site baseline recommendation: {recommendation}.")
            if hp_ratio <= 0.45:
                context.hard_rules.append(
                    "If HEAL is available at critically low HP, do not choose a greedy smith line."
                )

        if decision == "shop":
            decision_context = memory.get("decision_context") or {}
            if decision_context.get("removal_affordable") and int(deck_profile.get("starter_cards", 0)) >= 5:
                context.warnings.append(
                    "Starter cards are still heavy and removal is affordable; avoid leaving for a marginal buy."
                )
            if int(facts.get("potion_slots_open") or 0) <= 0:
                context.hard_rules.append("Do not buy a potion when potion slots are already full.")
            priorities = decision_context.get("priorities") or []
            context.suggested_actions.extend(str(item) for item in priorities[:2])

        if decision == "map_select" and world_model:
            recommended = world_model.get("recommended_choice")
            if recommended:
                col = recommended.get("col")
                row = recommended.get("row")
                room_type = recommended.get("type")
                context.suggested_actions.append(
                    f"World-model preferred route is {room_type} at ({col},{row})."
                )

        return context

    def validate(
        self,
        state: Dict[str, Any],
        command: Dict[str, Any],
        memory: Dict[str, Any],
    ) -> SafetyDecision:
        decision = state.get("decision", state.get("type", "unknown"))
        action = command.get("action")
        args = command.get("args", {})
        facts = memory.get("facts", {})

        if decision == "rest_site" and action == "choose_option":
            options = {option.get("index"): option for option in state.get("options", [])}
            chosen = options.get(args.get("option_index"))
            option_id = str((chosen or {}).get("option_id") or "").upper()
            heal_available = any(
                str(option.get("option_id") or "").upper() == "HEAL" and option.get("is_enabled")
                for option in state.get("options", [])
            )
            if heal_available and _hp_ratio(facts) <= 0.45 and option_id and option_id != "HEAL":
                return SafetyDecision(
                    allowed=False,
                    reason="critical_hp_requires_heal",
                    warnings=["HP is too low to spend the rest site on smithing."],
                )

        if decision == "shop" and action == "buy_potion":
            potion_slots_open = int(facts.get("potion_slots_open") or 0)
            if potion_slots_open <= 0:
                return SafetyDecision(
                    allowed=False,
                    reason="potion_slots_full",
                    warnings=["Potion slots are full; buying a potion would waste gold."],
                )

        if decision == "map_select" and action == "select_map_node":
            hp_ratio = _hp_ratio(facts)
            choices = state.get("choices", [])
            current_choice = None
            for choice in choices:
                if choice.get("col") == args.get("col") and choice.get("row") == args.get("row"):
                    current_choice = choice
                    break
            if (
                current_choice
                and current_choice.get("type") == "Elite"
                and hp_ratio <= 0.4
                and any(choice.get("type") != "Elite" for choice in choices)
            ):
                return SafetyDecision(
                    allowed=False,
                    reason="low_hp_avoid_optional_elite",
                    warnings=["HP is too low to route into an optional elite right now."],
                )

        return SafetyDecision(allowed=True)
