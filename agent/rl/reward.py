#!/usr/bin/env python3
"""Continuous reward shaping for hierarchical STS2 RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from agent.memory import _analyze_deck
from agent.rl.schema import RewardBreakdown


def _extract_name(obj: Any) -> str:
    if isinstance(obj, dict):
        if obj.get("en"):
            return str(obj["en"])
        if obj.get("zh"):
            return str(obj["zh"])
    return str(obj or "")


def _hp_ratio(player: Dict[str, Any]) -> float:
    try:
        hp = float(player.get("hp") or 0.0)
        max_hp = float(player.get("max_hp") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if max_hp <= 0:
        return 0.0
    return max(0.0, min(1.5, hp / max_hp))


def _enemy_total_hp(state: Optional[Dict[str, Any]]) -> int:
    if not state:
        return 0
    total = 0
    for enemy in state.get("enemies", []) or []:
        try:
            total += max(0, int(enemy.get("hp") or 0))
        except (TypeError, ValueError):
            continue
    return total


def _enemy_count(state: Optional[Dict[str, Any]]) -> int:
    if not state:
        return 0
    return len(state.get("enemies", []) or [])


def _incoming_damage(state: Optional[Dict[str, Any]]) -> int:
    if not state:
        return 0
    total = 0
    for enemy in state.get("enemies", []) or []:
        for intent in enemy.get("intents", []) or []:
            try:
                total += max(0, int(intent.get("damage") or 0)) * max(1, int(intent.get("hits") or 1))
            except (TypeError, ValueError):
                continue
    return total


def _potion_count(state: Optional[Dict[str, Any]]) -> int:
    if not state:
        return 0
    return len([pot for pot in ((state.get("player", {}) or {}).get("potions", []) or []) if pot])


def _deck_profile(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    player = (state or {}).get("player", {}) or {}
    deck = player.get("deck", []) or []
    if not isinstance(deck, list):
        deck = []
    return _analyze_deck(deck)


def _room_type(state: Optional[Dict[str, Any]]) -> str:
    if not state:
        return ""
    return str(((state.get("context") or {}).get("room_type") or ""))


def _potential(state: Optional[Dict[str, Any]]) -> float:
    if not state:
        return 0.0
    player = state.get("player", {}) or {}
    deck_profile = _deck_profile(state)
    hp_ratio = _hp_ratio(player)
    gold = int(player.get("gold") or 0)
    enemy_pressure = _incoming_damage(state)
    enemy_total_hp = _enemy_total_hp(state)
    deck_size = max(1, int(deck_profile.get("deck_size") or 1))

    upgraded_ratio = float(deck_profile.get("upgraded_cards") or 0) / deck_size
    starter_ratio = float(deck_profile.get("starter_cards") or 0) / deck_size
    draw_ratio = float(deck_profile.get("draw_cards") or 0) / deck_size
    block_ratio = float(deck_profile.get("block_cards") or 0) / deck_size
    strength_ratio = min(float(deck_profile.get("strength_sources") or 0), 3.0) / 3.0
    potion_ratio = _potion_count(state) / 3.0

    value = 0.0
    value += 2.5 * hp_ratio
    value += 0.7 * min(gold / 180.0, 1.3)
    value += 1.5 * upgraded_ratio
    value += 0.5 * draw_ratio
    value += 0.4 * block_ratio
    value += 0.6 * strength_ratio
    value += 0.25 * potion_ratio
    value -= 1.4 * starter_ratio
    value -= 0.9 * min(enemy_pressure / 45.0, 1.5)
    value -= 0.5 * min(enemy_total_hp / 150.0, 1.5)
    return value


@dataclass
class RewardResult:
    reward: float
    breakdown: RewardBreakdown
    done: bool
    terminal_type: str


class ContinuousRewardModel:
    """Dense reward model for offline RL data collection."""

    def __init__(self, gamma: float = 0.99) -> None:
        self.gamma = gamma

    def evaluate(
        self,
        state: Dict[str, Any],
        command: Dict[str, Any],
        next_state: Optional[Dict[str, Any]],
        agent_context: Optional[Dict[str, Any]] = None,
    ) -> RewardResult:
        del agent_context  # Reserved for future reward shaping extensions.

        breakdown = RewardBreakdown()
        player = state.get("player", {}) or {}
        next_player = (next_state or {}).get("player", {}) or {}
        decision = str(state.get("decision") or state.get("type") or "")
        action = str(command.get("action") or command.get("cmd") or "")
        response_type = str((next_state or {}).get("type") or "")
        next_decision = str((next_state or {}).get("decision") or response_type or "")

        current_hp = int(player.get("hp") or 0)
        next_hp = int(next_player.get("hp") or current_hp)
        hp_delta = next_hp - current_hp
        if hp_delta > 0:
            breakdown.hp_delta += min(hp_delta, 20) * 0.03
        elif hp_delta < 0:
            breakdown.hp_delta += max(hp_delta, -40) * 0.035

        current_enemy_hp = _enemy_total_hp(state)
        next_enemy_hp = _enemy_total_hp(next_state)
        enemy_hp_delta = max(0, current_enemy_hp - next_enemy_hp)
        if enemy_hp_delta > 0:
            breakdown.combat_delta += min(enemy_hp_delta, 80) * 0.02

        enemy_kills = max(0, _enemy_count(state) - _enemy_count(next_state))
        if enemy_kills > 0:
            breakdown.combat_delta += enemy_kills * 0.4

        if decision == "combat_play" and action == "end_turn":
            playable = [card for card in state.get("hand", []) or [] if card.get("can_play")]
            energy = int(state.get("energy") or 0)
            if playable and energy > 0:
                breakdown.action_penalty -= min(0.2, 0.05 * len(playable))

        current_gold = int(player.get("gold") or 0)
        next_gold = int(next_player.get("gold") or current_gold)
        gold_delta = next_gold - current_gold
        if gold_delta > 0:
            breakdown.economy_delta += min(gold_delta, 60) * 0.01

        current_deck = _deck_profile(state)
        next_deck = _deck_profile(next_state)
        starter_delta = int(current_deck.get("starter_cards") or 0) - int(next_deck.get("starter_cards") or 0)
        upgraded_delta = int(next_deck.get("upgraded_cards") or 0) - int(current_deck.get("upgraded_cards") or 0)
        if starter_delta > 0:
            breakdown.deck_delta += starter_delta * 0.45
        if upgraded_delta > 0:
            breakdown.deck_delta += upgraded_delta * 0.3

        relic_delta = len((next_player.get("relics") or [])) - len((player.get("relics") or []))
        potion_delta = _potion_count(next_state) - _potion_count(state)
        if relic_delta > 0:
            breakdown.economy_delta += relic_delta * 0.8
        if potion_delta > 0:
            breakdown.economy_delta += potion_delta * 0.12

        if next_decision == "card_reward":
            room_type = _room_type(state)
            if room_type == "Elite":
                breakdown.room_progress += 1.5
            elif room_type == "Boss":
                breakdown.room_progress += 3.0
            else:
                breakdown.room_progress += 0.5

        if decision == "shop":
            if action == "remove_card":
                breakdown.room_progress += 0.45
            elif action in {"buy_card", "buy_relic", "buy_potion"}:
                breakdown.room_progress += 0.2
            elif action == "leave_room":
                breakdown.room_progress -= 0.05

        if decision == "rest_site" and action == "choose_option":
            option_index = (command.get("args") or {}).get("option_index")
            option = next((item for item in state.get("options", []) or [] if item.get("index") == option_index), {})
            option_id = str(option.get("option_id") or "")
            if option_id == "HEAL" and hp_delta > 0:
                breakdown.room_progress += 0.3
            elif option_id == "SMITH" and upgraded_delta > 0:
                breakdown.room_progress += 0.35

        terminal_type = "continuing"
        done = False
        if response_type == "error":
            done = True
            terminal_type = "engine_error"
            breakdown.engine_penalty -= 2.0
        elif next_decision == "game_over":
            done = True
            if next_state and next_state.get("victory"):
                terminal_type = "victory"
                breakdown.terminal += 10.0
            else:
                terminal_type = "defeat"
                breakdown.terminal -= 10.0

        phi_before = _potential(state)
        phi_after = _potential(next_state)
        breakdown.shaping += (self.gamma * phi_after) - phi_before

        return RewardResult(
            reward=breakdown.total,
            breakdown=breakdown,
            done=done,
            terminal_type=terminal_type,
        )
