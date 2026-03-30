#!/usr/bin/env python3
"""Dataset helpers for offline BC/IQL training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from agent.memory import _analyze_deck
from agent.rl.schema import RLTransition

KNOWN_DECISIONS = [
    "map_select",
    "combat_play",
    "card_reward",
    "bundle_select",
    "card_select",
    "rest_site",
    "event_choice",
    "shop",
    "game_over",
]
KNOWN_ACTIONS = [
    "select_map_node",
    "play_card",
    "end_turn",
    "use_potion",
    "select_card_reward",
    "skip_card_reward",
    "select_bundle",
    "select_cards",
    "skip_select",
    "choose_option",
    "leave_room",
    "buy_card",
    "buy_relic",
    "buy_potion",
    "remove_card",
    "proceed",
]
NODE_TYPES = ["Monster", "Elite", "RestSite", "Shop", "Treasure", "Event", "Unknown", "Boss", "Ancient"]
CARD_TYPES = ["Attack", "Skill", "Power", "Status", "Curse"]
TARGET_TYPES = ["None", "AnyEnemy", "Self", "Unknown"]
STRATEGY_MODES = ["survival", "stabilize", "balanced", "elite_hunt", "shop_window", "upgrade_window"]
SKILL_BUCKETS = ["combat", "route", "rest", "shop", "reward", "event", "bundle", "card_select", "generic"]


def _extract_name(obj: Any) -> str:
    if isinstance(obj, dict):
        if obj.get("en"):
            return str(obj["en"])
        if obj.get("zh"):
            return str(obj["zh"])
    return str(obj or "")


def _one_hot(value: str, vocab: Sequence[str]) -> List[float]:
    return [1.0 if value == item else 0.0 for item in vocab]


def _clip_number(value: Any, scale: float = 1.0, maximum: float = 1.0) -> float:
    try:
        numeric = float(value) / scale
    except (TypeError, ValueError):
        return 0.0
    return max(-maximum, min(maximum, numeric))


def _stat(card: Dict[str, Any], key: str) -> float:
    stats = card.get("stats") or {}
    value = stats.get(key)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _hp_ratio(player: Dict[str, Any]) -> float:
    try:
        hp = float(player.get("hp") or 0.0)
        max_hp = float(player.get("max_hp") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if max_hp <= 0:
        return 0.0
    return max(0.0, min(1.5, hp / max_hp))


def _incoming_damage(state: Dict[str, Any]) -> int:
    total = 0
    for enemy in state.get("enemies", []) or []:
        for intent in enemy.get("intents", []) or []:
            try:
                total += max(0, int(intent.get("damage") or 0)) * max(1, int(intent.get("hits") or 1))
            except (TypeError, ValueError):
                continue
    return total


def _enemy_total_hp(state: Dict[str, Any]) -> int:
    total = 0
    for enemy in state.get("enemies", []) or []:
        try:
            total += max(0, int(enemy.get("hp") or 0))
        except (TypeError, ValueError):
            continue
    return total


def _primary_skill_bucket(agent_context: Dict[str, Any]) -> str:
    skill_name = str((((agent_context or {}).get("skills") or {}).get("primary_skill") or {}).get("name") or "")
    if not skill_name:
        return "generic"
    prefix = skill_name.split("_", 1)[0]
    return prefix if prefix in SKILL_BUCKETS else "generic"


def _strategy_mode(agent_context: Dict[str, Any]) -> str:
    return str((((agent_context or {}).get("world_model") or {}).get("strategy_mode") or "balanced"))


def command_key(command: Dict[str, Any]) -> str:
    action = str(command.get("action") or command.get("cmd") or "")
    args = command.get("args") or {}
    if action == "select_map_node":
        return f"{action}:{args.get('col')}:{args.get('row')}"
    if action in {"play_card", "use_potion"}:
        return f"{action}:{args.get('card_index', args.get('potion_index'))}:{args.get('target_index')}"
    if action in {"select_card_reward", "select_bundle", "buy_card", "buy_relic", "buy_potion"}:
        for key in ("card_index", "bundle_index", "relic_index", "potion_index"):
            if key in args:
                return f"{action}:{args.get(key)}"
    if action == "select_cards":
        return f"{action}:{args.get('indices', '')}"
    if action == "choose_option":
        return f"{action}:{args.get('option_index')}"
    return action


def available_actions_from_state(state: Dict[str, Any]) -> List[str]:
    decision = str(state.get("decision") or state.get("type") or "")
    if decision == "map_select":
        return ["select_map_node"]
    if decision == "combat_play":
        actions = ["play_card", "end_turn"]
        if any(pot for pot in (state.get("player", {}) or {}).get("potions", []) or []):
            actions.append("use_potion")
        return actions
    if decision == "card_reward":
        return ["select_card_reward", "skip_card_reward"]
    if decision == "bundle_select":
        return ["select_bundle"]
    if decision == "card_select":
        actions = ["select_cards"]
        if int(state.get("min_select", 1)) == 0:
            actions.append("skip_select")
        return actions
    if decision == "rest_site":
        return ["choose_option"]
    if decision == "event_choice":
        actions = ["choose_option"]
        if state.get("options"):
            actions.append("leave_room")
        return actions
    if decision == "shop":
        return ["buy_card", "buy_relic", "buy_potion", "remove_card", "leave_room"]
    return ["proceed"]


def action_hints_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    decision = str(state.get("decision") or state.get("type") or "")
    if decision == "map_select":
        return {"choices": state.get("choices", [])}
    if decision == "combat_play":
        return {
            "playable_cards": [
                {
                    "index": card.get("index"),
                    "name": _extract_name(card.get("name")),
                    "target_type": card.get("target_type"),
                }
                for card in state.get("hand", []) or []
                if card.get("can_play")
            ],
            "potions": [
                {
                    "index": potion.get("index"),
                    "name": _extract_name(potion.get("name")),
                    "target_type": potion.get("target_type"),
                }
                for potion in (state.get("player", {}) or {}).get("potions", []) or []
                if potion
            ],
        }
    if decision == "card_reward":
        return {"cards": state.get("cards", [])}
    if decision == "bundle_select":
        return {"bundles": state.get("bundles", [])}
    if decision == "card_select":
        return {
            "cards": state.get("cards", []),
            "min_select": state.get("min_select"),
            "max_select": state.get("max_select"),
        }
    if decision in {"rest_site", "event_choice"}:
        return {"options": state.get("options", [])}
    if decision == "shop":
        return {
            "cards": state.get("cards", []),
            "relics": state.get("relics", []),
            "potions": state.get("potions", []),
            "card_removal_cost": state.get("card_removal_cost"),
        }
    return {}


def summarize_state_for_rl(state: Dict[str, Any], agent_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    player = state.get("player", {}) or {}
    context = state.get("context", {}) or {}
    deck = player.get("deck", []) or []
    deck_profile = _analyze_deck(deck if isinstance(deck, list) else [])
    playable_cards = [card for card in state.get("hand", []) or [] if card.get("can_play")]
    playable_attacks = [card for card in playable_cards if card.get("type") == "Attack"]
    playable_skills = [card for card in playable_cards if card.get("type") == "Skill"]
    agent_context = agent_context or {}

    return {
        "decision": str(state.get("decision") or state.get("type") or ""),
        "act": int(context.get("act") or state.get("act") or 0),
        "floor": int(context.get("floor") or state.get("floor") or 0),
        "room_type": str(context.get("room_type") or ""),
        "hp_ratio": round(_hp_ratio(player), 6),
        "hp": int(player.get("hp") or 0),
        "max_hp": int(player.get("max_hp") or 0),
        "gold": int(player.get("gold") or 0),
        "energy": int(state.get("energy") or 0),
        "deck_size": int(deck_profile.get("deck_size") or 0),
        "starter_cards": int(deck_profile.get("starter_cards") or 0),
        "upgraded_cards": int(deck_profile.get("upgraded_cards") or 0),
        "attacks": int(deck_profile.get("attacks") or 0),
        "skills": int(deck_profile.get("skills") or 0),
        "powers": int(deck_profile.get("powers") or 0),
        "block_cards": int(deck_profile.get("block_cards") or 0),
        "draw_cards": int(deck_profile.get("draw_cards") or 0),
        "strength_sources": int(deck_profile.get("strength_sources") or 0),
        "enemy_count": len(state.get("enemies", []) or []),
        "enemy_total_hp": _enemy_total_hp(state),
        "incoming_damage": _incoming_damage(state),
        "playable_count": len(playable_cards),
        "playable_attack_count": len(playable_attacks),
        "playable_skill_count": len(playable_skills),
        "potion_count": len([pot for pot in player.get("potions", []) or [] if pot]),
        "choice_count": len(state.get("choices", []) or []),
        "strategy_mode": _strategy_mode(agent_context),
        "primary_skill_bucket": _primary_skill_bucket(agent_context),
        "risk_flags": list(((agent_context.get("safety") or {}).get("risk_flags") or [])[:4]),
    }


def summarize_action_for_rl(state: Dict[str, Any], command: Dict[str, Any]) -> Dict[str, Any]:
    action = str(command.get("action") or command.get("cmd") or "")
    args = command.get("args") or {}
    summary: Dict[str, Any] = {
        "action": action,
        "action_key": command_key(command),
        "decision": str(state.get("decision") or state.get("type") or ""),
    }

    if action == "select_map_node":
        choice = next(
            (
                item
                for item in state.get("choices", []) or []
                if item.get("col") == args.get("col") and item.get("row") == args.get("row")
            ),
            {},
        )
        summary.update(
            {
                "col": args.get("col"),
                "row": args.get("row"),
                "node_type": choice.get("type"),
            }
        )
        return summary

    if action == "play_card":
        card = next((item for item in state.get("hand", []) or [] if item.get("index") == args.get("card_index")), {})
        target = next((enemy for enemy in state.get("enemies", []) or [] if enemy.get("index") == args.get("target_index")), {})
        summary.update(
            {
                "card_index": args.get("card_index"),
                "target_index": args.get("target_index"),
                "card_name": _extract_name(card.get("name")),
                "card_type": card.get("type"),
                "card_cost": card.get("cost"),
                "damage": _stat(card, "damage"),
                "block": _stat(card, "block"),
                "draw": _stat(card, "cards"),
                "vulnerable": _stat(card, "vulnerablepower"),
                "target_hp": target.get("hp"),
                "target_type": card.get("target_type") or "None",
            }
        )
        return summary

    if action == "use_potion":
        potion = next(
            (item for item in (state.get("player", {}) or {}).get("potions", []) or [] if item and item.get("index") == args.get("potion_index")),
            {},
        )
        target = next((enemy for enemy in state.get("enemies", []) or [] if enemy.get("index") == args.get("target_index")), {})
        summary.update(
            {
                "potion_index": args.get("potion_index"),
                "target_index": args.get("target_index"),
                "potion_name": _extract_name(potion.get("name")),
                "target_hp": target.get("hp"),
                "target_type": potion.get("target_type") or "None",
            }
        )
        return summary

    if action == "select_card_reward":
        card = next((item for item in state.get("cards", []) or [] if item.get("index") == args.get("card_index")), {})
        summary.update(
            {
                "card_index": args.get("card_index"),
                "card_name": _extract_name(card.get("name")),
                "card_type": card.get("type"),
                "card_cost": card.get("cost"),
                "damage": _stat(card, "damage"),
                "block": _stat(card, "block"),
                "draw": _stat(card, "cards"),
                "vulnerable": _stat(card, "vulnerablepower"),
            }
        )
        return summary

    if action == "select_bundle":
        bundle = next((item for item in state.get("bundles", []) or [] if item.get("index") == args.get("bundle_index")), {})
        summary.update({"bundle_index": args.get("bundle_index"), "bundle_name": _extract_name(bundle.get("name"))})
        return summary

    if action == "select_cards":
        summary["indices"] = str(args.get("indices") or "")
        summary["selected_count"] = len([item for item in str(args.get("indices") or "").split(",") if item])
        return summary

    if action == "choose_option":
        option = next((item for item in state.get("options", []) or [] if item.get("index") == args.get("option_index")), {})
        summary.update(
            {
                "option_index": args.get("option_index"),
                "option_id": option.get("option_id"),
                "enabled": option.get("is_enabled", not option.get("is_locked", False)),
            }
        )
        return summary

    if action in {"buy_card", "buy_relic", "buy_potion"}:
        field = {
            "buy_card": ("cards", "card_index"),
            "buy_relic": ("relics", "relic_index"),
            "buy_potion": ("potions", "potion_index"),
        }[action]
        entries = state.get(field[0], []) or []
        entry = next((item for item in entries if item.get("index") == args.get(field[1])), {})
        summary.update(
            {
                field[1]: args.get(field[1]),
                "item_name": _extract_name(entry.get("name")),
                "item_type": entry.get("type"),
                "cost": entry.get("cost"),
                "on_sale": bool(entry.get("on_sale")),
                "is_stocked": bool(entry.get("is_stocked")),
            }
        )
        return summary

    if action == "remove_card":
        summary["cost"] = state.get("card_removal_cost")
        summary["affordable"] = int((state.get("player", {}) or {}).get("gold") or 0) >= int(state.get("card_removal_cost") or 0)
        return summary

    return summary


@dataclass
class ActionCandidate:
    command: Dict[str, Any]
    key: str
    summary: Dict[str, Any]


def enumerate_action_candidates(state: Dict[str, Any]) -> List[ActionCandidate]:
    decision = str(state.get("decision") or state.get("type") or "")
    candidates: List[ActionCandidate] = []

    def add(command: Dict[str, Any]) -> None:
        candidates.append(ActionCandidate(command=command, key=command_key(command), summary=summarize_action_for_rl(state, command)))

    if decision == "map_select":
        for choice in state.get("choices", []) or []:
            add({"cmd": "action", "action": "select_map_node", "args": {"col": choice.get("col"), "row": choice.get("row")}})
        return candidates

    if decision == "combat_play":
        for card in state.get("hand", []) or []:
            if not card.get("can_play"):
                continue
            if card.get("target_type") == "AnyEnemy":
                for enemy in state.get("enemies", []) or []:
                    add({"cmd": "action", "action": "play_card", "args": {"card_index": card.get("index"), "target_index": enemy.get("index")}})
            else:
                add({"cmd": "action", "action": "play_card", "args": {"card_index": card.get("index")}})
        for potion in (state.get("player", {}) or {}).get("potions", []) or []:
            if not potion:
                continue
            if potion.get("target_type") == "AnyEnemy":
                for enemy in state.get("enemies", []) or []:
                    add({"cmd": "action", "action": "use_potion", "args": {"potion_index": potion.get("index"), "target_index": enemy.get("index")}})
            else:
                add({"cmd": "action", "action": "use_potion", "args": {"potion_index": potion.get("index")}})
        add({"cmd": "action", "action": "end_turn"})
        return candidates

    if decision == "card_reward":
        for card in state.get("cards", []) or []:
            add({"cmd": "action", "action": "select_card_reward", "args": {"card_index": card.get("index")}})
        add({"cmd": "action", "action": "skip_card_reward"})
        return candidates

    if decision == "bundle_select":
        for bundle in state.get("bundles", []) or []:
            add({"cmd": "action", "action": "select_bundle", "args": {"bundle_index": bundle.get("index")}})
        return candidates

    if decision == "card_select":
        cards = state.get("cards", []) or []
        min_select = int(state.get("min_select", 1))
        max_select = int(state.get("max_select", len(cards)))
        if min_select == 1 and max_select == 1:
            for card in cards:
                add({"cmd": "action", "action": "select_cards", "args": {"indices": str(card.get("index"))}})
        if min_select == 0:
            add({"cmd": "action", "action": "skip_select"})
        return candidates

    if decision in {"rest_site", "event_choice"}:
        for option in state.get("options", []) or []:
            if decision == "rest_site" and not option.get("is_enabled"):
                continue
            if decision == "event_choice" and option.get("is_locked"):
                continue
            add({"cmd": "action", "action": "choose_option", "args": {"option_index": option.get("index")}})
        if decision == "event_choice" and state.get("options"):
            add({"cmd": "action", "action": "leave_room"})
        return candidates

    if decision == "shop":
        add({"cmd": "action", "action": "leave_room"})
        player_gold = int((state.get("player", {}) or {}).get("gold") or 0)
        removal_cost = int(state.get("card_removal_cost") or 0)
        if removal_cost <= 0 or player_gold >= removal_cost:
            add({"cmd": "action", "action": "remove_card"})
        for entry in state.get("cards", []) or []:
            if entry.get("is_stocked") and player_gold >= int(entry.get("cost") or 0):
                add({"cmd": "action", "action": "buy_card", "args": {"card_index": entry.get("index")}})
        for entry in state.get("relics", []) or []:
            if entry.get("is_stocked") and player_gold >= int(entry.get("cost") or 0):
                add({"cmd": "action", "action": "buy_relic", "args": {"relic_index": entry.get("index")}})
        for entry in state.get("potions", []) or []:
            if entry.get("is_stocked") and player_gold >= int(entry.get("cost") or 0):
                add({"cmd": "action", "action": "buy_potion", "args": {"potion_index": entry.get("index")}})
        return candidates

    add({"cmd": "action", "action": "proceed"})
    return candidates


def _vectorize_state(summary: Dict[str, Any]) -> List[float]:
    vector: List[float] = []
    vector.extend(_one_hot(str(summary.get("decision") or ""), KNOWN_DECISIONS))
    vector.extend(_one_hot(str(summary.get("room_type") or ""), NODE_TYPES))
    vector.extend(_one_hot(str(summary.get("strategy_mode") or "balanced"), STRATEGY_MODES))
    vector.extend(_one_hot(str(summary.get("primary_skill_bucket") or "generic"), SKILL_BUCKETS))
    vector.extend(
        [
            _clip_number(summary.get("act"), scale=5.0),
            _clip_number(summary.get("floor"), scale=60.0),
            _clip_number(summary.get("hp_ratio"), scale=1.0, maximum=1.5),
            _clip_number(summary.get("gold"), scale=250.0, maximum=2.0),
            _clip_number(summary.get("energy"), scale=6.0),
            _clip_number(summary.get("deck_size"), scale=40.0),
            _clip_number(summary.get("starter_cards"), scale=10.0),
            _clip_number(summary.get("upgraded_cards"), scale=12.0),
            _clip_number(summary.get("attacks"), scale=20.0),
            _clip_number(summary.get("skills"), scale=20.0),
            _clip_number(summary.get("powers"), scale=10.0),
            _clip_number(summary.get("block_cards"), scale=15.0),
            _clip_number(summary.get("draw_cards"), scale=10.0),
            _clip_number(summary.get("strength_sources"), scale=4.0),
            _clip_number(summary.get("enemy_count"), scale=5.0),
            _clip_number(summary.get("enemy_total_hp"), scale=200.0, maximum=2.0),
            _clip_number(summary.get("incoming_damage"), scale=60.0, maximum=2.0),
            _clip_number(summary.get("playable_count"), scale=10.0),
            _clip_number(summary.get("playable_attack_count"), scale=10.0),
            _clip_number(summary.get("playable_skill_count"), scale=10.0),
            _clip_number(summary.get("potion_count"), scale=3.0),
            _clip_number(summary.get("choice_count"), scale=10.0),
            _clip_number(len(summary.get("risk_flags", []) or []), scale=6.0),
        ]
    )
    return vector


def _vectorize_action(summary: Dict[str, Any]) -> List[float]:
    action = str(summary.get("action") or "")
    vector: List[float] = []
    vector.extend(_one_hot(action, KNOWN_ACTIONS))
    vector.extend(_one_hot(str(summary.get("node_type") or "Unknown"), NODE_TYPES))
    vector.extend(_one_hot(str(summary.get("card_type") or summary.get("item_type") or "Status"), CARD_TYPES))
    vector.extend(_one_hot(str(summary.get("target_type") or "None"), TARGET_TYPES))
    vector.extend(
        [
            _clip_number(summary.get("card_cost", summary.get("cost")), scale=4.0),
            _clip_number(summary.get("damage"), scale=30.0, maximum=2.0),
            _clip_number(summary.get("block"), scale=30.0, maximum=2.0),
            _clip_number(summary.get("draw"), scale=5.0),
            _clip_number(summary.get("vulnerable"), scale=5.0),
            _clip_number(summary.get("target_hp"), scale=120.0, maximum=2.0),
            1.0 if summary.get("on_sale") else 0.0,
            1.0 if summary.get("is_stocked") else 0.0,
            1.0 if summary.get("enabled", True) else 0.0,
            1.0 if summary.get("affordable", False) else 0.0,
            _clip_number(summary.get("selected_count"), scale=4.0),
        ]
    )
    return vector


@dataclass
class CandidateTrainingRow:
    group_id: str
    run_id: str
    step: int
    decision: str
    action_key: str
    chosen: int
    reward: float
    done: int
    state_vector: List[float]
    action_vector: List[float]
    next_state_vector: List[float]
    metadata: Dict[str, Any]

    @property
    def feature_vector(self) -> List[float]:
        return self.state_vector + self.action_vector


def load_transitions(path: str) -> List[RLTransition]:
    records: List[RLTransition] = []
    dataset_path = Path(path)
    if not dataset_path.is_file():
        return records
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(RLTransition.from_dict(payload))
    return records


def build_candidate_rows(
    transitions: Iterable[RLTransition],
    decisions: Optional[Iterable[str]] = None,
    include_combat: bool = False,
) -> List[CandidateTrainingRow]:
    decision_filter = set(decisions or [])
    rows: List[CandidateTrainingRow] = []
    for transition in transitions:
        if decision_filter and transition.decision not in decision_filter:
            continue
        if not include_combat and transition.decision == "combat_play":
            continue

        state_features = transition.state_features or summarize_state_for_rl(transition.state, transition.agent_context)
        next_features = transition.next_state_features or summarize_state_for_rl(transition.next_state or {}, {})
        state_vector = _vectorize_state(state_features)
        next_state_vector = _vectorize_state(next_features) if next_features else [0.0] * len(state_vector)
        candidates = enumerate_action_candidates(transition.state)
        if not candidates:
            candidates = [ActionCandidate(command=transition.command, key=transition.action_key, summary=transition.chosen_action_features)]

        chosen_key = transition.action_key or command_key(transition.command)
        group_id = f"{transition.run_id}:{transition.step}"
        for candidate in candidates:
            rows.append(
                CandidateTrainingRow(
                    group_id=group_id,
                    run_id=transition.run_id,
                    step=transition.step,
                    decision=transition.decision,
                    action_key=candidate.key,
                    chosen=1 if candidate.key == chosen_key else 0,
                    reward=float(transition.reward),
                    done=1 if transition.done else 0,
                    state_vector=state_vector,
                    action_vector=_vectorize_action(candidate.summary),
                    next_state_vector=next_state_vector,
                    metadata={
                        "terminal_type": transition.terminal_type,
                        "provider": transition.provider,
                    },
                )
            )
    return rows
