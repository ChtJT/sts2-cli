#!/usr/bin/env python3
"""Layered memory storage for the STS2 agent."""

from __future__ import annotations

import json
import math
import re
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


def _extract_name(obj: Any) -> str:
    if isinstance(obj, dict):
        if obj.get("en"):
            return str(obj["en"])
        if obj.get("zh"):
            return str(obj["zh"])
    return str(obj or "")


def _normalize_name(obj: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _extract_name(obj).lower())


def _keywords(card: Dict[str, Any]) -> List[str]:
    result: List[str] = []
    for item in card.get("keywords") or []:
        text = str(item).strip().lower()
        if text:
            result.append(text)
    return result


def _stat(card: Dict[str, Any], key: str) -> float:
    stats = card.get("stats") or {}
    value = stats.get(key)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_starter_card(card: Dict[str, Any]) -> bool:
    normalized = _normalize_name(card.get("name"))
    return normalized in {"strike", "defend"}


def _card_tags(card: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    damage = _stat(card, "damage")
    block = _stat(card, "block")
    draw = _stat(card, "cards")
    vulnerable = _stat(card, "vulnerablepower")
    energy = _stat(card, "energy")
    hp_loss = _stat(card, "hploss")
    if damage > 0:
        tags.append(f"damage {int(damage)}")
    if block > 0:
        tags.append(f"block {int(block)}")
    if draw > 0:
        tags.append(f"draw {int(draw)}")
    if vulnerable > 0:
        tags.append(f"vulnerable {int(vulnerable)}")
    if energy > 0:
        tags.append(f"energy {int(energy)}")
    if hp_loss > 0:
        tags.append(f"hp_loss {int(hp_loss)}")
    if "exhaust" in _keywords(card):
        tags.append("exhaust")
    return tags


def _score_upgrade_candidate(card: Dict[str, Any]) -> float:
    score = 0.0
    card_type = str(card.get("type") or "")
    if card_type == "Power":
        score += 18
    elif card_type == "Attack":
        score += 12
    elif card_type == "Skill":
        score += 10

    cost = card.get("cost")
    if cost == 0:
        score += 6
    elif cost == 1:
        score += 4
    elif cost == 2:
        score += 2

    score += min(_stat(card, "damage"), 20) * 0.45
    score += min(_stat(card, "block"), 20) * 0.35
    score += _stat(card, "cards") * 4
    score += _stat(card, "vulnerablepower") * 5
    score += _stat(card, "energy") * 6

    if "exhaust" in _keywords(card):
        score += 2
    if _is_starter_card(card):
        score -= 4
    return score


def _smith_candidates(deck: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for card in deck:
        if card.get("upgraded"):
            continue
        name = _extract_name(card.get("name"))
        if not name:
            continue
        score = _score_upgrade_candidate(card)
        reason_tags = _card_tags(card)
        scored.append(
            {
                "name": name,
                "type": card.get("type"),
                "score": round(score, 2),
                "reason": ", ".join(reason_tags[:3]) or str(card.get("type") or "general scaling"),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:limit]


def _analyze_deck(deck: List[Dict[str, Any]]) -> Dict[str, Any]:
    attacks = 0
    skills = 0
    powers = 0
    upgraded = 0
    zero_cost = 0
    exhaust_cards = 0
    block_cards = 0
    draw_cards = 0
    vulnerable_cards = 0
    hp_loss_cards = 0
    expensive_cards = 0
    starter_strikes = 0
    starter_defends = 0
    name_counts: Dict[str, int] = {}
    notable_names: List[str] = []
    normalized_names: List[str] = []

    for card in deck:
        name = _extract_name(card.get("name"))
        normalized = _normalize_name(card.get("name"))
        normalized_names.append(normalized)
        name_counts[normalized] = name_counts.get(normalized, 0) + 1

        if card.get("type") == "Attack":
            attacks += 1
        elif card.get("type") == "Skill":
            skills += 1
        elif card.get("type") == "Power":
            powers += 1

        if card.get("upgraded"):
            upgraded += 1
        if card.get("cost") == 0:
            zero_cost += 1
        if card.get("cost") and card.get("cost") >= 2:
            expensive_cards += 1
        if "exhaust" in _keywords(card):
            exhaust_cards += 1
        if _stat(card, "block") > 0:
            block_cards += 1
        if _stat(card, "cards") > 0:
            draw_cards += 1
        if _stat(card, "vulnerablepower") > 0:
            vulnerable_cards += 1
        if _stat(card, "hploss") > 0:
            hp_loss_cards += 1
        if normalized == "strike":
            starter_strikes += 1
        elif normalized == "defend":
            starter_defends += 1
        elif name and name not in notable_names:
            notable_names.append(name)

    removal_candidates: List[str] = []
    removal_candidates.extend(["Strike"] * starter_strikes)
    removal_candidates.extend(["Defend"] * starter_defends)

    strength_sources = sum(
        1
        for normalized in normalized_names
        if normalized in {"inflame", "spotweakness", "demonform", "limitbreak"}
    )

    return {
        "deck_size": len(deck),
        "attacks": attacks,
        "skills": skills,
        "powers": powers,
        "upgraded_cards": upgraded,
        "zero_cost_cards": zero_cost,
        "exhaust_cards": exhaust_cards,
        "block_cards": block_cards,
        "draw_cards": draw_cards,
        "vulnerable_cards": vulnerable_cards,
        "hp_loss_cards": hp_loss_cards,
        "expensive_cards": expensive_cards,
        "starter_strikes": starter_strikes,
        "starter_defends": starter_defends,
        "starter_cards": starter_strikes + starter_defends,
        "strength_sources": strength_sources,
        "notable_cards": notable_names[:8],
        "removal_candidates": removal_candidates[:6],
        "smith_candidates": _smith_candidates(deck),
        "name_counts": name_counts,
    }


def _hp_ratio(player: Dict[str, Any]) -> float:
    hp = player.get("hp")
    max_hp = player.get("max_hp")
    if not isinstance(hp, (int, float)) or not isinstance(max_hp, (int, float)) or max_hp <= 0:
        return 1.0
    return float(hp) / float(max_hp)


def _specific_card_bonus(name: str, deck_profile: Dict[str, Any]) -> float:
    strength_sources = int(deck_profile.get("strength_sources", 0))
    exhaust_cards = int(deck_profile.get("exhaust_cards", 0))
    expensive_cards = int(deck_profile.get("expensive_cards", 0))
    attack_count = int(deck_profile.get("attacks", 0))
    hp_loss_cards = int(deck_profile.get("hp_loss_cards", 0))

    bonuses = {
        "rage": 10 + min(attack_count, 8) * 0.8,
        "pommelstrike": 14,
        "battletrance": 16,
        "shrugitoff": 15,
        "offering": 18,
        "armaments": 12,
        "bludgeon": 9,
        "flamebarrier": 12,
        "reaper": 4 + strength_sources * 3,
        "fiendfire": 8 + exhaust_cards * 1.5,
        "bloodletting": 4 + expensive_cards * 1.5,
        "swordboomerang": 2 + strength_sources * 4,
        "limitbreak": -8 + strength_sources * 8,
        "rupture": -6 + hp_loss_cards * 6,
        "impervious": 17,
        "burningpact": 12,
        "spotweakness": 12,
    }
    return bonuses.get(name, 0.0)


def _specific_relic_bonus(name: str, deck_profile: Dict[str, Any], potion_slots_open: int) -> float:
    skills = int(deck_profile.get("skills", 0))
    bonuses = {
        "gremlinhorn": 18,
        "bagofpreparation": 16,
        "preservedinsect": 16,
        "vajra": 14,
        "anchor": 12,
        "lantern": 12,
        "happyflower": 12,
        "oddlysmoothstone": 12,
        "potionbelt": 6 if potion_slots_open > 0 else 12,
        "cauldron": 10 if potion_slots_open > 0 else -6,
        "letteropener": 3 + min(skills, 8) * 1.1,
    }
    return bonuses.get(name, 0.0)


def _specific_potion_bonus(name: str, hp_ratio: float, smith_candidates: List[Dict[str, Any]]) -> float:
    bonuses = {
        "regenpotion": 12 if hp_ratio < 0.6 else 4,
        "liquidbronze": 10 if hp_ratio < 0.6 else 5,
        "flexpotion": 7,
        "firepotion": 8,
        "explosivepotion": 8,
        "blessingoftheforge": 9 if smith_candidates else 4,
    }
    return bonuses.get(name, 0.0)


def _score_shop_card(entry: Dict[str, Any], deck_profile: Dict[str, Any], gold: int) -> float:
    if not entry.get("is_stocked"):
        return -999.0
    cost = int(entry.get("cost") or 0)
    if cost > gold:
        return -999.0

    score = 0.0
    if entry.get("on_sale"):
        score += 18
    if cost <= 50:
        score += 7
    elif cost <= 80:
        score += 4
    elif cost >= 150:
        score -= 6

    card_type = str(entry.get("type") or "")
    if card_type == "Attack" and int(deck_profile.get("attacks", 0)) <= int(deck_profile.get("skills", 0)):
        score += 4
    if card_type == "Skill" and int(deck_profile.get("block_cards", 0)) < 5:
        score += 5
    if card_type == "Power":
        score += 4

    score += _specific_card_bonus(_normalize_name(entry.get("name")), deck_profile)
    return score


def _score_shop_relic(entry: Dict[str, Any], deck_profile: Dict[str, Any], gold: int, potion_slots_open: int) -> float:
    if not entry.get("is_stocked"):
        return -999.0
    cost = int(entry.get("cost") or 0)
    if cost > gold:
        return -999.0

    score = 8.0
    if cost >= math.floor(gold * 0.9):
        score -= 2
    score += _specific_relic_bonus(_normalize_name(entry.get("name")), deck_profile, potion_slots_open)
    return score


def _score_shop_potion(
    entry: Dict[str, Any],
    gold: int,
    potion_slots_open: int,
    hp_ratio: float,
    smith_candidates: List[Dict[str, Any]],
) -> float:
    if not entry.get("is_stocked") or potion_slots_open <= 0:
        return -999.0
    cost = int(entry.get("cost") or 0)
    if cost > gold:
        return -999.0

    score = 3.0
    if cost <= 60:
        score += 2
    score += _specific_potion_bonus(_normalize_name(entry.get("name")), hp_ratio, smith_candidates)
    return score


def _top_shop_candidates(state: Dict[str, Any], deck_profile: Dict[str, Any], player: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    gold = int(player.get("gold") or 0)
    potions = [pot for pot in player.get("potions", []) if pot]
    potion_slots_open = max(0, 3 - len(potions))
    hp_ratio = _hp_ratio(player)
    smith_candidates = deck_profile.get("smith_candidates", [])

    cards: List[Dict[str, Any]] = []
    for entry in state.get("cards", []):
        score = _score_shop_card(entry, deck_profile, gold)
        if score <= -999:
            continue
        cards.append(
            {
                "index": entry.get("index"),
                "name": _extract_name(entry.get("name")),
                "type": entry.get("type"),
                "cost": entry.get("cost"),
                "score": round(score, 2),
                "on_sale": bool(entry.get("on_sale")),
            }
        )
    cards.sort(key=lambda item: item["score"], reverse=True)

    relics: List[Dict[str, Any]] = []
    for entry in state.get("relics", []):
        score = _score_shop_relic(entry, deck_profile, gold, potion_slots_open)
        if score <= -999:
            continue
        relics.append(
            {
                "index": entry.get("index"),
                "name": _extract_name(entry.get("name")),
                "cost": entry.get("cost"),
                "score": round(score, 2),
            }
        )
    relics.sort(key=lambda item: item["score"], reverse=True)

    potions_out: List[Dict[str, Any]] = []
    for entry in state.get("potions", []):
        score = _score_shop_potion(entry, gold, potion_slots_open, hp_ratio, smith_candidates)
        if score <= -999:
            continue
        potions_out.append(
            {
                "index": entry.get("index"),
                "name": _extract_name(entry.get("name")),
                "cost": entry.get("cost"),
                "score": round(score, 2),
            }
        )
    potions_out.sort(key=lambda item: item["score"], reverse=True)

    return {
        "cards": cards[:3],
        "relics": relics[:2],
        "potions": potions_out[:2],
    }


def _build_run_plan(state: Dict[str, Any], facts: Dict[str, Any], deck_profile: Dict[str, Any]) -> List[str]:
    plan: List[str] = []
    hp_ratio = float(facts.get("hp_ratio") or 1.0)
    floor = int(facts.get("floor") or 0)
    gold = int(facts.get("gold") or 0)
    starter_cards = int(deck_profile.get("starter_cards", 0))
    smith_candidates = deck_profile.get("smith_candidates", [])
    removal_candidates = deck_profile.get("removal_candidates", [])

    if hp_ratio <= 0.45:
        plan.append("Survival is the priority: favor healing, safer lines, and efficient block.")
    elif hp_ratio <= 0.65:
        plan.append("Balance survival and scaling: avoid greedy damage-only choices unless they are discounted.")

    if starter_cards >= 5 and gold >= 75 and removal_candidates:
        plan.append("Trim starter cards when removal is affordable; Strike is the first cut, then Defend.")

    if smith_candidates and hp_ratio >= 0.55:
        top = smith_candidates[0]
        plan.append(f"Upgrade priority is {top['name']} for immediate tempo and scaling.")

    if int(deck_profile.get("draw_cards", 0)) == 0:
        plan.append("Deck lacks draw; prioritize consistency over narrow high-roll cards.")
    if int(deck_profile.get("vulnerable_cards", 0)) == 0 and int(deck_profile.get("attacks", 0)) >= 7:
        plan.append("Deck wants a better damage amplifier; value Vulnerable and front-loaded attacks.")
    if floor >= 8:
        plan.append("Boss pressure is close; avoid entering the boss low on HP or with an over-greedy route.")

    return plan[:4]


def _rest_site_context(state: Dict[str, Any], facts: Dict[str, Any], deck_profile: Dict[str, Any]) -> Dict[str, Any]:
    available = [
        str(option.get("option_id") or option.get("name") or option.get("index"))
        for option in state.get("options", [])
        if option.get("is_enabled")
    ]
    hp_ratio = float(facts.get("hp_ratio") or 1.0)
    floor = int(facts.get("floor") or 0)
    smith_candidates = deck_profile.get("smith_candidates", [])[:3]
    recommended = "SMITH"
    reasons: List[str] = []

    if hp_ratio <= 0.45:
        recommended = "HEAL"
        reasons.append("HP is in the danger zone.")
    elif hp_ratio <= 0.6 and floor >= 8:
        recommended = "HEAL"
        reasons.append("Boss is close and HP is still shaky.")
    elif not smith_candidates:
        recommended = "HEAL"
        reasons.append("No meaningful smith target stands out.")
    else:
        reasons.append("HP is stable enough to take a greedy upgrade.")
        reasons.append(f"Best upgrade target is {smith_candidates[0]['name']}.")

    if recommended not in available and available:
        recommended = available[0]

    return {
        "available_options": available,
        "recommended_option_id": recommended,
        "smith_candidates": smith_candidates,
        "reasons": reasons,
    }


def _shop_context(state: Dict[str, Any], facts: Dict[str, Any], deck_profile: Dict[str, Any]) -> Dict[str, Any]:
    player = state.get("player", {})
    gold = int(player.get("gold") or 0)
    hp_ratio = float(facts.get("hp_ratio") or 1.0)
    potions = [pot for pot in player.get("potions", []) if pot]
    potion_slots_open = max(0, 3 - len(potions))
    removal_cost = state.get("card_removal_cost")
    starter_cards = int(deck_profile.get("starter_cards", 0))
    removal_affordable = removal_cost is not None and gold >= int(removal_cost)
    top_candidates = _top_shop_candidates(state, deck_profile, player)

    priorities: List[str] = []
    if removal_affordable and starter_cards >= 5:
        priorities.append("Pay for card removal if there is no clearly stronger discounted buy.")

    if top_candidates["cards"]:
        priorities.append(f"Top card buy: {top_candidates['cards'][0]['name']}.")
    if top_candidates["relics"] and top_candidates["relics"][0]["score"] >= 18:
        priorities.append(f"Premium relic is available: {top_candidates['relics'][0]['name']}.")
    if potion_slots_open <= 0:
        priorities.append("Potion belt is full; avoid potion purchases unless a slot opens.")
    elif hp_ratio < 0.55 and top_candidates["potions"]:
        priorities.append(f"Emergency potion buy is reasonable: {top_candidates['potions'][0]['name']}.")

    if not priorities:
        priorities.append("Leave the shop if no removal or high-value purchase is available.")

    return {
        "gold": gold,
        "card_removal_cost": removal_cost,
        "removal_affordable": bool(removal_affordable),
        "potion_slots_open": potion_slots_open,
        "top_card_buys": top_candidates["cards"],
        "top_relic_buys": top_candidates["relics"],
        "top_potion_buys": top_candidates["potions"],
        "priorities": priorities[:4],
    }


def _decision_context(state: Dict[str, Any], facts: Dict[str, Any], deck_profile: Dict[str, Any]) -> Dict[str, Any]:
    decision = state.get("decision", state.get("type"))
    if decision == "rest_site":
        return {"decision": decision, **_rest_site_context(state, facts, deck_profile)}
    if decision == "shop":
        return {"decision": decision, **_shop_context(state, facts, deck_profile)}
    if decision == "card_reward":
        return {
            "decision": decision,
            "priorities": [
                "Prefer efficient non-starter cards over narrow build-arounds.",
                "Skip a low-impact reward if it dilutes the deck.",
            ],
        }
    if decision == "map_select":
        hp_ratio = float(facts.get("hp_ratio") or 1.0)
        gold = int(facts.get("gold") or 0)
        priorities = []
        if hp_ratio <= 0.5:
            priorities.append("Prefer safer nodes and rest sites when available.")
        if gold >= 140:
            priorities.append("A shop can be worthwhile if card removal or a discounted card is likely.")
        if not priorities:
            priorities.append("Prefer pathing that balances fights with future upgrades.")
        return {"decision": decision, "priorities": priorities}
    return {"decision": decision, "priorities": []}


@dataclass
class MemorySnapshot:
    current_run: Dict[str, Any]
    facts: Dict[str, Any]
    deck_profile: Dict[str, Any]
    run_plan: List[str]
    decision_context: Dict[str, Any]
    recent_events: List[Dict[str, Any]]
    recent_reflections: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_run": self.current_run,
            "facts": self.facts,
            "deck_profile": self.deck_profile,
            "run_plan": self.run_plan,
            "decision_context": self.decision_context,
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
            "deck_profile": {},
            "run_plan": [],
            "decision_context": {},
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
        self.working["facts"] = {}
        self.working["deck_profile"] = {}
        self.working["run_plan"] = []
        self.working["decision_context"] = {}
        self.working["recent_events"] = []
        self.working["recent_reflections"] = []
        self.working["last_updated"] = _utc_now()
        _safe_write_json(self.working_path, self.working)

    def snapshot(self, limit: int = 6) -> MemorySnapshot:
        return MemorySnapshot(
            current_run=self.working.get("current_run", {}),
            facts=self.working.get("facts", {}),
            deck_profile=self.working.get("deck_profile", {}),
            run_plan=self.working.get("run_plan", [])[-4:],
            decision_context=self.working.get("decision_context", {}),
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
        deck = player.get("deck", []) if isinstance(player, dict) else []
        deck_profile = _analyze_deck(deck if isinstance(deck, list) else [])
        facts = self.working.setdefault("facts", {})
        facts.update(
            {
                "decision": state.get("decision", state.get("type")),
                "room_type": context.get("room_type"),
                "act": context.get("act") or state.get("act"),
                "floor": context.get("floor") or state.get("floor"),
                "boss_name": _extract_name((context.get("boss") or {}).get("name")),
                "hp": player.get("hp"),
                "max_hp": player.get("max_hp"),
                "hp_ratio": round(_hp_ratio(player), 3),
                "gold": player.get("gold"),
                "deck_size": player.get("deck_size") or deck_profile.get("deck_size"),
                "potion_slots_used": len([pot for pot in player.get("potions", []) if pot]),
                "potion_slots_open": max(0, 3 - len([pot for pot in player.get("potions", []) if pot])),
            }
        )
        self.working["deck_profile"] = deck_profile
        self.working["run_plan"] = _build_run_plan(state, facts, deck_profile)
        self.working["decision_context"] = _decision_context(state, facts, deck_profile)
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
