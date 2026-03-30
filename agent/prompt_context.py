#!/usr/bin/env python3
"""Summaries for provider prompting and step-level observability."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List


def _clip(text: Any, limit: int = 220) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _copy_keys(source: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key in keys:
        if key in source and source[key] not in (None, "", [], {}):
            output[key] = source[key]
    return output


def summarize_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    facts = _copy_keys(
        memory.get("facts", {}) or {},
        [
            "decision",
            "room_type",
            "act",
            "floor",
            "boss_name",
            "hp",
            "max_hp",
            "hp_ratio",
            "gold",
            "deck_size",
            "potion_slots_open",
        ],
    )
    deck_profile = _copy_keys(
        memory.get("deck_profile", {}) or {},
        [
            "deck_size",
            "attacks",
            "skills",
            "powers",
            "upgraded_cards",
            "draw_cards",
            "block_cards",
            "vulnerable_cards",
            "zero_cost_cards",
            "starter_cards",
            "strength_sources",
        ],
    )
    if memory.get("deck_profile", {}).get("notable_cards"):
        deck_profile["notable_cards"] = list(memory["deck_profile"]["notable_cards"][:5])
    if memory.get("deck_profile", {}).get("smith_candidates"):
        deck_profile["smith_candidates"] = [
            {
                "name": item.get("name"),
                "score": item.get("score"),
                "reason": item.get("reason"),
            }
            for item in memory["deck_profile"]["smith_candidates"][:3]
        ]

    decision_context = _copy_keys(
        memory.get("decision_context", {}) or {},
        [
            "decision",
            "recommended_option_id",
            "available_options",
            "gold",
            "card_removal_cost",
            "removal_affordable",
            "potion_slots_open",
        ],
    )
    priorities = (memory.get("decision_context", {}) or {}).get("priorities") or []
    if priorities:
        decision_context["priorities"] = list(priorities[:3])

    recent_events = []
    for event in (memory.get("recent_events", []) or [])[-2:]:
        recent_events.append(
            _copy_keys(
                event,
                ["step", "decision", "action", "response", "floor", "hp", "gold", "provider", "memory_note"],
            )
        )

    recent_reflections = []
    for reflection in (memory.get("recent_reflections", []) or [])[-2:]:
        recent_reflections.append(
            _copy_keys(reflection, ["kind", "summary"])
        )

    return {
        "run_plan": list((memory.get("run_plan") or [])[:4]),
        "facts": facts,
        "deck_profile": deck_profile,
        "decision_context": decision_context,
        "recent_events": recent_events,
        "recent_reflections": recent_reflections,
    }


def summarize_world_model(world_model: Dict[str, Any]) -> Dict[str, Any]:
    if not world_model:
        return {}

    summary = _copy_keys(
        world_model,
        ["character", "strategy_mode", "hp_ratio", "gold", "room_preferences"],
    )
    summary["strategic_goals"] = list((world_model.get("strategic_goals") or [])[:3])

    recommended = world_model.get("recommended_choice") or {}
    if recommended:
        summary["recommended_choice"] = {
            "col": recommended.get("col"),
            "row": recommended.get("row"),
            "type": recommended.get("type"),
            "score": recommended.get("score"),
            "reasons": list((recommended.get("reasons") or [])[:2]),
        }

    top_choices = []
    for choice in (world_model.get("scored_choices") or [])[:3]:
        top_choices.append(
            {
                "col": choice.get("col"),
                "row": choice.get("row"),
                "type": choice.get("type"),
                "score": choice.get("score"),
                "reasons": list((choice.get("reasons") or [])[:2]),
            }
        )
    if top_choices:
        summary["top_choices"] = top_choices
    return summary


def summarize_skills(skills: Dict[str, Any]) -> Dict[str, Any]:
    if not skills:
        return {}

    summary = {
        "decision": skills.get("decision"),
        "selection_notes": list((skills.get("selection_notes") or [])[:3]),
    }
    primary = skills.get("primary_skill") or {}
    if primary:
        summary["primary_skill"] = {
            "name": primary.get("name"),
            "description": primary.get("description"),
            "trigger": primary.get("trigger"),
            "priorities": list((primary.get("priorities") or [])[:2]),
            "constraints": list((primary.get("constraints") or [])[:1]),
            "success_metric": primary.get("success_metric"),
            "confidence": primary.get("confidence"),
        }

    active = []
    for skill in (skills.get("active_skills") or [])[:3]:
        active.append(
            {
                "name": skill.get("name"),
                "priorities": list((skill.get("priorities") or [])[:2]),
                "constraints": list((skill.get("constraints") or [])[:1]),
                "confidence": skill.get("confidence"),
            }
        )
    if active:
        summary["active_skills"] = active
    return summary


def summarize_retrieval_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for hit in hits[:2]:
        summary.append(
            {
                "source": Path(str(hit.get("source") or "")).name,
                "title": hit.get("title"),
                "score": hit.get("score"),
                "excerpt": _clip(hit.get("content") or "", 180),
            }
        )
    return summary


def summarize_episodic_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for hit in hits[:2]:
        summary.append(
            {
                "run_id": hit.get("run_id"),
                "step": hit.get("step"),
                "decision": hit.get("decision"),
                "action": hit.get("action"),
                "score": hit.get("score"),
                "floor": hit.get("floor"),
                "hp_ratio": hit.get("hp_ratio"),
                "enemy_names": list((hit.get("enemy_names") or [])[:3]),
                "card_names": list((hit.get("card_names") or [])[:4]),
                "rationale": _clip(hit.get("rationale") or "", 140),
                "memory_note": _clip(hit.get("memory_note") or "", 140),
            }
        )
    return summary


def summarize_safety(safety: Dict[str, Any]) -> Dict[str, Any]:
    if not safety:
        return {}
    return {
        "hard_rules": list((safety.get("hard_rules") or [])[:3]),
        "warnings": list((safety.get("warnings") or [])[:3]),
        "suggested_actions": list((safety.get("suggested_actions") or [])[:3]),
        "risk_flags": list((safety.get("risk_flags") or [])[:4]),
    }


def build_prompt_context(
    memory: Dict[str, Any],
    world_model: Dict[str, Any],
    skills: Dict[str, Any],
    retrieval_hits: List[Dict[str, Any]],
    episodic_hits: List[Dict[str, Any]],
    safety: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "memory": summarize_memory(memory),
        "world_model": summarize_world_model(world_model),
        "skills": summarize_skills(skills),
        "retrieval": summarize_retrieval_hits(retrieval_hits),
        "episodic": summarize_episodic_hits(episodic_hits),
        "safety": summarize_safety(safety),
    }
