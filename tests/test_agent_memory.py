#!/usr/bin/env python3
"""Regression tests for agent memory summaries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.memory import LayeredMemory


def _ironclad_deck() -> list[dict]:
    return [
        {"name": {"en": "Strike"}, "cost": 1, "type": "Attack", "stats": {"damage": 6}},
        {"name": {"en": "Strike"}, "cost": 1, "type": "Attack", "stats": {"damage": 6}},
        {"name": {"en": "Strike"}, "cost": 1, "type": "Attack", "stats": {"damage": 6}},
        {"name": {"en": "Defend"}, "cost": 1, "type": "Skill", "stats": {"block": 5}},
        {"name": {"en": "Defend"}, "cost": 1, "type": "Skill", "stats": {"block": 5}},
        {
            "name": {"en": "Pommel Strike"},
            "cost": 1,
            "type": "Attack",
            "stats": {"damage": 9, "cards": 1},
            "upgraded": False,
        },
        {
            "name": {"en": "Rage"},
            "cost": 0,
            "type": "Skill",
            "stats": {"power": 3},
            "upgraded": False,
        },
    ]


class LayeredMemoryTests(unittest.TestCase):
    def _memory(self) -> LayeredMemory:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        return LayeredMemory(tempdir.name)

    def test_low_hp_rest_site_recommends_heal(self) -> None:
        memory = self._memory()
        memory.begin_run("run-1", {"provider": "openai", "character": "Ironclad"})
        state = {
            "type": "decision",
            "decision": "rest_site",
            "context": {"act": 1, "floor": 8, "room_type": "RestSite"},
            "options": [
                {"index": 0, "option_id": "HEAL", "is_enabled": True},
                {"index": 1, "option_id": "SMITH", "is_enabled": True},
            ],
            "player": {
                "hp": 25,
                "max_hp": 80,
                "gold": 90,
                "deck_size": 7,
                "potions": [],
                "deck": _ironclad_deck(),
            },
        }

        memory.observe_state(state)
        snapshot = memory.snapshot().to_dict()

        self.assertEqual(snapshot["decision_context"]["recommended_option_id"], "HEAL")
        self.assertTrue(snapshot["run_plan"])

    def test_high_hp_rest_site_recommends_smith(self) -> None:
        memory = self._memory()
        memory.begin_run("run-2", {"provider": "openai", "character": "Ironclad"})
        state = {
            "type": "decision",
            "decision": "rest_site",
            "context": {"act": 1, "floor": 4, "room_type": "RestSite"},
            "options": [
                {"index": 0, "option_id": "HEAL", "is_enabled": True},
                {"index": 1, "option_id": "SMITH", "is_enabled": True},
            ],
            "player": {
                "hp": 68,
                "max_hp": 80,
                "gold": 90,
                "deck_size": 7,
                "potions": [],
                "deck": _ironclad_deck(),
            },
        }

        memory.observe_state(state)
        snapshot = memory.snapshot().to_dict()

        self.assertEqual(snapshot["decision_context"]["recommended_option_id"], "SMITH")
        self.assertTrue(snapshot["decision_context"]["smith_candidates"])

    def test_shop_context_prefers_removal_and_scored_buys(self) -> None:
        memory = self._memory()
        memory.begin_run("run-3", {"provider": "openai", "character": "Ironclad"})
        state = {
            "type": "decision",
            "decision": "shop",
            "context": {"act": 1, "floor": 6, "room_type": "Shop"},
            "card_removal_cost": 75,
            "cards": [
                {
                    "index": 0,
                    "name": {"en": "Rage"},
                    "type": "Skill",
                    "cost": 37,
                    "is_stocked": True,
                    "on_sale": True,
                },
                {
                    "index": 1,
                    "name": {"en": "Bludgeon"},
                    "type": "Attack",
                    "cost": 72,
                    "is_stocked": True,
                    "on_sale": False,
                },
            ],
            "relics": [
                {"index": 0, "name": {"en": "Gremlin Horn"}, "cost": 250, "is_stocked": False}
            ],
            "potions": [
                {"index": 0, "name": {"en": "Flex Potion"}, "cost": 50, "is_stocked": True}
            ],
            "player": {
                "hp": 44,
                "max_hp": 80,
                "gold": 159,
                "deck_size": 7,
                "potions": [],
                "deck": _ironclad_deck(),
            },
        }

        memory.observe_state(state)
        snapshot = memory.snapshot().to_dict()
        context = snapshot["decision_context"]

        self.assertTrue(context["removal_affordable"])
        self.assertTrue(context["top_card_buys"])
        self.assertEqual(context["top_card_buys"][0]["name"], "Rage")
        self.assertTrue(any("remov" in item.lower() for item in context["priorities"]))

    def test_record_step_persists_agent_context_summary(self) -> None:
        memory = self._memory()
        memory.begin_run("run-4", {"provider": "openai", "character": "Ironclad"})
        memory.record_step(
            step=1,
            state={"decision": "map_select", "context": {"floor": 5}, "player": {"hp": 60, "gold": 90}},
            command={"cmd": "action", "action": "select_map_node", "args": {"col": 1, "row": 2}},
            response={"decision": "combat_play"},
            provider_name="openai",
            rationale="Take the safer route.",
            memory_note="Routing toward rest site.",
            decision_steps=["HP is medium.", "Path offers a rest site."],
            retrieval_hits=[{"source": "playbook.md", "title": "Map", "score": 1.2}],
            agent_context={
                "skills": {"primary_skill": {"name": "route_survival"}},
                "safety": {"warnings": ["Avoid elite."]},
                "episodic": [{"run_id": "old-run", "step": 12}],
                "world_model": {"strategy_mode": "survival"},
            },
            rl_transition={
                "ts": "2026-03-30T00:00:00Z",
                "run_id": "run-4",
                "step": 1,
                "provider": "openai",
                "character": "Ironclad",
                "decision": "map_select",
                "action": "select_map_node",
                "action_key": "select_map_node:1:2",
                "command": {"cmd": "action", "action": "select_map_node", "args": {"col": 1, "row": 2}},
                "available_actions": ["select_map_node"],
                "action_hints": {"choices": [{"col": 1, "row": 2}]},
                "chosen_action_features": {"node_type": "Monster"},
                "state": {"decision": "map_select"},
                "next_state": {"decision": "combat_play"},
                "state_features": {"strategy_mode": "survival"},
                "next_state_features": {"decision": "combat_play"},
                "reward": 0.42,
                "reward_breakdown": {"room_progress": 0.42, "total": 0.42},
                "done": False,
                "terminal_type": "continuing",
                "agent_context": {"world_model": {"strategy_mode": "survival"}},
                "metadata": {},
            },
        )

        snapshot = memory.snapshot().to_dict()
        self.assertEqual(
            snapshot["recent_events"][-1]["agent_context"]["skills"]["primary_skill"]["name"],
            "route_survival",
        )

        steps_path = Path(memory.run_steps_path)
        lines = steps_path.read_text(encoding="utf-8").splitlines()
        event = json.loads(lines[-1])
        self.assertEqual(event["agent_context"]["world_model"]["strategy_mode"], "survival")
        self.assertEqual(event["agent_context"]["episodic"][0]["run_id"], "old-run")
        self.assertEqual(snapshot["recent_events"][-1]["rl_reward"], 0.42)

        rl_dataset = Path(memory.rl_dataset_path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(rl_dataset[-1])["reward"], 0.42)

        run_rl = Path(memory.run_rl_path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(run_rl[-1])["terminal_type"], "continuing")


if __name__ == "__main__":
    unittest.main()
