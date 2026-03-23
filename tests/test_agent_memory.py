#!/usr/bin/env python3
"""Regression tests for agent memory summaries."""

from __future__ import annotations

import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
