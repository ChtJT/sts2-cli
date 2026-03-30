#!/usr/bin/env python3
"""Regression tests for explicit skill selection."""

from __future__ import annotations

import unittest

from agent.skills import SkillRegistry


class SkillRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = SkillRegistry("Ironclad")

    def test_low_hp_map_prefers_survival_skill(self) -> None:
        state = {
            "decision": "map_select",
            "choices": [
                {"col": 0, "row": 1, "type": "Elite"},
                {"col": 1, "row": 1, "type": "RestSite"},
            ],
        }
        memory = {
            "facts": {"hp_ratio": 0.3, "gold": 90},
            "deck_profile": {"starter_cards": 6},
        }
        world_model = {"strategy_mode": "survival", "recommended_choice": {"col": 1, "row": 1, "type": "RestSite"}}

        result = self.registry.select(state, memory, world_model)

        self.assertEqual(result["primary_skill"]["name"], "route_survival")

    def test_shop_prefers_removal_skill_when_affordable(self) -> None:
        state = {
            "decision": "shop",
            "card_removal_cost": 75,
            "cards": [],
            "relics": [],
        }
        memory = {
            "facts": {"gold": 150, "hp_ratio": 0.8},
            "deck_profile": {"starter_cards": 7},
        }

        result = self.registry.select(state, memory, {})

        active = [skill["name"] for skill in result["active_skills"]]
        self.assertIn("shop_remove_starters", active)

    def test_combat_can_activate_finish_skill(self) -> None:
        state = {
            "decision": "combat_play",
            "player": {"hp": 40, "max_hp": 80},
            "hand": [
                {"name": {"en": "Strike"}, "type": "Attack", "can_play": True},
                {"name": {"en": "Defend"}, "type": "Skill", "can_play": True},
            ],
            "enemies": [
                {"name": {"en": "Cultist"}, "hp": 8, "intents": []},
            ],
        }
        memory = {
            "facts": {"hp_ratio": 0.5, "hp": 40},
            "deck_profile": {},
        }

        result = self.registry.select(state, memory)

        active = [skill["name"] for skill in result["active_skills"]]
        self.assertIn("combat_finish_low_hp_target", active)


if __name__ == "__main__":
    unittest.main()
