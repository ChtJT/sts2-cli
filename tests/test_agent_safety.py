#!/usr/bin/env python3
"""Regression tests for agent safety policy guardrails."""

from __future__ import annotations

import unittest

from agent.safety import AgentSafetyPolicy


class AgentSafetyPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = AgentSafetyPolicy("Ironclad")

    def test_low_hp_rest_site_blocks_smith(self) -> None:
        state = {
            "decision": "rest_site",
            "options": [
                {"index": 0, "option_id": "HEAL", "is_enabled": True},
                {"index": 1, "option_id": "SMITH", "is_enabled": True},
            ],
        }
        memory = {
            "facts": {"hp_ratio": 0.3},
            "decision_context": {"recommended_option_id": "HEAL"},
        }

        result = self.policy.validate(
            state,
            {"cmd": "action", "action": "choose_option", "args": {"option_index": 1}},
            memory,
        )

        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "critical_hp_requires_heal")

    def test_full_potion_slots_block_buy_potion(self) -> None:
        state = {"decision": "shop"}
        memory = {"facts": {"potion_slots_open": 0}}

        result = self.policy.validate(
            state,
            {"cmd": "action", "action": "buy_potion", "args": {"potion_index": 0}},
            memory,
        )

        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "potion_slots_full")

    def test_map_context_mentions_world_model_preference(self) -> None:
        state = {"decision": "map_select"}
        memory = {"facts": {"hp_ratio": 0.8}, "deck_profile": {}}
        world_model = {"recommended_choice": {"col": 3, "row": 5, "type": "RestSite"}}

        context = self.policy.build_context(state, memory, world_model).to_dict()

        self.assertTrue(context["suggested_actions"])
        self.assertIn("(3,5)", context["suggested_actions"][0])


if __name__ == "__main__":
    unittest.main()
