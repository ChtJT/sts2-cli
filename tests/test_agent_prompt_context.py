#!/usr/bin/env python3
"""Regression tests for compact provider prompt context."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from agent.prompt_context import build_prompt_context
from agent.providers import OpenAIProvider


class AgentPromptContextTests(unittest.TestCase):
    def test_build_prompt_context_compacts_sections(self) -> None:
        context = build_prompt_context(
            memory={
                "run_plan": ["Plan A", "Plan B"],
                "facts": {"hp": 52, "max_hp": 80, "hp_ratio": 0.65, "gold": 144, "floor": 9},
                "deck_profile": {
                    "starter_cards": 5,
                    "notable_cards": ["Rage", "Pommel Strike", "Inflame", "Shrug It Off", "Battle Trance", "Wild Strike"],
                    "smith_candidates": [
                        {"name": "Pommel Strike", "score": 15.4, "reason": "draw 1"},
                        {"name": "Inflame", "score": 14.1, "reason": "strength"},
                    ],
                },
                "decision_context": {"decision": "shop", "priorities": ["Remove Strike", "Buy premium relic"]},
                "recent_events": [{"step": 8, "decision": "map_select", "action": "select_map_node", "response": "combat_play"}],
                "recent_reflections": [{"kind": "provider_retry", "summary": "invalid card index"}],
            },
            world_model={
                "strategy_mode": "stabilize",
                "strategic_goals": ["Reach a rest site.", "Avoid an elite this floor."],
                "recommended_choice": {"col": 2, "row": 5, "type": "RestSite", "score": 19.2, "reasons": ["Safer route", "Boss soon"]},
                "scored_choices": [{"col": 2, "row": 5, "type": "RestSite", "score": 19.2, "reasons": ["Safer route", "Boss soon", "Extra"]}],
            },
            skills={
                "decision": "shop",
                "primary_skill": {
                    "name": "shop_remove_starters",
                    "description": "Thin the deck.",
                    "trigger": "Removal affordable.",
                    "priorities": ["Remove Strike", "Skip filler"],
                    "constraints": ["Do not buy filler"],
                    "success_metric": "Deck improves",
                    "confidence": 0.91,
                },
                "active_skills": [
                    {"name": "shop_remove_starters", "priorities": ["Remove Strike"], "constraints": ["Do not buy filler"], "confidence": 0.91},
                    {"name": "shop_leave_disciplined", "priorities": ["Leave"], "constraints": ["No filler"], "confidence": 0.58},
                ],
                "selection_notes": ["Primary skill selected."],
            },
            retrieval_hits=[
                {"source": "/tmp/playbook.md", "title": "Shop", "score": 1.8, "content": "Remove starter cards before buying low-impact filler cards."},
                {"source": "/tmp/bug.md", "title": "Merchant", "score": 1.2, "content": "Do not buy potions with full slots."},
            ],
            episodic_hits=[
                {"run_id": "run-1", "step": 19, "decision": "shop", "action": "remove_card", "score": 8.4, "floor": 9, "hp_ratio": 0.62, "enemy_names": [], "card_names": ["strike"], "rationale": "Removed Strike.", "memory_note": "Good thinning."},
            ],
            safety={"hard_rules": ["No potion buys with full slots."], "warnings": ["Removal is better than filler."], "suggested_actions": ["Remove Strike"], "risk_flags": ["starter_bloat"]},
        )

        self.assertIn("memory", context)
        self.assertIn("skills", context)
        self.assertEqual(context["skills"]["primary_skill"]["name"], "shop_remove_starters")
        self.assertEqual(len(context["retrieval"]), 2)
        self.assertNotIn("Wild Strike", json.dumps(context, ensure_ascii=False))

    def test_provider_prompt_uses_prompt_context_only(self) -> None:
        env = patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)
        provider = OpenAIProvider()

        payload = {
            "state": {"decision": "shop", "player": {"gold": 140}},
            "prompt_context": {
                "memory": {"run_plan": ["Remove Strike first."]},
                "skills": {"primary_skill": {"name": "shop_remove_starters"}},
            },
            "memory": {"run_plan": ["THIS SHOULD NOT APPEAR RAW"]},
            "skills": {"primary_skill": {"name": "raw_skill"}},
            "world_model": {"strategy_mode": "raw"},
            "retrieval": [{"content": "raw retrieval"}],
            "episodic": [{"memory_note": "raw episode"}],
            "safety": {"warnings": ["raw warning"]},
        }

        prompt = json.loads(provider._user_prompt(payload))

        self.assertIn("prompt_context", prompt)
        self.assertEqual(prompt["strategy_focus"]["skills"]["primary_skill"]["name"], "shop_remove_starters")
        self.assertNotIn("memory", prompt)
        self.assertNotIn("retrieval", prompt)
        self.assertNotIn("episodic_examples", prompt)


if __name__ == "__main__":
    unittest.main()
