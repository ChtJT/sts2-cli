#!/usr/bin/env python3
"""Regression tests for RL dataset helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.rl.dataset import build_candidate_rows, command_key, enumerate_action_candidates, summarize_action_for_rl
from agent.rl.schema import RLTransition


class RLDatasetTests(unittest.TestCase):
    def test_enumerate_shop_candidates_includes_leave_and_buys(self) -> None:
        state = {
            "decision": "shop",
            "player": {"gold": 120, "deck": []},
            "card_removal_cost": 75,
            "cards": [{"index": 0, "name": {"en": "Rage"}, "type": "Skill", "cost": 37, "is_stocked": True, "on_sale": True}],
            "relics": [],
            "potions": [],
        }

        candidates = enumerate_action_candidates(state)
        keys = {item.key for item in candidates}

        self.assertIn("leave_room", keys)
        self.assertIn("remove_card", keys)
        self.assertIn("buy_card:0", keys)

    def test_build_candidate_rows_marks_chosen_action(self) -> None:
        transition = RLTransition(
            ts="2026-03-30T00:00:00Z",
            run_id="run-1",
            step=3,
            provider="openai",
            character="Ironclad",
            decision="rest_site",
            action="choose_option",
            action_key="choose_option:1",
            command={"cmd": "action", "action": "choose_option", "args": {"option_index": 1}},
            available_actions=["choose_option"],
            action_hints={},
            chosen_action_features={"option_id": "SMITH"},
            state={
                "decision": "rest_site",
                "context": {"act": 1, "floor": 8, "room_type": "RestSite"},
                "options": [
                    {"index": 0, "option_id": "HEAL", "is_enabled": True},
                    {"index": 1, "option_id": "SMITH", "is_enabled": True},
                ],
                "player": {"hp": 70, "max_hp": 80, "gold": 90, "deck": []},
            },
            next_state={"decision": "map_select", "player": {"hp": 70, "max_hp": 80, "gold": 90, "deck": []}},
            state_features={"decision": "rest_site", "room_type": "RestSite", "strategy_mode": "balanced", "primary_skill_bucket": "rest"},
            next_state_features={"decision": "map_select", "room_type": "", "strategy_mode": "balanced", "primary_skill_bucket": "generic"},
            reward=0.35,
            reward_breakdown={"room_progress": 0.35, "total": 0.35},
            done=False,
            terminal_type="continuing",
        )

        rows = build_candidate_rows([transition], decisions=["rest_site"])
        chosen = [row for row in rows if row.chosen]

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0].action_key, "choose_option:1")
        self.assertGreater(len(chosen[0].feature_vector), 10)

    def test_action_summary_matches_command_key(self) -> None:
        state = {
            "decision": "map_select",
            "choices": [{"col": 2, "row": 4, "type": "RestSite"}],
        }
        command = {"cmd": "action", "action": "select_map_node", "args": {"col": 2, "row": 4}}
        summary = summarize_action_for_rl(state, command)

        self.assertEqual(summary["action_key"], command_key(command))
        self.assertEqual(summary["node_type"], "RestSite")


if __name__ == "__main__":
    unittest.main()
