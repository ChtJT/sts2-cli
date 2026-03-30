#!/usr/bin/env python3
"""Regression tests for continuous RL reward shaping."""

from __future__ import annotations

import unittest

from agent.rl.reward import ContinuousRewardModel


class RewardModelTests(unittest.TestCase):
    def test_combat_win_reward_is_positive(self) -> None:
        model = ContinuousRewardModel()
        state = {
            "decision": "combat_play",
            "context": {"room_type": "Monster", "floor": 4},
            "player": {
                "hp": 52,
                "max_hp": 80,
                "gold": 90,
                "potions": [],
                "deck": [{"name": {"en": "Strike"}, "type": "Attack", "cost": 1}],
            },
            "hand": [{"index": 0, "name": {"en": "Strike"}, "type": "Attack", "cost": 1, "can_play": True, "target_type": "AnyEnemy", "stats": {"damage": 6}}],
            "enemies": [{"index": 1, "name": {"en": "Slime"}, "hp": 6, "intents": [{"damage": 5, "hits": 1}]}],
        }
        command = {"cmd": "action", "action": "play_card", "args": {"card_index": 0, "target_index": 1}}
        next_state = {
            "decision": "card_reward",
            "context": {"room_type": "Monster", "floor": 4},
            "player": {
                "hp": 52,
                "max_hp": 80,
                "gold": 105,
                "potions": [],
                "deck": [{"name": {"en": "Strike"}, "type": "Attack", "cost": 1}],
            },
            "cards": [{"index": 0, "name": {"en": "Pommel Strike"}, "type": "Attack", "cost": 1}],
        }

        result = model.evaluate(state, command, next_state)

        self.assertGreater(result.reward, 0.0)
        self.assertGreater(result.breakdown.room_progress, 0.0)
        self.assertEqual(result.terminal_type, "continuing")
        self.assertFalse(result.done)

    def test_defeat_terminal_reward_marks_done(self) -> None:
        model = ContinuousRewardModel()
        state = {
            "decision": "combat_play",
            "player": {"hp": 5, "max_hp": 80, "gold": 20, "deck": []},
            "hand": [],
            "enemies": [{"index": 1, "hp": 14, "intents": [{"damage": 12, "hits": 1}]}],
        }
        command = {"cmd": "action", "action": "end_turn"}
        next_state = {"decision": "game_over", "victory": False, "player": {"hp": 0, "max_hp": 80, "gold": 20, "deck": []}}

        result = model.evaluate(state, command, next_state)

        self.assertTrue(result.done)
        self.assertEqual(result.terminal_type, "defeat")
        self.assertLess(result.reward, -1.0)


if __name__ == "__main__":
    unittest.main()

