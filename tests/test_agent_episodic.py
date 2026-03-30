#!/usr/bin/env python3
"""Regression tests for episodic retrieval."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.episodic import EpisodicRetriever


class EpisodicRetrieverTests(unittest.TestCase):
    def test_retrieves_similar_combat_episode(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)

        state_dir = Path(tempdir.name)
        episodes_path = state_dir / "episodes.jsonl"
        episode = {
            "run_id": "old-run",
            "step": 12,
            "state": {
                "decision": "combat_play",
                "context": {"floor": 8, "room_type": "Monster"},
                "player": {"hp": 40, "max_hp": 80, "deck_size": 12},
                "hand": [{"name": {"en": "Strike"}}, {"name": {"en": "Defend"}}],
                "enemies": [{"name": {"en": "Slithering Strangler"}, "hp": 21}],
            },
            "command": {"action": "play_card"},
            "response": {"decision": "combat_play"},
            "rationale": "Focus the Strangler first.",
            "memory_note": "Targeted the higher-damage threat first.",
        }
        episodes_path.write_text(json.dumps(episode, ensure_ascii=False) + "\n", encoding="utf-8")

        retriever = EpisodicRetriever(str(state_dir))
        query_state = {
            "decision": "combat_play",
            "context": {"floor": 9, "room_type": "Monster"},
            "player": {"hp": 38, "max_hp": 80},
            "hand": [{"name": {"en": "Strike"}}],
            "enemies": [{"name": {"en": "Slithering Strangler"}, "hp": 19}],
        }
        memory = {"deck_profile": {"starter_cards": 6}}

        hits = retriever.search(query_state, memory, limit=2)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].run_id, "old-run")
        self.assertEqual(hits[0].decision, "combat_play")
        self.assertGreater(hits[0].score, 6.0)


if __name__ == "__main__":
    unittest.main()
