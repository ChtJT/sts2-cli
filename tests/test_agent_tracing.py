#!/usr/bin/env python3
"""Regression tests for agent trace logging."""

from __future__ import annotations

import json
import tempfile
import unittest

from agent.tracing import AgentTraceRecorder


class AgentTraceRecorderTests(unittest.TestCase):
    def test_trace_file_records_nodes(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)

        tracer = AgentTraceRecorder(tempdir.name)
        tracer.begin_run("run-1", {"provider": "openai"})
        tracer.record(
            node="provider_request",
            step=2,
            status="ok",
            inputs={"decision": "combat_play"},
            outputs={"command": {"action": "end_turn"}},
        )

        trace_path = tracer.current_path()
        self.assertIsNotNone(trace_path)
        lines = trace_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        self.assertEqual(first["node"], "run_start")
        self.assertEqual(second["node"], "provider_request")
        self.assertEqual(second["step"], 2)


if __name__ == "__main__":
    unittest.main()
