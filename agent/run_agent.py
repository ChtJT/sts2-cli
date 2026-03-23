#!/usr/bin/env python3
"""CLI entrypoint for the STS2 agent runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.runner import AgentRunner, RunnerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an automated STS2 agent.")
    parser.add_argument("--provider", default="openai", choices=["openai", "codex"], help="Decision provider.")
    parser.add_argument("--character", default="Ironclad", choices=["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"], help="Character to play.")
    parser.add_argument("--seed", default=None, help="Seed for reproducible runs.")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum number of agent steps before stopping.")
    parser.add_argument("--game-dir", default=None, help="Explicit Slay the Spire 2 data directory.")
    parser.add_argument("--dotnet", default=None, help="Explicit dotnet binary path.")
    parser.add_argument("--state-dir", default=None, help="Directory for layered memory and logs.")
    parser.add_argument("--knowledge", action="append", default=[], help="Extra knowledge path to index for retrieval. Can be repeated.")
    parser.add_argument("--build", action="store_true", help="Allow dotnet run to build instead of forcing --no-build.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = RunnerConfig(
        provider=args.provider,
        character=args.character,
        seed=args.seed,
        max_steps=args.max_steps,
        game_dir=args.game_dir,
        dotnet=args.dotnet,
        state_dir=args.state_dir or RunnerConfig().state_dir,
        no_build=not args.build,
        verbose=not args.quiet,
    )
    if args.knowledge:
        config.knowledge_paths.extend(args.knowledge)

    runner = AgentRunner(config)
    result = runner.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
