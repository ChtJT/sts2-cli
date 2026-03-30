#!/usr/bin/env python3
"""Offline behavior cloning over logged STS2 RL transitions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl.dataset import build_candidate_rows, load_transitions


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for offline BC training. Install torch in the sts2-cli environment first."
        ) from exc
    return torch, nn, F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an offline BC scorer from rl_transitions.jsonl.")
    parser.add_argument("--dataset", default=str(ROOT / "agent" / "state" / "rl_transitions.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "agent" / "state" / "rl_models" / "bc_model.pt"))
    parser.add_argument("--decision-types", default="map_select,rest_site,shop,card_reward,event_choice,bundle_select,card_select")
    parser.add_argument("--include-combat", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _group_rows(rows):
    groups: Dict[str, List] = {}
    for row in rows:
        groups.setdefault(row.group_id, []).append(row)
    return list(groups.values())


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch, nn, F = _require_torch()
    torch.manual_seed(args.seed)

    decisions = [item.strip() for item in args.decision_types.split(",") if item.strip()]
    transitions = load_transitions(args.dataset)
    rows = build_candidate_rows(transitions, decisions=decisions, include_combat=args.include_combat)
    groups = [group for group in _group_rows(rows) if any(item.chosen for item in group) and len(group) >= 2]
    if not groups:
        raise SystemExit("No training groups found. Collect rl_transitions.jsonl first.")

    input_dim = len(groups[0][0].feature_vector)

    class CandidateScorer(nn.Module):
        def __init__(self, dim: int, hidden: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = CandidateScorer(input_dim, args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(groups)
        total_loss = 0.0
        total_correct = 0
        total_groups = 0
        for group in groups:
            features = torch.tensor([item.feature_vector for item in group], dtype=torch.float32)
            chosen_idx = next(index for index, item in enumerate(group) if item.chosen)

            scores = model(features)
            loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([chosen_idx], dtype=torch.long))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_groups += 1
            if int(torch.argmax(scores).item()) == chosen_idx:
                total_correct += 1

        avg_loss = total_loss / max(total_groups, 1)
        accuracy = total_correct / max(total_groups, 1)
        print(f"[bc] epoch={epoch:03d} loss={avg_loss:.4f} top1_acc={accuracy:.3f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "hidden_size": args.hidden_size,
            "decisions": decisions,
            "include_combat": args.include_combat,
        },
        output_path,
    )
    metadata = {
        "dataset": str(Path(args.dataset).resolve()),
        "output": str(output_path.resolve()),
        "groups": len(groups),
        "input_dim": input_dim,
        "decision_types": decisions,
        "include_combat": args.include_combat,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

