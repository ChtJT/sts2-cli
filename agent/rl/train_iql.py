#!/usr/bin/env python3
"""Offline IQL-style candidate scorer for STS2 transitions."""

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
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for offline IQL training. Install torch in the sts2-cli environment first."
        ) from exc
    return torch, nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an offline IQL-style scorer from rl_transitions.jsonl.")
    parser.add_argument("--dataset", default=str(ROOT / "agent" / "state" / "rl_transitions.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "agent" / "state" / "rl_models" / "iql_model.pt"))
    parser.add_argument("--decision-types", default="map_select,rest_site,shop,card_reward,event_choice,bundle_select,card_select")
    parser.add_argument("--include-combat", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _group_rows(rows):
    groups: Dict[str, List] = {}
    for row in rows:
        groups.setdefault(row.group_id, []).append(row)
    return list(groups.values())


def _expectile_loss(diff, expectile):
    import torch

    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch, nn = _require_torch()
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    decisions = [item.strip() for item in args.decision_types.split(",") if item.strip()]
    transitions = load_transitions(args.dataset)
    rows = build_candidate_rows(transitions, decisions=decisions, include_combat=args.include_combat)
    groups = [group for group in _group_rows(rows) if any(item.chosen for item in group) and len(group) >= 2]
    if not groups:
        raise SystemExit("No training groups found. Collect rl_transitions.jsonl first.")

    state_dim = len(groups[0][0].state_vector)
    action_dim = len(groups[0][0].action_vector)
    sa_dim = state_dim + action_dim

    class MLP(nn.Module):
        def __init__(self, in_dim: int, hidden: int, out_dim: int = 1) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, x):
            return self.net(x)

    q_net = MLP(sa_dim, args.hidden_size, 1)
    v_net = MLP(state_dim, args.hidden_size, 1)
    actor = MLP(sa_dim, args.hidden_size, 1)

    q_opt = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=args.lr)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(groups)
        q_loss_total = 0.0
        v_loss_total = 0.0
        actor_loss_total = 0.0
        total_correct = 0
        total_groups = 0

        for group in groups:
            chosen = next(item for item in group if item.chosen)
            sa_tensor = torch.tensor([chosen.feature_vector], dtype=torch.float32)
            state_tensor = torch.tensor([chosen.state_vector], dtype=torch.float32)
            next_state_tensor = torch.tensor([chosen.next_state_vector], dtype=torch.float32)
            reward_tensor = torch.tensor([[chosen.reward]], dtype=torch.float32)
            done_tensor = torch.tensor([[float(chosen.done)]], dtype=torch.float32)

            with torch.no_grad():
                next_v = v_net(next_state_tensor)
                q_target = reward_tensor + args.discount * (1.0 - done_tensor) * next_v

            q_pred = q_net(sa_tensor)
            q_loss = F.mse_loss(q_pred, q_target)
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()

            with torch.no_grad():
                q_detached = q_net(sa_tensor)
            v_pred = v_net(state_tensor)
            v_loss = _expectile_loss(q_detached - v_pred, args.expectile)
            v_opt.zero_grad()
            v_loss.backward()
            v_opt.step()

            with torch.no_grad():
                advantage = (q_net(sa_tensor) - v_net(state_tensor)).clamp(-10.0, 10.0)
                weight = torch.exp(args.beta * advantage).clamp(max=20.0).squeeze().item()

            candidate_features = torch.tensor([item.feature_vector for item in group], dtype=torch.float32)
            scores = actor(candidate_features).squeeze(-1)
            chosen_idx = next(index for index, item in enumerate(group) if item.chosen)
            log_probs = F.log_softmax(scores, dim=0)
            actor_loss = -weight * log_probs[chosen_idx]

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            q_loss_total += float(q_loss.item())
            v_loss_total += float(v_loss.item())
            actor_loss_total += float(actor_loss.item())
            total_groups += 1
            if int(torch.argmax(scores).item()) == chosen_idx:
                total_correct += 1

        print(
            f"[iql] epoch={epoch:03d} "
            f"q_loss={q_loss_total / max(total_groups, 1):.4f} "
            f"v_loss={v_loss_total / max(total_groups, 1):.4f} "
            f"actor_loss={actor_loss_total / max(total_groups, 1):.4f} "
            f"top1_acc={total_correct / max(total_groups, 1):.3f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "q_state": q_net.state_dict(),
            "v_state": v_net.state_dict(),
            "actor_state": actor.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_size": args.hidden_size,
            "decision_types": decisions,
            "include_combat": args.include_combat,
            "discount": args.discount,
            "expectile": args.expectile,
            "beta": args.beta,
        },
        output_path,
    )
    metadata = {
        "dataset": str(Path(args.dataset).resolve()),
        "output": str(output_path.resolve()),
        "groups": len(groups),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "decision_types": decisions,
        "include_combat": args.include_combat,
        "discount": args.discount,
        "expectile": args.expectile,
        "beta": args.beta,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
