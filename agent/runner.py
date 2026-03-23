#!/usr/bin/env python3
"""Main agent loop for STS2."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory import LayeredMemory
from agent.providers import ProviderDecision, build_provider
from agent.retrieval import LocalRetriever, RetrievalHit
from agent.runtime import ROOT, Sts2Process, compact_json


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _extract_name(obj: Any) -> str:
    if isinstance(obj, dict):
        if "en" in obj and obj["en"]:
            return str(obj["en"])
        if "zh" in obj and obj["zh"]:
            return str(obj["zh"])
    return str(obj)


def _default_knowledge_paths() -> List[str]:
    return [
        str(ROOT / "README.md"),
        str(ROOT / "agent" / "bug.md"),
        str(ROOT / "agent" / "knowledge"),
    ]


@dataclass
class RunnerConfig:
    provider: str = "openai"
    character: str = "Ironclad"
    seed: Optional[str] = None
    max_steps: int = 300
    provider_attempts: int = 3
    game_dir: Optional[str] = None
    dotnet: Optional[str] = None
    state_dir: str = str(ROOT / "agent" / "state")
    knowledge_paths: List[str] = field(default_factory=_default_knowledge_paths)
    no_build: bool = True
    verbose: bool = True


class AgentRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.provider = build_provider(config.provider)
        self.retriever = LocalRetriever.from_paths(config.knowledge_paths)
        self.memory = LayeredMemory(config.state_dir)

    def run(self) -> Dict[str, Any]:
        run_id = _utc_stamp()
        self.memory.begin_run(
            run_id,
            {
                "provider": self.provider.name,
                "character": self.config.character,
                "seed": self.config.seed,
                "knowledge_paths": self.config.knowledge_paths,
            },
        )

        runtime = Sts2Process(
            game_dir=self.config.game_dir,
            dotnet=self.config.dotnet,
            no_build=self.config.no_build,
        )
        ready = runtime.start()
        self.memory.reflect("startup", "Agent process started.", {"ready": ready})
        self._print(f"[run={run_id}] ready: {json.dumps(ready, ensure_ascii=False)}")

        state = runtime.send(
            {
                "cmd": "start_run",
                "character": self.config.character,
                "seed": self.config.seed or f"agent_{run_id}",
            }
        )

        try:
            step = 0
            while step < self.config.max_steps and state is not None:
                if state.get("type") == "error":
                    self.memory.reflect("engine_error", state.get("message", "unknown error"), {"step": step})
                    self._print(f"[step={step:03d}] engine error -> proceed: {state.get('message')}")
                    state = runtime.send({"cmd": "action", "action": "proceed"})
                    step += 1
                    continue

                self.memory.observe_state(state)
                if state.get("decision") == "game_over":
                    summary = "Victory" if state.get("victory") else "Defeat"
                    self.memory.reflect("game_over", summary, state)
                    self._print(f"[step={step:03d}] game_over -> {summary}")
                    return state

                retrieval_queries = self._build_queries(state)
                hits = self.retriever.search_many(retrieval_queries, limit=4)
                payload = {
                    "state": compact_json(state),
                    "memory": self.memory.snapshot().to_dict(),
                    "retrieval": [hit.to_dict() for hit in hits],
                }

                decision, command = self._request_validated_decision(state, payload)
                response = runtime.send(command)
                step += 1

                retrieval_dicts = [hit.to_dict() for hit in hits]
                self.memory.record_step(
                    step=step,
                    state=payload["state"],
                    command=command,
                    response=response,
                    provider_name=decision.provider_name,
                    rationale=decision.rationale,
                    memory_note=decision.memory_note,
                    decision_steps=decision.decision_steps,
                    retrieval_hits=retrieval_dicts,
                )
                self._print(
                    f"[step={step:03d}] {state.get('decision', state.get('type'))} "
                    f"-> {command.get('action', command.get('cmd'))} "
                    f"-> {None if response is None else response.get('decision', response.get('type'))}"
                )
                if decision.decision_steps:
                    self._print(f"           steps: {' | '.join(decision.decision_steps)}")
                state = response

            self.memory.reflect(
                "max_steps",
                "Reached max_steps without terminal state.",
                {"max_steps": self.config.max_steps},
            )
            return state or {"type": "error", "message": "No response from simulator"}
        finally:
            runtime.close()

    def _build_queries(self, state: Dict[str, Any]) -> List[str]:
        decision = state.get("decision", state.get("type", "unknown"))
        queries = [str(decision), f"{self.config.character} {decision}"]
        if decision == "combat_play":
            enemy_names = " ".join(_extract_name(enemy.get("name")) for enemy in state.get("enemies", []))
            if enemy_names:
                queries.append(f"combat {enemy_names}")
            queries.append("combat play card target enemy")
        elif decision == "rest_site":
            queries.append("rest site heal smith")
        elif decision == "card_reward":
            queries.append("card reward skip curse status")
        elif decision == "event_choice":
            queries.append("event choice risk reward")
        elif decision == "shop":
            queries.append("shop leave remove card")
        queries.append("bug tracker")
        return queries

    def _request_validated_decision(
        self,
        state: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> (ProviderDecision, Dict[str, Any]):
        feedback = None
        last_decision = None
        for attempt in range(1, self.config.provider_attempts + 1):
            request_payload = dict(payload)
            request_payload["attempt"] = attempt
            if feedback is not None:
                request_payload["provider_feedback"] = feedback
            decision = self.provider.decide(request_payload)
            command = self._validate_command(state, decision.command)
            if command is not None:
                return decision, command

            last_decision = decision
            feedback = {
                "validation_error": f"Invalid command for decision {state.get('decision')}: {decision.command}",
                "invalid_command": decision.command,
                "previous_rationale": decision.rationale,
                "previous_steps": decision.decision_steps,
            }
            self.memory.reflect(
                "provider_retry",
                "Provider returned an invalid command. Retrying with feedback.",
                {"attempt": attempt, "feedback": feedback},
            )

        raise RuntimeError(
            f"Provider failed to produce a valid command after {self.config.provider_attempts} attempts: "
            f"{None if last_decision is None else last_decision.command}"
        )

    def _validate_command(self, state: Dict[str, Any], command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if command.get("cmd") != "action":
            return None
        decision = state.get("decision", "")
        action = command.get("action")
        args = dict(command.get("args", {}))

        if decision == "map_select" and action == "select_map_node":
            for choice in state.get("choices", []):
                if choice.get("col") == args.get("col") and choice.get("row") == args.get("row"):
                    return {"cmd": "action", "action": action, "args": args}
            return None

        if decision == "combat_play":
            if action == "end_turn":
                return {"cmd": "action", "action": action}
            if action == "play_card":
                hand = {card["index"]: card for card in state.get("hand", [])}
                card = hand.get(args.get("card_index"))
                if not card or not card.get("can_play"):
                    return None
                if card.get("target_type") == "AnyEnemy":
                    enemies = {enemy["index"]: enemy for enemy in state.get("enemies", [])}
                    if "target_index" not in args and enemies:
                        args["target_index"] = min(enemies.values(), key=lambda item: item.get("hp", 999))["index"]
                    if args.get("target_index") not in enemies:
                        return None
                else:
                    args.pop("target_index", None)
                return {"cmd": "action", "action": action, "args": args}
            if action == "use_potion":
                potions = {pot["index"]: pot for pot in state.get("player", {}).get("potions", []) if pot}
                if args.get("potion_index") not in potions:
                    return None
                return {"cmd": "action", "action": action, "args": args}
            return None

        if decision == "card_reward":
            if action == "skip_card_reward":
                return {"cmd": "action", "action": action}
            if action == "select_card_reward":
                cards = {card["index"]: card for card in state.get("cards", [])}
                if args.get("card_index") in cards:
                    return {"cmd": "action", "action": action, "args": args}
            return None

        if decision == "bundle_select" and action == "select_bundle":
            bundles = {bundle["index"]: bundle for bundle in state.get("bundles", [])}
            if args.get("bundle_index") in bundles:
                return {"cmd": "action", "action": action, "args": args}
            return None

        if decision == "card_select":
            if action == "skip_select" and int(state.get("min_select", 1)) == 0:
                return {"cmd": "action", "action": action}
            if action == "select_cards":
                raw = args.get("indices", "")
                if isinstance(raw, list):
                    indices = [str(item) for item in raw]
                else:
                    indices = [item.strip() for item in str(raw).split(",") if item.strip()]
                cards = {str(card["index"]) for card in state.get("cards", [])}
                if not indices or any(index not in cards for index in indices):
                    return None
                min_select = int(state.get("min_select", 1))
                max_select = int(state.get("max_select", len(cards)))
                if not (min_select <= len(indices) <= max_select):
                    return None
                return {"cmd": "action", "action": action, "args": {"indices": ",".join(indices)}}
            return None

        if decision in {"rest_site", "event_choice"} and action == "choose_option":
            options = state.get("options", [])
            valid = {}
            for option in options:
                if decision == "event_choice" and option.get("is_locked"):
                    continue
                if decision == "rest_site" and not option.get("is_enabled"):
                    continue
                valid[option["index"]] = option
            if args.get("option_index") in valid:
                return {"cmd": "action", "action": action, "args": args}
            return None

        if decision == "event_choice" and action == "leave_room":
            return {"cmd": "action", "action": action}

        if decision == "shop":
            player_gold = state.get("player", {}).get("gold", 0) or 0
            if action == "leave_room":
                return {"cmd": "action", "action": action}
            if action == "remove_card":
                removal_cost = state.get("card_removal_cost")
                if removal_cost is None or player_gold >= removal_cost:
                    return {"cmd": "action", "action": action}
                return None
            if action == "buy_card":
                cards = {card["index"]: card for card in state.get("cards", [])}
                entry = cards.get(args.get("card_index"))
                if entry and entry.get("is_stocked") and player_gold >= (entry.get("cost") or 0):
                    return {"cmd": "action", "action": action, "args": {"card_index": entry["index"]}}
                return None
            if action == "buy_relic":
                relics = {relic["index"]: relic for relic in state.get("relics", [])}
                entry = relics.get(args.get("relic_index"))
                if entry and entry.get("is_stocked") and player_gold >= (entry.get("cost") or 0):
                    return {"cmd": "action", "action": action, "args": {"relic_index": entry["index"]}}
                return None
            if action == "buy_potion":
                potions = {potion["index"]: potion for potion in state.get("potions", [])}
                entry = potions.get(args.get("potion_index"))
                if entry and entry.get("is_stocked") and player_gold >= (entry.get("cost") or 0):
                    return {"cmd": "action", "action": action, "args": {"potion_index": entry["index"]}}
                return None
            return None

        if action == "proceed":
            return {"cmd": "action", "action": action}

        return None

    def _print(self, message: str) -> None:
        if self.config.verbose:
            print(message)
