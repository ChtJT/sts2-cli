#!/usr/bin/env python3
"""Main agent loop for STS2."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.combat_log import CombatLogRecorder
from agent.episodic import EpisodicRetriever
from agent.memory import LayeredMemory
from agent.prompt_context import build_prompt_context
from agent.providers import ProviderDecision, build_provider
from agent.rl.dataset import (
    action_hints_from_state,
    available_actions_from_state,
    command_key,
    summarize_action_for_rl,
    summarize_state_for_rl,
)
from agent.rl.reward import ContinuousRewardModel
from agent.rl.schema import RLTransition
from agent.retrieval import LocalRetriever, RetrievalHit
from agent.runtime import ROOT, Sts2Process, compact_json
from agent.safety import AgentSafetyPolicy
from agent.skills import SkillRegistry
from agent.tracing import AgentTraceRecorder
from agent.world_model import WorldModelPlanner


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


def _state_signature(state: Dict[str, Any]) -> str:
    context = state.get("context", {}) if isinstance(state, dict) else {}
    player = state.get("player", {}) if isinstance(state, dict) else {}
    hand = state.get("hand", []) if isinstance(state, dict) else []
    enemies = state.get("enemies", []) if isinstance(state, dict) else []
    choices = state.get("choices", []) if isinstance(state, dict) else []
    payload = {
        "decision": state.get("decision", state.get("type")),
        "floor": context.get("floor", state.get("floor")),
        "room_type": context.get("room_type"),
        "hp": player.get("hp"),
        "gold": player.get("gold"),
        "energy": state.get("energy"),
        "hand": [
            (card.get("index"), card.get("can_play"), card.get("cost"))
            for card in hand[:10]
        ],
        "enemies": [
            (enemy.get("index"), enemy.get("hp"), enemy.get("block"))
            for enemy in enemies[:6]
        ],
        "choices": [
            (
                choice.get("index"),
                choice.get("col"),
                choice.get("row"),
                choice.get("type"),
                choice.get("option_id"),
            )
            for choice in choices[:12]
        ],
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


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
    max_identical_states: int = 4


class AgentRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.provider = build_provider(config.provider)
        self.retriever = LocalRetriever.from_paths(config.knowledge_paths)
        self.episodic = EpisodicRetriever(config.state_dir)
        self.memory = LayeredMemory(config.state_dir)
        self.world_model = WorldModelPlanner(config.character)
        self.combat_log = CombatLogRecorder(config.state_dir)
        self.safety = AgentSafetyPolicy(config.character)
        self.skills = SkillRegistry(config.character)
        self.tracer = AgentTraceRecorder(config.state_dir)
        self.reward_model = ContinuousRewardModel()
        self._last_state_signature: Optional[str] = None
        self._identical_state_count = 0

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
        self.combat_log.begin_run(run_id)
        self.tracer.begin_run(
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
        self.tracer.record("runtime_start", step=0, status="ok", outputs={"ready": ready})
        self._print(f"[run={run_id}] ready: {json.dumps(ready, ensure_ascii=False)}")

        state = runtime.send(
            {
                "cmd": "start_run",
                "character": self.config.character,
                "seed": self.config.seed or f"agent_{run_id}",
            }
        )
        self.tracer.record(
            "start_run",
            step=0,
            status="ok",
            outputs={"decision": None if state is None else state.get("decision", state.get("type"))},
        )

        try:
            step = 0
            while step < self.config.max_steps and state is not None:
                self.tracer.record(
                    "observe_state",
                    step=step,
                    status="ok",
                    inputs={
                        "decision": state.get("decision", state.get("type")),
                        "floor": state.get("context", {}).get("floor", state.get("floor")),
                    },
                )
                if state.get("type") == "error":
                    self.memory.reflect("engine_error", state.get("message", "unknown error"), {"step": step})
                    self.tracer.record(
                        "stop",
                        step=step,
                        status="error",
                        outputs={"message": state.get("message")},
                    )
                    self._print(f"[step={step:03d}] engine error -> stop: {state.get('message')}")
                    self._print_combat_log_path()
                    return state

                self.memory.observe_state(state)
                compact_state = compact_json(state)
                repeated_state = self._check_repeated_state(state)
                if repeated_state is not None:
                    self.memory.reflect("engine_stuck", repeated_state["message"], {"step": step})
                    self.tracer.record(
                        "stop",
                        step=step,
                        status="error",
                        outputs=repeated_state,
                    )
                    self._print(f"[step={step:03d}] engine stuck -> stop: {repeated_state['message']}")
                    self._print_combat_log_path()
                    return repeated_state
                if state.get("decision") == "game_over":
                    summary = "Victory" if state.get("victory") else "Defeat"
                    self.memory.reflect("game_over", summary, state)
                    self.tracer.record(
                        "stop",
                        step=step,
                        status="ok",
                        outputs={"decision": "game_over", "victory": state.get("victory")},
                    )
                    self._print(f"[step={step:03d}] game_over -> {summary}")
                    if not state.get("victory"):
                        self._print_combat_log_path()
                    return state

                retrieval_queries = self._build_queries(state)
                hits = self.retriever.search_many(retrieval_queries, limit=4)
                self.tracer.record(
                    "retrieve_context",
                    step=step,
                    status="ok",
                    inputs={"queries": retrieval_queries},
                    outputs={"hits": [hit.to_dict() for hit in hits]},
                )
                memory_snapshot = self.memory.snapshot().to_dict()
                if state.get("decision") == "map_select":
                    map_data = runtime.send({"cmd": "get_map"})
                    if map_data and map_data.get("type") == "map":
                        world_model = self.world_model.plan(compact_state, map_data, memory_snapshot)
                        self.memory.update_world_model(world_model)
                        memory_snapshot = self.memory.snapshot().to_dict()
                        self.tracer.record(
                            "world_model_plan",
                            step=step,
                            status="ok",
                            outputs={"recommended_choice": world_model.get("recommended_choice")},
                        )
                safety_context = self.safety.build_context(
                    compact_state,
                    memory_snapshot,
                    memory_snapshot.get("world_model", {}),
                ).to_dict()
                episodic_hits = self.episodic.search(
                    compact_state,
                    memory_snapshot,
                    limit=3,
                    exclude_run_id=self.memory.run_id,
                )
                self.tracer.record(
                    "episodic_retrieval",
                    step=step,
                    status="ok",
                    outputs={"hits": [hit.to_dict() for hit in episodic_hits]},
                )
                skills = self.skills.select(
                    compact_state,
                    memory_snapshot,
                    memory_snapshot.get("world_model", {}),
                    safety_context,
                    [hit.to_dict() for hit in episodic_hits],
                )
                self.memory.update_skills(skills)
                memory_snapshot = self.memory.snapshot().to_dict()
                retrieval_dicts = [hit.to_dict() for hit in hits]
                episodic_dicts = [hit.to_dict() for hit in episodic_hits]
                prompt_context = build_prompt_context(
                    memory_snapshot,
                    memory_snapshot.get("world_model", {}),
                    memory_snapshot.get("skills", {}),
                    retrieval_dicts,
                    episodic_dicts,
                    safety_context,
                )
                self.tracer.record(
                    "skill_selection",
                    step=step,
                    status="ok",
                    outputs={
                        "primary_skill": skills.get("primary_skill"),
                        "selection_notes": skills.get("selection_notes", []),
                    },
                )
                self.tracer.record(
                    "prompt_context",
                    step=step,
                    status="ok",
                    outputs=prompt_context,
                    metadata={
                        "chars": len(json.dumps(prompt_context, ensure_ascii=False)),
                    },
                )
                payload = {
                    "state": compact_state,
                    "memory": memory_snapshot,
                    "prompt_context": prompt_context,
                }

                decision, command = self._request_validated_decision(step, state, payload)
                self.combat_log.record(step + 1, state, command, decision.decision_steps, decision.rationale)
                response = runtime.send(command)
                self.combat_log.finalize(state, response)
                self.tracer.record(
                    "execute_action",
                    step=step + 1,
                    status="ok",
                    inputs={"command": command},
                    outputs={"response": None if response is None else response.get("decision", response.get("type"))},
                )
                step += 1
                rl_transition = self._build_rl_transition(
                    run_id=run_id,
                    step=step,
                    state=state,
                    command=command,
                    response=response,
                    decision=decision,
                    agent_context={
                        "skills": prompt_context.get("skills", {}),
                        "safety": prompt_context.get("safety", {}),
                        "episodic": prompt_context.get("episodic", []),
                        "world_model": prompt_context.get("world_model", {}),
                        "memory": prompt_context.get("memory", {}),
                    },
                )

                self.memory.record_step(
                    step=step,
                    state=compact_state,
                    command=command,
                    response=response,
                    provider_name=decision.provider_name,
                    rationale=decision.rationale,
                    memory_note=decision.memory_note,
                    decision_steps=decision.decision_steps,
                    retrieval_hits=prompt_context.get("retrieval", []),
                    agent_context={
                        "skills": prompt_context.get("skills", {}),
                        "safety": prompt_context.get("safety", {}),
                        "episodic": prompt_context.get("episodic", []),
                        "world_model": prompt_context.get("world_model", {}),
                        "memory": prompt_context.get("memory", {}),
                    },
                    rl_transition=rl_transition.to_dict(),
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
            self.tracer.record(
                "stop",
                step=step,
                status="error",
                outputs={"message": "Reached max_steps without terminal state."},
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
            queries.append("rest site heal smith threshold best upgrade target")
        elif decision == "card_reward":
            queries.append("card reward skip curse status deck dilution efficient pick")
        elif decision == "event_choice":
            queries.append("event choice risk reward")
        elif decision == "shop":
            queries.append("shop card removal discounted card relic priority potion slots")
        queries.append("bug tracker")
        return queries

    def _request_validated_decision(
        self,
        step: int,
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
            self.tracer.record(
                "provider_request",
                step=step,
                status="ok",
                inputs={
                    "attempt": attempt,
                    "decision": state.get("decision", state.get("type")),
                    "provider_feedback": feedback,
                },
            )
            try:
                decision = self.provider.decide(request_payload)
            except Exception as exc:
                self.tracer.record(
                    "provider_decision",
                    step=step,
                    status="error",
                    metadata={"attempt": attempt},
                    error=str(exc),
                )
                raise
            self.tracer.record(
                "provider_decision",
                step=step,
                status="ok",
                outputs={
                    "command": decision.command,
                    "rationale": decision.rationale,
                    "decision_steps": decision.decision_steps,
                },
                metadata={"attempt": attempt},
            )
            command = self._validate_command(state, decision.command)
            if command is not None:
                safety = self.safety.validate(state, command, payload.get("memory", {}))
                if safety.allowed:
                    self.tracer.record(
                        "validate_command",
                        step=step,
                        status="ok",
                        outputs={"command": command},
                        metadata={"attempt": attempt, "warnings": safety.warnings},
                    )
                    return decision, command

                last_decision = decision
                feedback = {
                    "validation_error": f"Safety policy rejected command for decision {state.get('decision')}: {safety.reason}",
                    "invalid_command": decision.command,
                    "previous_rationale": decision.rationale,
                    "previous_steps": decision.decision_steps,
                    "safety_warnings": safety.warnings,
                }
                self.tracer.record(
                    "validate_command",
                    step=step,
                    status="error",
                    outputs={"command": command},
                    metadata={"attempt": attempt, "reason": safety.reason, "warnings": safety.warnings},
                    error=safety.reason,
                )
                self.memory.reflect(
                    "provider_retry",
                    "Provider returned a command rejected by the safety policy. Retrying with feedback.",
                    {"attempt": attempt, "feedback": feedback},
                )
                continue

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
            self.tracer.record(
                "validate_command",
                step=step,
                status="error",
                outputs={"command": decision.command},
                metadata={"attempt": attempt},
                error="invalid_command",
            )

        raise RuntimeError(
            f"Provider failed to produce a valid command after {self.config.provider_attempts} attempts: "
            f"{None if last_decision is None else last_decision.command}"
        )

    def _check_repeated_state(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        signature = _state_signature(state)
        if signature == self._last_state_signature:
            self._identical_state_count += 1
        else:
            self._last_state_signature = signature
            self._identical_state_count = 1

        if self._identical_state_count < self.config.max_identical_states:
            return None

        return {
            "type": "error",
            "message": (
                f"Identical state repeated {self._identical_state_count} times at "
                f"decision={state.get('decision', state.get('type'))}; stopping to avoid a stuck loop."
            ),
        }

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

    def _print_combat_log_path(self) -> None:
        path = self.combat_log.current_or_last_path()
        if path is not None:
            self._print(f"[combat_log] {path}")

    def _build_rl_transition(
        self,
        run_id: str,
        step: int,
        state: Dict[str, Any],
        command: Dict[str, Any],
        response: Optional[Dict[str, Any]],
        decision: ProviderDecision,
        agent_context: Dict[str, Any],
    ) -> RLTransition:
        reward = self.reward_model.evaluate(state, command, response, agent_context)
        return RLTransition(
            ts=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
            step=step,
            provider=decision.provider_name,
            character=self.config.character,
            decision=str(state.get("decision", state.get("type", ""))),
            action=str(command.get("action", command.get("cmd", ""))),
            action_key=command_key(command),
            command=command,
            available_actions=available_actions_from_state(state),
            action_hints=action_hints_from_state(state),
            chosen_action_features=summarize_action_for_rl(state, command),
            state=state,
            next_state=response,
            state_features=summarize_state_for_rl(state, agent_context),
            next_state_features=summarize_state_for_rl(response or {}, {}),
            reward=reward.reward,
            reward_breakdown=reward.breakdown.to_dict(),
            done=reward.done,
            terminal_type=reward.terminal_type,
            rationale=decision.rationale,
            memory_note=decision.memory_note,
            decision_steps=list(decision.decision_steps),
            agent_context=agent_context,
            metadata={
                "seed": self.config.seed,
                "max_steps": self.config.max_steps,
            },
        )
