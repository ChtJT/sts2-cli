#!/usr/bin/env python3
"""Provider abstraction for STS2 agents."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List
from urllib import error, request


def _command(action: str, args: Dict[str, Any] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"cmd": "action", "action": action}
    if args:
        payload["args"] = args
    return payload


def _name(obj: Any) -> str:
    if isinstance(obj, dict):
        if "en" in obj and obj["en"]:
            return str(obj["en"])
        if "zh" in obj and obj["zh"]:
            return str(obj["zh"])
    return str(obj)


@dataclass
class ProviderDecision:
    command: Dict[str, Any]
    rationale: str
    memory_note: str = ""
    decision_steps: List[str] = field(default_factory=list)
    provider_name: str = "openai"


class AgentProvider:
    name = "base"

    def decide(self, payload: Dict[str, Any]) -> ProviderDecision:
        raise NotImplementedError


class OpenAIProvider(AgentProvider):
    name = "openai"

    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required when --provider openai is used.")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.timeout = int(os.environ.get("OPENAI_TIMEOUT_SECS", "60"))
        self.max_output_tokens = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "1600"))
        self.reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT", "minimal").strip().lower()
        self.max_api_retries = int(os.environ.get("OPENAI_MAX_API_RETRIES", "3"))

    def decide(self, payload: Dict[str, Any]) -> ProviderDecision:
        state = payload["state"]
        response = self._request_decision_response(payload)
        parsed = self._parse_response_json(response)
        args = parsed.get("args")
        if not isinstance(args, dict):
            args = {}
        args = {key: value for key, value in args.items() if value is not None}
        steps = parsed.get("decision_steps")
        if not isinstance(steps, list):
            steps = []
        steps = [str(item) for item in steps if str(item).strip()]
        return ProviderDecision(
            command=_command(parsed["action"], args),
            rationale=parsed.get("rationale", "OpenAI selected an action."),
            memory_note=parsed.get("memory_note", ""),
            decision_steps=steps,
            provider_name=self.name,
        )

    def _request_decision_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = payload["state"]
        current_max_output_tokens = self.max_output_tokens
        current_reasoning_effort = self.reasoning_effort

        for attempt in range(1, self.max_api_retries + 1):
            body: Dict[str, Any] = {
                "model": self.model,
                "input": [
                    {
                        "role": "developer",
                        "content": [
                            {
                                "type": "input_text",
                                "text": self._developer_prompt(),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": self._user_prompt(payload),
                            }
                        ],
                    },
                ],
                "max_output_tokens": current_max_output_tokens,
                "store": False,
                "parallel_tool_calls": False,
                "text": {"verbosity": "low"},
                "tools": [self._decision_tool(state)],
                "tool_choice": {"type": "function", "name": "submit_sts2_action"},
            }
            if current_reasoning_effort:
                body["reasoning"] = {"effort": current_reasoning_effort}

            response = self._responses_create(body)
            if self._has_function_call(response):
                return response

            incomplete_reason = ((response.get("incomplete_details") or {}).get("reason"))
            if incomplete_reason != "max_output_tokens":
                return response

            current_max_output_tokens *= 2
            current_reasoning_effort = "minimal"

        return response

    def _responses_create(self, body: Dict[str, Any]) -> Dict[str, Any]:
        payload = json.dumps(body).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/responses",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI API error {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Failed to reach OpenAI API: {exc.reason}") from exc

    def _parse_response_json(self, response: Dict[str, Any]) -> Dict[str, Any]:
        if response.get("error") is not None:
            raise RuntimeError(f"OpenAI API returned error: {response['error']}")

        if not self._has_function_call(response):
            incomplete_reason = ((response.get("incomplete_details") or {}).get("reason"))
            status = response.get("status")
            raise RuntimeError(
                f"OpenAI response did not include submit_sts2_action. "
                f"status={status}, incomplete_reason={incomplete_reason}, response={response}"
            )

        for item in response.get("output", []):
            if item.get("type") not in {"function_call", "tool_call"}:
                continue
            if item.get("name") != "submit_sts2_action":
                continue
            arguments = item.get("arguments") or ""
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"OpenAI tool arguments were not valid JSON: {arguments}") from exc
            if not isinstance(parsed, dict) or "action" not in parsed:
                raise RuntimeError(f"OpenAI tool call missing action field: {parsed}")
            return parsed

        raise RuntimeError(f"OpenAI response did not include submit_sts2_action: {response}")

    def _has_function_call(self, response: Dict[str, Any]) -> bool:
        for item in response.get("output", []):
            if item.get("type") in {"function_call", "tool_call"} and item.get("name") == "submit_sts2_action":
                return True
        return False

    def _developer_prompt(self) -> str:
        return (
            "You are controlling Slay the Spire 2 through a strict action interface. "
            "Think step by step internally before acting. "
            "Return exactly one function call named submit_sts2_action. "
            "The function arguments must contain: the chosen legal action, action args, "
            "2-5 concise public decision_steps, a short rationale, and a short memory_note. "
            "Do not invent fields, indices, or unsupported actions."
        )

    def _user_prompt(self, payload: Dict[str, Any]) -> str:
        state = payload["state"]
        prompt = {
            "task": "Return the next legal action for the current Slay the Spire 2 state.",
            "attempt": payload.get("attempt", 1),
            "decision": state.get("decision", state.get("type")),
            "allowed_actions": self._allowed_actions(state),
            "action_hints": self._action_hints(state),
            "state": state,
            "memory": payload.get("memory", {}),
            "retrieval": payload.get("retrieval", []),
            "provider_feedback": payload.get("provider_feedback"),
        }
        return json.dumps(prompt, ensure_ascii=False, indent=2)

    def _decision_tool(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": "submit_sts2_action",
            "description": "Submit the next legal STS2 action plus concise public reasoning.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["action", "args", "decision_steps", "rationale", "memory_note"],
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": self._allowed_actions(state),
                    },
                    "args": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "col",
                            "row",
                            "card_index",
                            "target_index",
                            "potion_index",
                            "bundle_index",
                            "indices",
                            "option_index",
                            "relic_index",
                        ],
                        "properties": {
                            "col": {"type": ["integer", "null"]},
                            "row": {"type": ["integer", "null"]},
                            "card_index": {"type": ["integer", "null"]},
                            "target_index": {"type": ["integer", "null"]},
                            "potion_index": {"type": ["integer", "null"]},
                            "bundle_index": {"type": ["integer", "null"]},
                            "indices": {"type": ["string", "null"]},
                            "option_index": {"type": ["integer", "null"]},
                            "relic_index": {"type": ["integer", "null"]},
                        },
                    },
                    "decision_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 5,
                        "description": "2-5 concise public reasoning steps, not hidden chain-of-thought.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One short explanation for the chosen action.",
                    },
                    "memory_note": {
                        "type": "string",
                        "description": "A concise note worth storing in agent memory.",
                    },
                },
            },
        }

    def _allowed_actions(self, state: Dict[str, Any]) -> List[str]:
        decision = state.get("decision", "")
        if decision == "map_select":
            return ["select_map_node"]
        if decision == "combat_play":
            actions = ["play_card", "end_turn"]
            if any(pot for pot in state.get("player", {}).get("potions", [])):
                actions.append("use_potion")
            return actions
        if decision == "card_reward":
            return ["select_card_reward", "skip_card_reward"]
        if decision == "bundle_select":
            return ["select_bundle"]
        if decision == "card_select":
            actions = ["select_cards"]
            if int(state.get("min_select", 1)) == 0:
                actions.append("skip_select")
            return actions
        if decision == "rest_site":
            return ["choose_option"]
        if decision == "event_choice":
            actions = ["choose_option"]
            if state.get("options"):
                actions.append("leave_room")
            return actions
        if decision == "shop":
            return ["buy_card", "buy_relic", "buy_potion", "remove_card", "leave_room"]
        return ["proceed"]

    def _action_hints(self, state: Dict[str, Any]) -> Dict[str, Any]:
        decision = state.get("decision", "")
        if decision == "map_select":
            return {"choices": state.get("choices", [])}
        if decision == "combat_play":
            return {
                "playable_cards": [
                    {
                        "index": card.get("index"),
                        "name": _name(card.get("name")),
                        "target_type": card.get("target_type"),
                    }
                    for card in state.get("hand", [])
                    if card.get("can_play")
                ],
                "enemies": [
                    {
                        "index": enemy.get("index"),
                        "name": _name(enemy.get("name")),
                        "hp": enemy.get("hp"),
                    }
                    for enemy in state.get("enemies", [])
                ],
                "potions": [
                    {
                        "index": potion.get("index"),
                        "name": _name(potion.get("name")),
                        "target_type": potion.get("target_type"),
                    }
                    for potion in state.get("player", {}).get("potions", [])
                    if potion
                ],
            }
        if decision == "card_reward":
            return {"cards": state.get("cards", [])}
        if decision == "bundle_select":
            return {"bundles": state.get("bundles", [])}
        if decision == "card_select":
            return {
                "cards": state.get("cards", []),
                "min_select": state.get("min_select"),
                "max_select": state.get("max_select"),
            }
        if decision in {"rest_site", "event_choice"}:
            return {"options": state.get("options", [])}
        if decision == "shop":
            return {
                "cards": state.get("cards", []),
                "relics": state.get("relics", []),
                "potions": state.get("potions", []),
                "card_removal_cost": state.get("card_removal_cost"),
                "gold": state.get("player", {}).get("gold"),
            }
        return {}


def build_provider(name: str) -> AgentProvider:
    normalized = name.lower()
    if normalized == "openai":
        return OpenAIProvider()
    if normalized == "codex":
        raise ValueError("codex provider is not wired yet. Use --provider openai for now.")
    raise ValueError(f"Unsupported provider: {name}")
