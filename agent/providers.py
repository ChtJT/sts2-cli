#!/usr/bin/env python3
"""Provider abstraction for STS2 agents."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
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
        response = self._request_decision_response(payload)
        return self._parse_response_decision(response, payload["state"])

    def _request_decision_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = payload["state"]
        current_max_output_tokens = self.max_output_tokens
        current_reasoning_effort = self.reasoning_effort
        safe_mode = False

        for attempt in range(1, self.max_api_retries + 1):
            body: Dict[str, Any] = {
                "model": self.model,
                "input": [
                    {
                        "role": "developer",
                        "content": [
                            {
                                "type": "input_text",
                                "text": self._developer_prompt(safe_mode=safe_mode),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": self._user_prompt(payload, safe_mode=safe_mode),
                            }
                        ],
                    },
                ],
                "max_output_tokens": current_max_output_tokens,
                "store": False,
                "parallel_tool_calls": False,
                "text": {"verbosity": "low"},
                "tools": self._decision_tools(state),
                "tool_choice": "required",
            }
            if current_reasoning_effort and not safe_mode:
                body["reasoning"] = {"effort": current_reasoning_effort}

            try:
                response = self._responses_create(body)
            except RuntimeError as exc:
                if "invalid_prompt" in str(exc) and not safe_mode:
                    safe_mode = True
                    current_reasoning_effort = ""
                    continue
                raise
            if self._has_function_call(response, state):
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

    def _parse_response_decision(self, response: Dict[str, Any], state: Dict[str, Any]) -> ProviderDecision:
        if response.get("error") is not None:
            raise RuntimeError(f"OpenAI API returned error: {response['error']}")

        if not self._has_function_call(response, state):
            incomplete_reason = ((response.get("incomplete_details") or {}).get("reason"))
            status = response.get("status")
            raise RuntimeError(
                f"OpenAI response did not include a valid action tool call. "
                f"status={status}, incomplete_reason={incomplete_reason}, response={response}"
            )

        for item in response.get("output", []):
            if item.get("type") not in {"function_call", "tool_call"}:
                continue
            tool_name = item.get("name")
            if tool_name not in self._tool_names(state):
                continue
            arguments = item.get("arguments") or ""
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"OpenAI tool arguments were not valid JSON: {arguments}") from exc
            if not isinstance(parsed, dict):
                raise RuntimeError(f"OpenAI tool call arguments were not an object: {parsed}")
            return self._normalize_tool_call(tool_name, parsed)

        raise RuntimeError(f"OpenAI response did not include a valid action tool call: {response}")

    def _has_function_call(self, response: Dict[str, Any], state: Dict[str, Any]) -> bool:
        valid_tools = self._tool_names(state)
        for item in response.get("output", []):
            if item.get("type") in {"function_call", "tool_call"} and item.get("name") in valid_tools:
                return True
        return False

    def _developer_prompt(self, safe_mode: bool = False) -> str:
        if safe_mode:
            return (
                "You are controlling Slay the Spire 2 through a strict action interface. "
                "Return exactly one required function call. "
                "Each function already encodes one legal API shape for the current state. "
                "Use prompt_context as the complete strategic summary for this step. "
                "Respect safety.hard_rules and avoid choices that directly conflict with safety warnings when a safer legal line exists. "
                "Use the selected skill and episodic examples as guidance, not rigid scripts. "
                "Choose the correct function and fill only its defined fields. "
                "Include brief public decision notes, a short rationale, and a short memory note. "
                "Do not invent indices, unsupported fields, or unsupported actions."
            )
        return (
            "You are controlling Slay the Spire 2 through a strict action interface. "
            "Return exactly one required function call. "
            "Each function already encodes one legal API shape for the current state. "
            "Use prompt_context as the complete strategic summary for this step. "
            "Respect safety.hard_rules and avoid choices that directly conflict with safety warnings when a safer legal line exists. "
            "Choose the correct function and fill only its defined fields. "
            "Always include 2-5 concise public decision notes, a short rationale, and a short memory note. "
            "Do not invent indices, unsupported fields, or unsupported actions."
        )

    def _user_prompt(self, payload: Dict[str, Any], safe_mode: bool = False) -> str:
        state = payload["state"]
        prompt: Dict[str, Any] = {
            "task": "Return the next legal action for the current Slay the Spire 2 state.",
            "attempt": payload.get("attempt", 1),
            "decision": state.get("decision", state.get("type")),
            "allowed_actions": self._allowed_actions(state),
            "available_tools": self._tool_names(state),
            "action_hints": self._action_hints(state),
            "state": state,
            "prompt_context": payload.get("prompt_context", {}),
            "provider_feedback": payload.get("provider_feedback"),
        }
        prompt["strategy_focus"] = payload.get("prompt_context", {})
        if safe_mode:
            prompt["safe_mode"] = True
        return json.dumps(prompt, ensure_ascii=False, indent=2)

    def _tool_names(self, state: Dict[str, Any]) -> List[str]:
        return [tool["name"] for tool in self._decision_tools(state)]

    def _decision_tools(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        decision = state.get("decision", "")
        if decision == "map_select":
            return [
                self._tool(
                    "select_map_node_action",
                    "Select a map node by column and row.",
                    {
                        "col": {"type": "integer"},
                        "row": {"type": "integer"},
                    },
                    ["col", "row"],
                )
            ]
        if decision == "combat_play":
            tools = [
                self._tool(
                    "play_card_action",
                    "Play one card from hand. Use null target_index for non-targeted cards.",
                    {
                        "card_index": {"type": "integer"},
                        "target_index": {"type": ["integer", "null"]},
                    },
                    ["card_index", "target_index"],
                ),
                self._tool("end_turn_action", "End the current turn.", {}, []),
            ]
            if any(pot for pot in state.get("player", {}).get("potions", [])):
                tools.append(
                    self._tool(
                        "use_potion_action",
                        "Use a potion. Use null target_index for non-targeted potions.",
                        {
                            "potion_index": {"type": "integer"},
                            "target_index": {"type": ["integer", "null"]},
                        },
                        ["potion_index", "target_index"],
                    )
                )
            return tools
        if decision == "card_reward":
            return [
                self._tool(
                    "select_card_reward_action",
                    "Choose one card from the reward screen.",
                    {"card_index": {"type": "integer"}},
                    ["card_index"],
                ),
                self._tool("skip_card_reward_action", "Skip the card reward.", {}, []),
            ]
        if decision == "bundle_select":
            return [
                self._tool(
                    "select_bundle_action",
                    "Choose one bundle by bundle_index.",
                    {"bundle_index": {"type": "integer"}},
                    ["bundle_index"],
                )
            ]
        if decision == "card_select":
            tools = []
            min_select = int(state.get("min_select", 1))
            max_select = int(state.get("max_select", 1))
            if min_select == 0:
                tools.append(self._tool("skip_select_action", "Skip this optional card selection.", {}, []))
            if max_select == 1:
                tools.append(
                    self._tool(
                        "select_single_card_action",
                        "Choose exactly one card by card_index.",
                        {"card_index": {"type": "integer"}},
                        ["card_index"],
                    )
                )
            else:
                tools.append(
                    self._tool(
                        "select_cards_action",
                        "Choose one or more cards using a comma-separated indices string like '0,2'.",
                        {"indices": {"type": "string"}},
                        ["indices"],
                    )
                )
            return tools
        if decision == "rest_site":
            return [
                self._tool(
                    "choose_option_action",
                    "Choose a rest site option by option_index.",
                    {"option_index": {"type": "integer"}},
                    ["option_index"],
                )
            ]
        if decision == "event_choice":
            tools = [
                self._tool(
                    "choose_option_action",
                    "Choose an unlocked event option by option_index.",
                    {"option_index": {"type": "integer"}},
                    ["option_index"],
                )
            ]
            tools.append(self._tool("leave_room_action", "Leave the current room.", {}, []))
            return tools
        if decision == "shop":
            return [
                self._tool(
                    "buy_card_action",
                    "Buy a card by card_index.",
                    {"card_index": {"type": "integer"}},
                    ["card_index"],
                ),
                self._tool(
                    "buy_relic_action",
                    "Buy a relic by relic_index.",
                    {"relic_index": {"type": "integer"}},
                    ["relic_index"],
                ),
                self._tool(
                    "buy_potion_action",
                    "Buy a potion by potion_index.",
                    {"potion_index": {"type": "integer"}},
                    ["potion_index"],
                ),
                self._tool("remove_card_action", "Pay to remove a card from the deck.", {}, []),
                self._tool("leave_room_action", "Leave the current room.", {}, []),
            ]
        return [self._tool("proceed_action", "Proceed after a non-decision or transient error.", {}, [])]

    def _tool(self, name: str, description: str, arg_properties: Dict[str, Any], arg_required: List[str]) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": arg_required + ["decision_steps", "rationale", "memory_note"],
                "properties": {
                    **arg_properties,
                    "decision_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 5,
                        "description": "1-5 concise public decision notes.",
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

    def _normalize_tool_call(self, tool_name: str, parsed: Dict[str, Any]) -> ProviderDecision:
        decision_steps = parsed.get("decision_steps")
        if not isinstance(decision_steps, list):
            decision_steps = []
        decision_steps = [str(item).strip() for item in decision_steps if str(item).strip()]
        rationale = str(parsed.get("rationale") or "OpenAI selected an action.")
        memory_note = str(parsed.get("memory_note") or "")

        action_map = {
            "select_map_node_action": ("select_map_node", ["col", "row"]),
            "play_card_action": ("play_card", ["card_index", "target_index"]),
            "end_turn_action": ("end_turn", []),
            "use_potion_action": ("use_potion", ["potion_index", "target_index"]),
            "select_card_reward_action": ("select_card_reward", ["card_index"]),
            "skip_card_reward_action": ("skip_card_reward", []),
            "select_bundle_action": ("select_bundle", ["bundle_index"]),
            "select_cards_action": ("select_cards", ["indices"]),
            "skip_select_action": ("skip_select", []),
            "select_single_card_action": ("select_cards", ["card_index"]),
            "choose_option_action": ("choose_option", ["option_index"]),
            "leave_room_action": ("leave_room", []),
            "buy_card_action": ("buy_card", ["card_index"]),
            "buy_relic_action": ("buy_relic", ["relic_index"]),
            "buy_potion_action": ("buy_potion", ["potion_index"]),
            "remove_card_action": ("remove_card", []),
            "proceed_action": ("proceed", []),
        }
        if tool_name not in action_map:
            raise RuntimeError(f"Unknown tool call name: {tool_name}")

        action, keys = action_map[tool_name]
        args = {key: parsed.get(key) for key in keys if parsed.get(key) is not None}
        if tool_name == "select_single_card_action":
            args = {"indices": str(parsed["card_index"])}

        return ProviderDecision(
            command=_command(action, args),
            rationale=rationale,
            memory_note=memory_note,
            decision_steps=decision_steps,
            provider_name=self.name,
        )

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
