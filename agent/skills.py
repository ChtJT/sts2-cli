#!/usr/bin/env python3
"""Explicit skill selection layer for STS2 agent decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


def _extract_name(obj: Any) -> str:
    if isinstance(obj, dict):
        if obj.get("en"):
            return str(obj["en"])
        if obj.get("zh"):
            return str(obj["zh"])
    return str(obj or "")


def _hp_ratio(facts: Dict[str, Any]) -> float:
    try:
        return float(facts.get("hp_ratio") or 1.0)
    except (TypeError, ValueError):
        return 1.0


def _incoming_damage(state: Dict[str, Any]) -> int:
    total = 0
    for enemy in state.get("enemies", []) or []:
        for intent in enemy.get("intents", []) or []:
            try:
                damage = int(intent.get("damage") or 0)
                hits = int(intent.get("hits") or 1)
            except (TypeError, ValueError):
                continue
            total += max(0, damage) * max(1, hits)
    return total


@dataclass
class SkillSpec:
    name: str
    description: str
    trigger: str
    priorities: List[str]
    constraints: List[str]
    success_metric: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "trigger": self.trigger,
            "priorities": self.priorities,
            "constraints": self.constraints,
            "success_metric": self.success_metric,
            "confidence": round(self.confidence, 3),
        }


class SkillRegistry:
    """Select explicit skills to shape the next action."""

    def __init__(self, character: str) -> None:
        self.character = character

    def select(
        self,
        state: Dict[str, Any],
        memory: Dict[str, Any],
        world_model: Optional[Dict[str, Any]] = None,
        safety: Optional[Dict[str, Any]] = None,
        episodic_hits: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        decision = state.get("decision", state.get("type", "unknown"))
        facts = memory.get("facts", {})
        deck_profile = memory.get("deck_profile", {})
        safety = safety or {}
        world_model = world_model or {}
        episodic_hits = list(episodic_hits or [])

        skills = self._decision_skills(decision, state, facts, deck_profile, world_model, safety, episodic_hits)
        skills.sort(key=lambda item: item.confidence, reverse=True)
        primary = skills[0].to_dict() if skills else None
        return {
            "decision": decision,
            "primary_skill": primary,
            "active_skills": [skill.to_dict() for skill in skills[:4]],
            "selection_notes": self._selection_notes(decision, skills, safety, episodic_hits),
        }

    def _decision_skills(
        self,
        decision: str,
        state: Dict[str, Any],
        facts: Dict[str, Any],
        deck_profile: Dict[str, Any],
        world_model: Dict[str, Any],
        safety: Dict[str, Any],
        episodic_hits: List[Dict[str, Any]],
    ) -> List[SkillSpec]:
        if decision == "combat_play":
            return self._combat_skills(state, facts, deck_profile, safety, episodic_hits)
        if decision == "map_select":
            return self._map_skills(facts, deck_profile, world_model, safety)
        if decision == "rest_site":
            return self._rest_site_skills(memory=deck_profile, facts=facts, state=state)
        if decision == "shop":
            return self._shop_skills(state, facts, deck_profile)
        if decision == "card_reward":
            return self._reward_skills(state, facts, deck_profile)
        if decision == "event_choice":
            return self._event_skills(facts, safety)
        if decision == "bundle_select":
            return self._bundle_skills()
        if decision == "card_select":
            return self._card_select_skills(state, deck_profile)
        return [
            SkillSpec(
                name="generic_legal_progression",
                description="Prefer the safest legal action that advances the run.",
                trigger=f"Fallback for decision {decision}.",
                priorities=["Advance the game state cleanly.", "Avoid unnecessary risk."],
                constraints=["Only use legal actions present in the current state."],
                success_metric="State advances without retries or invalid commands.",
                confidence=0.45,
            )
        ]

    def _combat_skills(
        self,
        state: Dict[str, Any],
        facts: Dict[str, Any],
        deck_profile: Dict[str, Any],
        safety: Dict[str, Any],
        episodic_hits: List[Dict[str, Any]],
    ) -> List[SkillSpec]:
        skills: List[SkillSpec] = []
        hp_ratio = _hp_ratio(facts)
        incoming = _incoming_damage(state)
        enemies = state.get("enemies", []) or []
        hand = state.get("hand", []) or []
        playable_attacks = [card for card in hand if card.get("can_play") and card.get("type") == "Attack"]
        playable_skills = [card for card in hand if card.get("can_play") and card.get("type") == "Skill"]
        lowest_hp = min((int(enemy.get("hp") or 9999) for enemy in enemies), default=9999)

        if incoming > 0 and (hp_ratio <= 0.45 or incoming >= int(facts.get("hp") or 0) * 0.45):
            skills.append(
                SkillSpec(
                    name="combat_survive_turn",
                    description="Bias the turn toward block, potion usage, or reducing immediate incoming damage.",
                    trigger=f"Incoming damage is about {incoming} while HP ratio is {hp_ratio:.2f}.",
                    priorities=[
                        "Survive the enemy turn.",
                        "Spend resources that materially reduce lethal or near-lethal damage.",
                    ],
                    constraints=["Do not choose a greedy line that ignores clearly dangerous incoming damage."],
                    success_metric="HP remains stable enough to continue the combat.",
                    confidence=0.94,
                )
            )

        if lowest_hp <= 10 and playable_attacks:
            skills.append(
                SkillSpec(
                    name="combat_finish_low_hp_target",
                    description="Focus damage on the most fragile enemy to reduce future incoming pressure.",
                    trigger=f"An enemy is at {lowest_hp} HP with attack cards still playable.",
                    priorities=[
                        "Remove one enemy from the board if possible.",
                        "Prefer lines that reduce the number of enemy actions next turn.",
                    ],
                    constraints=["Do not spread damage if a clean kill is available."],
                    success_metric="Combat board state shrinks this turn or next.",
                    confidence=0.88,
                )
            )

        if incoming == 0 and playable_attacks:
            skills.append(
                SkillSpec(
                    name="combat_frontload_damage",
                    description="Convert the turn into maximum efficient damage when defense is unnecessary.",
                    trigger="Enemies are not threatening immediate damage.",
                    priorities=[
                        "Spend energy on efficient attacks first.",
                        "Push tempo while defense is low-value.",
                    ],
                    constraints=["Avoid low-impact setup if direct damage is available."],
                    success_metric="Enemy HP drops quickly without wasting energy on unnecessary block.",
                    confidence=0.76,
                )
            )

        if int(deck_profile.get("strength_sources", 0)) >= 1 and playable_skills:
            skills.append(
                SkillSpec(
                    name="combat_scaling_setup",
                    description="Use the turn to establish scaling when survival is already acceptable.",
                    trigger="Deck contains scaling sources and the hand can support setup.",
                    priorities=[
                        "Invest in durable combat scaling if it does not lose tempo badly.",
                        "Balance setup against immediate lethal pressure.",
                    ],
                    constraints=["Do not set up scaling if the current turn is unsafe."],
                    success_metric="Future turns gain stronger damage or defense output.",
                    confidence=0.63,
                )
            )

        if not skills:
            skills.append(
                SkillSpec(
                    name="combat_safe_default",
                    description="Play efficient legal cards and avoid wasting energy.",
                    trigger="No sharper combat pattern stands out.",
                    priorities=[
                        "Use playable cards efficiently.",
                        "Reduce enemy HP while respecting immediate danger.",
                    ],
                    constraints=["Never issue an illegal card or target index."],
                    success_metric="Turn advances with no invalid commands and useful resource conversion.",
                    confidence=0.55,
                )
            )

        if episodic_hits:
            top = episodic_hits[0]
            skills.append(
                SkillSpec(
                    name="combat_episode_reference",
                    description="Use a similar prior combat as a light reference, not a hard script.",
                    trigger=f"Found similar past combat at step {top.get('step')} in run {top.get('run_id')}.",
                    priorities=[
                        "Reuse successful patterns from similar enemy boards.",
                        "Prefer lessons from non-error prior steps.",
                    ],
                    constraints=["Do not copy a past line if the current hand or HP state differs materially."],
                    success_metric="Past experience sharpens target choice or tempo.",
                    confidence=min(0.7, float(top.get("score") or 0) / 10.0 + 0.25),
                )
            )

        return skills

    def _map_skills(
        self,
        facts: Dict[str, Any],
        deck_profile: Dict[str, Any],
        world_model: Dict[str, Any],
        safety: Dict[str, Any],
    ) -> List[SkillSpec]:
        mode = str(world_model.get("strategy_mode") or "balanced")
        hp_ratio = _hp_ratio(facts)
        gold = int(facts.get("gold") or 0)
        skills: List[SkillSpec] = []

        if mode == "survival":
            skills.append(
                SkillSpec(
                    name="route_survival",
                    description="Prefer rooms that stabilize HP and reduce run failure risk.",
                    trigger=f"World model is in survival mode at HP ratio {hp_ratio:.2f}.",
                    priorities=[
                        "Route toward rest sites, shops, or safer branches.",
                        "Delay optional elites unless there is no better branch.",
                    ],
                    constraints=["Do not route into optional elites at dangerously low HP."],
                    success_metric="Reach the next major checkpoint with stable HP.",
                    confidence=0.95,
                )
            )
        elif mode == "elite_hunt":
            skills.append(
                SkillSpec(
                    name="route_elite_hunt",
                    description="Use current strength to convert the map into relic and reward upside.",
                    trigger="World model sees enough deck strength and HP to take greedier fights.",
                    priorities=[
                        "Take elites when downstream pathing remains healthy.",
                        "Preserve a later rest or bailout option.",
                    ],
                    constraints=["Do not greed elites if safety flags indicate HP instability."],
                    success_metric="Gain high-value rewards without collapsing the run.",
                    confidence=0.88,
                )
            )
        else:
            skills.append(
                SkillSpec(
                    name="route_balanced_progression",
                    description="Maintain a route with a healthy mix of fights, upgrades, and optional shops.",
                    trigger="No extreme routing mode is active.",
                    priorities=[
                        "Keep future branches flexible.",
                        "Prefer paths with sensible downstream room mixes.",
                    ],
                    constraints=["Avoid pathing that creates unnecessary dead ends."],
                    success_metric="Route remains adaptable and efficient.",
                    confidence=0.62,
                )
            )

        if gold >= 120 and int(deck_profile.get("starter_cards", 0)) >= 4:
            skills.append(
                SkillSpec(
                    name="route_shop_window",
                    description="Open a near-term shop window when removal or a premium purchase is plausible.",
                    trigger=f"Gold is {gold} and the deck still contains many starter cards.",
                    priorities=[
                        "Route into a shop when it does not severely compromise safety.",
                        "Value removal or a premium buy over marginal extra combats.",
                    ],
                    constraints=["Do not force a shop if the path becomes far more dangerous."],
                    success_metric="Gold converts into deck quality at a meaningful timing window.",
                    confidence=0.71,
                )
            )

        return skills

    def _rest_site_skills(
        self,
        memory: Dict[str, Any],
        facts: Dict[str, Any],
        state: Dict[str, Any],
    ) -> List[SkillSpec]:
        available_ids = [str(option.get("option_id") or "").upper() for option in state.get("options", [])]
        hp_ratio = _hp_ratio(facts)
        smith_candidates = memory.get("smith_candidates", []) or []
        skills: List[SkillSpec] = []

        if hp_ratio <= 0.45 and "HEAL" in available_ids:
            skills.append(
                SkillSpec(
                    name="rest_heal_stabilize",
                    description="Use the rest site to stabilize HP rather than greed tempo.",
                    trigger=f"HP ratio is {hp_ratio:.2f} and HEAL is available.",
                    priorities=[
                        "Heal when HP is fragile.",
                        "Preserve enough life to survive the next combat sequence.",
                    ],
                    constraints=["Do not spend the rest site on smithing at critical HP."],
                    success_metric="Post-rest HP is comfortably above danger range.",
                    confidence=0.97,
                )
            )

        if "SMITH" in available_ids and smith_candidates and hp_ratio >= 0.55:
            skills.append(
                SkillSpec(
                    name="rest_smith_tempo",
                    description="Convert the rest site into a strong immediate upgrade.",
                    trigger=f"HP ratio is {hp_ratio:.2f} with an upgrade target available.",
                    priorities=[
                        f"Upgrade {smith_candidates[0].get('name')} if it remains the top target.",
                        "Take the greedier line only when HP is stable.",
                    ],
                    constraints=["Do not smith if HP is close to collapse."],
                    success_metric="Deck gains a high-leverage upgrade without destabilizing survival.",
                    confidence=0.84,
                )
            )

        return skills or [
            SkillSpec(
                name="rest_default_resolution",
                description="Choose the most broadly useful rest-site option.",
                trigger="No strong heal or smith bias was detected.",
                priorities=["Prefer the option with the clearest deck or survival value."],
                constraints=["Stay within enabled rest-site options."],
                success_metric="Rest site produces a concrete improvement.",
                confidence=0.52,
            )
        ]

    def _shop_skills(self, state: Dict[str, Any], facts: Dict[str, Any], deck_profile: Dict[str, Any]) -> List[SkillSpec]:
        gold = int(facts.get("gold") or 0)
        removal_cost = state.get("card_removal_cost")
        starter_cards = int(deck_profile.get("starter_cards", 0))
        skills: List[SkillSpec] = []

        if removal_cost is not None and gold >= int(removal_cost) and starter_cards >= 5:
            skills.append(
                SkillSpec(
                    name="shop_remove_starters",
                    description="Treat card removal as a top-tier shop action while the deck is still bloated.",
                    trigger=f"Removal is affordable at {gold} gold with {starter_cards} starter cards left.",
                    priorities=[
                        "Remove Strike first, then Defend, unless a clearly stronger premium buy exists.",
                        "Use removal to sharpen future draws.",
                    ],
                    constraints=["Do not skip removal for a low-impact purchase."],
                    success_metric="Deck quality improves immediately through thinning.",
                    confidence=0.91,
                )
            )

        stocked_relics = [relic for relic in state.get("relics", []) if relic.get("is_stocked")]
        stocked_cards = [card for card in state.get("cards", []) if card.get("is_stocked")]
        if stocked_relics or stocked_cards:
            skills.append(
                SkillSpec(
                    name="shop_premium_buy",
                    description="Spend gold only on clearly premium or discounted value.",
                    trigger="The shop contains stocked purchases that may outclass a pass.",
                    priorities=[
                        "Prefer premium relics or discounted, high-impact cards.",
                        "Keep enough discipline to leave if all buys are marginal.",
                    ],
                    constraints=["Do not buy filler that dilutes the deck or drains gold pointlessly."],
                    success_metric="Gold converts into a strong long-term gain.",
                    confidence=0.67,
                )
            )

        skills.append(
            SkillSpec(
                name="shop_leave_disciplined",
                description="Leave when the shop does not offer removal or a clearly strong purchase.",
                trigger="Every shop needs an explicit leave discipline.",
                priorities=[
                    "Avoid spending gold on low-impact filler.",
                    "Preserve gold for a future premium window.",
                ],
                constraints=["Do not force a purchase merely because gold is available."],
                success_metric="Gold is spent only when the shop meaningfully improves the run.",
                confidence=0.58,
            )
        )
        return skills

    def _reward_skills(self, state: Dict[str, Any], facts: Dict[str, Any], deck_profile: Dict[str, Any]) -> List[SkillSpec]:
        deck_size = int(deck_profile.get("deck_size", 0))
        starter_cards = int(deck_profile.get("starter_cards", 0))
        skills = [
            SkillSpec(
                name="reward_take_efficient_pick",
                description="Prefer the reward that immediately improves consistency, tempo, or scaling.",
                trigger="Card reward is available.",
                priorities=[
                    "Favor efficient, generally useful cards over narrow build-arounds.",
                    "Prefer cards that solve current deck weaknesses.",
                ],
                constraints=["Do not add low-impact filler just because a reward is offered."],
                success_metric="Reward choice improves the next several combats.",
                confidence=0.75,
            )
        ]
        if deck_size >= 18 or starter_cards >= 6:
            skills.append(
                SkillSpec(
                    name="reward_skip_dilution",
                    description="Skipping is acceptable when all offered cards dilute the deck.",
                    trigger=f"Deck size is already {deck_size} with {starter_cards} starter cards remaining.",
                    priorities=[
                        "Preserve draw quality when offers are weak.",
                        "Skip low-impact cards confidently.",
                    ],
                    constraints=["Do not force a pick that worsens consistency."],
                    success_metric="Deck quality stays high rather than drifting downward.",
                    confidence=0.69,
                )
            )
        return skills

    def _event_skills(self, facts: Dict[str, Any], safety: Dict[str, Any]) -> List[SkillSpec]:
        hp_ratio = _hp_ratio(facts)
        return [
            SkillSpec(
                name="event_low_risk" if hp_ratio <= 0.55 else "event_value_seek",
                description="Balance event upside against hidden or explicit downside.",
                trigger=f"Event resolution at HP ratio {hp_ratio:.2f}.",
                priorities=[
                    "Favor low-risk outcomes when HP or run stability is poor.",
                    "Take higher-upside options only when the downside is acceptable.",
                ],
                constraints=["Respect safety warnings about HP-loss or risky event lines."],
                success_metric="Event creates upside without destabilizing the run.",
                confidence=0.64,
            )
        ]

    def _bundle_skills(self) -> List[SkillSpec]:
        return [
            SkillSpec(
                name="bundle_value_maximizer",
                description="Choose the bundle that adds the strongest immediate and strategic value.",
                trigger="A bundle selection screen is active.",
                priorities=[
                    "Prefer bundles with the best net value for the current deck and route.",
                    "Consider both immediate payoff and future flexibility.",
                ],
                constraints=["Do not anchor on one card; evaluate the whole bundle."],
                success_metric="Bundle materially strengthens the run.",
                confidence=0.73,
            )
        ]

    def _card_select_skills(self, state: Dict[str, Any], deck_profile: Dict[str, Any]) -> List[SkillSpec]:
        min_select = int(state.get("min_select", 1))
        max_select = int(state.get("max_select", 1))
        return [
            SkillSpec(
                name="card_select_precision",
                description="Make the exact number of card selections that best preserves deck quality.",
                trigger=f"Selection requires between {min_select} and {max_select} cards.",
                priorities=[
                    "Select the highest-leverage cards for the screen's purpose.",
                    "Trim starter cards first when the screen represents removal.",
                ],
                constraints=["Respect the exact selection count and available indices."],
                success_metric="Selection advances the screen objective with minimal deck damage.",
                confidence=0.71,
            )
        ]

    def _selection_notes(
        self,
        decision: str,
        skills: List[SkillSpec],
        safety: Dict[str, Any],
        episodic_hits: List[Dict[str, Any]],
    ) -> List[str]:
        notes: List[str] = []
        if skills:
            notes.append(f"Primary skill for {decision}: {skills[0].name}.")
        warnings = safety.get("warnings") or []
        if warnings:
            notes.append(str(warnings[0]))
        if episodic_hits:
            top = episodic_hits[0]
            notes.append(f"Similar prior episode found in run {top.get('run_id')} step {top.get('step')}.")
        return notes[:3]
