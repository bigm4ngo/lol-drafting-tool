"""Deterministic draft explanations for v3 recommendations and team summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class DraftExplanation:
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    summary: str


def _level(value: float) -> str:
    if value >= 0.72:
        return "very high"
    if value >= 0.58:
        return "high"
    if value >= 0.42:
        return "moderate"
    if value >= 0.28:
        return "low"
    return "very low"


def explain_team(metrics: Mapping[str, float]) -> DraftExplanation:
    strengths: list[str] = []
    weaknesses: list[str] = []

    physical = float(metrics.get("physical_share", 0.0))
    magic = float(metrics.get("magic_share", 0.0))
    true = float(metrics.get("true_share", 0.0))
    if max(physical, magic) >= 0.72:
        dominant = "physical" if physical > magic else "magic"
        weaknesses.append(f"Damage is heavily {dominant}, making defensive itemisation easier.")
    elif physical >= 0.30 and magic >= 0.30:
        strengths.append("Balanced physical and magic damage is difficult to itemise against.")
    if true >= 0.08:
        strengths.append("Meaningful true damage helps against high-resistance frontlines.")

    labels = {
        "control": "crowd control",
        "engage": "engage",
        "pick_potential": "pick potential",
        "waveclear": "wave clear",
        "objective": "objective control",
        "mobility": "mobility",
        "frontline": "frontline",
    }
    for key, label in labels.items():
        value = float(metrics.get(key, 0.5))
        if value >= 0.68:
            strengths.append(f"{label.title()} is {_level(value)}.")
        elif value <= 0.30:
            weaknesses.append(f"{label.title()} is {_level(value)}.")

    early = float(metrics.get("early_strength", 0.5))
    mid = float(metrics.get("mid_strength", 0.5))
    late = float(metrics.get("late_strength", 0.5))
    strongest = max(((early, "early"), (mid, "mid"), (late, "late")), key=lambda item: item[0])
    weakest = min(((early, "early"), (mid, "mid"), (late, "late")), key=lambda item: item[0])
    if strongest[0] - weakest[0] >= 0.16:
        strengths.append(f"The composition is strongest in the {strongest[1]} game.")
        weaknesses.append(f"Relative power is lowest in the {weakest[1]} game.")
    else:
        strengths.append("Power is reasonably stable from early through late game.")

    if not strengths:
        strengths.append("No single composition strength clearly dominates the current sample.")
    if not weaknesses:
        weaknesses.append("No major structural weakness is visible from the current picks.")
    summary = (
        f"Damage: {physical:.0%} physical, {magic:.0%} magic, {true:.0%} true. "
        f"Control {_level(float(metrics.get('control', 0.5)))}, "
        f"engage {_level(float(metrics.get('engage', 0.5)))}, "
        f"late scaling {_level(late)}."
    )
    return DraftExplanation(tuple(strengths[:5]), tuple(weaknesses[:5]), summary)


def explain_candidate(
    *,
    champion_name: str,
    metrics: Mapping[str, float],
    synergy_delta: float,
    counter_delta: float,
    ml_uplift: float,
    confidence: float,
    team_before: Mapping[str, float] | None = None,
) -> DraftExplanation:
    strengths: list[str] = []
    weaknesses: list[str] = []
    if synergy_delta >= 0.012:
        strengths.append(f"Strong observed synergy with the current allied picks ({synergy_delta:+.1%}).")
    elif synergy_delta <= -0.012:
        weaknesses.append(f"Historical synergy with the current allies is below baseline ({synergy_delta:+.1%}).")
    if counter_delta >= 0.012:
        strengths.append(f"Favourable observed lane matchup contribution ({counter_delta:+.1%}).")
    elif counter_delta <= -0.012:
        weaknesses.append(f"The visible lane matchup is historically difficult ({counter_delta:+.1%}).")
    if ml_uplift >= 0.012:
        strengths.append(f"The neural ensemble raises the draft win estimate by {ml_uplift:+.1%}.")
    elif ml_uplift <= -0.012:
        weaknesses.append(f"The neural ensemble lowers the draft win estimate by {ml_uplift:+.1%}.")

    labels = {
        "control": "crowd control",
        "engage": "engage",
        "pick_potential": "pick potential",
        "waveclear": "wave clear",
        "objective": "objective control",
        "mobility": "mobility",
        "frontline": "frontline",
        "late_strength": "late-game scaling",
    }
    before = team_before or {}
    for key, label in labels.items():
        value = float(metrics.get(key, 0.5))
        previous = float(before.get(key, 0.5))
        if value - previous >= 0.07:
            strengths.append(f"Adds meaningful {label} to the composition.")
        elif value <= 0.27 and previous <= 0.32:
            weaknesses.append(f"Does not solve the team's low {label}.")

    if confidence < 40:
        weaknesses.append("Confidence is low because the matchup, patch, or role sample is limited.")
    elif confidence >= 75:
        strengths.append("The recommendation is supported by strong sample coverage and model agreement.")

    if not strengths:
        strengths.append("Provides a statistically competitive all-round fit for the open role.")
    if not weaknesses:
        weaknesses.append("No major draft-specific weakness is identified, but normal execution risk remains.")
    return DraftExplanation(
        strengths=tuple(dict.fromkeys(strengths))[:5],
        weaknesses=tuple(dict.fromkeys(weaknesses))[:5],
        summary=f"{champion_name}: recommendation confidence {confidence:.0f}%.",
    )
