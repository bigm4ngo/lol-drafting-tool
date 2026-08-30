"""Shared v3 feature definitions and team aggregation helpers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROLES = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")
CHAMPION_FEATURES = (
    "physical_share",
    "magic_share",
    "true_share",
    "control",
    "hard_cc",
    "engage",
    "pick_potential",
    "waveclear",
    "objective",
    "mobility",
    "frontline",
    "vision",
    "early_strength",
    "mid_strength",
    "late_strength",
)
# The live model receives ally-minus-enemy values.  Damage shares use team sums
# normalized to one; capability features use team means.
MODEL_FEATURE_NAMES = tuple(f"diff_{name}" for name in CHAMPION_FEATURES) + (
    "ally_damage_balance",
    "enemy_damage_balance",
    "ally_feature_coverage",
    "enemy_feature_coverage",
)


def neutral_feature_row() -> dict[str, float]:
    output = {name: 0.5 for name in CHAMPION_FEATURES}
    output.update({"physical_share": 0.5, "magic_share": 0.5, "true_share": 0.0})
    output["feature_confidence"] = 0.0
    return output


def aggregate_team(
    champion_ids: Iterable[int],
    role_by_champion: Mapping[int, str] | None,
    features: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, float]:
    rows: list[Mapping[str, Any]] = []
    confidences: list[float] = []
    for champion_id in champion_ids:
        role = (role_by_champion or {}).get(int(champion_id), "")
        row = features.get((role, int(champion_id)))
        if row is None:
            # Fall back to the champion's best-covered role if the current role
            # is unknown or the exact sample does not yet exist.
            candidates = [
                value for (candidate_role, candidate_id), value in features.items()
                if candidate_id == int(champion_id)
            ]
            row = max(candidates, key=lambda value: float(value.get("games", 0)), default=None)
        if row is not None:
            rows.append(row)
            confidences.append(float(row.get("feature_confidence", 0.0)))
    if not rows:
        return neutral_feature_row()

    output: dict[str, float] = {}
    for name in CHAMPION_FEATURES:
        values = [float(row.get(name, neutral_feature_row()[name])) for row in rows]
        output[name] = float(np.mean(values))

    damage_total = output["physical_share"] + output["magic_share"] + output["true_share"]
    if damage_total > 0:
        output["physical_share"] /= damage_total
        output["magic_share"] /= damage_total
        output["true_share"] /= damage_total
    output["feature_confidence"] = float(np.mean(confidences)) if confidences else 0.0
    return output


def damage_balance(metrics: Mapping[str, float]) -> float:
    physical = float(metrics.get("physical_share", 0.5))
    magic = float(metrics.get("magic_share", 0.5))
    # One at a roughly balanced 50/50 profile, zero for a single-damage team.
    return max(0.0, min(1.0, 1.0 - abs(physical - magic)))


def model_feature_vector(
    ally: Mapping[str, float], enemy: Mapping[str, float]
) -> dict[str, float]:
    output = {
        f"diff_{name}": float(ally.get(name, 0.5)) - float(enemy.get(name, 0.5))
        for name in CHAMPION_FEATURES
    }
    output.update({
        "ally_damage_balance": damage_balance(ally),
        "enemy_damage_balance": damage_balance(enemy),
        "ally_feature_coverage": float(ally.get("feature_confidence", 0.0)),
        "enemy_feature_coverage": float(enemy.get("feature_confidence", 0.0)),
    })
    return output
