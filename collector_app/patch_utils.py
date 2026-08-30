"""Patch parsing and recency weighting for League match analytics.

Riot Match-V5 stores versions such as ``16.14.623.1234`` while Data Dragon
usually exposes ``16.14.1``.  Draft Lab uses the major/minor pair as the patch
identity, preserves older games, and exponentially reduces their influence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

_PATCH_RE = re.compile(r"(?P<major>\d+)\.(?P<minor>\d+)")


@dataclass(frozen=True, order=True, slots=True)
class Patch:
    major: int
    minor: int

    @classmethod
    def parse(cls, value: Any) -> "Patch | None":
        match = _PATCH_RE.search(str(value or ""))
        if not match:
            return None
        return cls(int(match.group("major")), int(match.group("minor")))

    @property
    def label(self) -> str:
        return f"{self.major}.{self.minor}"

    @property
    def ordinal(self) -> int:
        # League generally has fewer than 100 numbered patches in a season.
        return self.major * 100 + self.minor


def patch_distance(game_version: Any, current_version: Any) -> int:
    game = Patch.parse(game_version)
    current = Patch.parse(current_version)
    if game is None or current is None:
        return 0
    return max(0, current.ordinal - game.ordinal)


def recency_weight(
    game_version: Any,
    current_version: Any,
    *,
    half_life_patches: float = 3.0,
    minimum_weight: float = 0.12,
) -> float:
    """Return an exponential patch weight without ever discarding old games."""
    distance = patch_distance(game_version, current_version)
    half_life = max(0.25, float(half_life_patches))
    weight = math.exp(-math.log(2.0) * distance / half_life)
    return max(float(minimum_weight), min(1.0, weight))


def patch_label(value: Any) -> str:
    parsed = Patch.parse(value)
    return parsed.label if parsed else "unknown"
