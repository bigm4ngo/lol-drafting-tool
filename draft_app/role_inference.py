"""Whole-draft probabilistic role inference for flex picks.

Instead of assigning each champion independently, this module enumerates valid
team-wide role assignments.  A flex pick can therefore move only when another
champion can occupy the vacated role, which is substantially more reliable than
choosing every champion's most common role in isolation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

ROLES = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")


@dataclass(frozen=True, slots=True)
class RoleGuess:
    champion_id: int
    role: str
    confidence: float
    alternatives: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class TeamRoleInference:
    guesses: tuple[RoleGuess, ...]
    assignment_confidence: float
    alternatives: tuple[tuple[tuple[tuple[int, str], ...], float], ...]

    def role_for(self, champion_id: int) -> str | None:
        return next((guess.role for guess in self.guesses if guess.champion_id == champion_id), None)


class RoleInferenceEngine:
    def __init__(self, priors: Mapping[tuple[int, str], float] | None = None) -> None:
        self.priors = dict(priors or {})

    def set_priors(self, priors: Mapping[tuple[int, str], float]) -> None:
        self.priors = dict(priors)

    def probability(self, champion_id: int, role: str) -> float:
        # A small floor lets rare/off-meta choices remain possible without
        # allowing zero-count roles to dominate a flex assignment.
        return max(0.002, float(self.priors.get((int(champion_id), str(role)), 0.002)))

    @staticmethod
    def _logsumexp(values: Sequence[float]) -> float:
        if not values:
            return float("-inf")
        maximum = max(values)
        return maximum + math.log(sum(math.exp(value - maximum) for value in values))

    def infer(
        self,
        picks: Sequence[Any],
        *,
        top_assignments: int = 3,
    ) -> TeamRoleInference:
        """Infer roles for objects exposing ``champion_id`` and optional ``role``."""
        clean = [pick for pick in picks if int(getattr(pick, "champion_id", 0) or 0) > 0]
        if not clean:
            return TeamRoleInference((), 0.0, ())

        known_roles: dict[int, str] = {}
        used: set[str] = set()
        for index, pick in enumerate(clean):
            role = str(getattr(pick, "role", "") or "").upper()
            if role in ROLES and role not in used:
                known_roles[index] = role
                used.add(role)

        unknown_indexes = [index for index in range(len(clean)) if index not in known_roles]
        available_roles = [role for role in ROLES if role not in used]
        if len(unknown_indexes) > len(available_roles):
            unknown_indexes = unknown_indexes[: len(available_roles)]

        assignments: list[tuple[float, dict[int, str]]] = []
        permutations: Iterable[tuple[str, ...]]
        if unknown_indexes:
            permutations = itertools.permutations(available_roles, len(unknown_indexes))
        else:
            permutations = [()]

        for role_values in permutations:
            assignment = dict(known_roles)
            assignment.update(dict(zip(unknown_indexes, role_values)))
            log_score = 0.0
            for index, role in assignment.items():
                champion_id = int(getattr(clean[index], "champion_id"))
                probability = 0.999 if index in known_roles else self.probability(champion_id, role)
                log_score += math.log(max(1e-9, probability))
            assignments.append((log_score, assignment))

        assignments.sort(key=lambda item: item[0], reverse=True)
        log_total = self._logsumexp([score for score, _ in assignments])
        weighted = [
            (math.exp(score - log_total), assignment)
            for score, assignment in assignments
        ]
        best_probability, best_assignment = weighted[0]

        guesses: list[RoleGuess] = []
        for index, pick in enumerate(clean):
            champion_id = int(getattr(pick, "champion_id"))
            role_mass = {role: 0.0 for role in ROLES}
            for probability, assignment in weighted:
                role = assignment.get(index)
                if role:
                    role_mass[role] += probability
            ordered = sorted(role_mass.items(), key=lambda item: item[1], reverse=True)
            chosen = best_assignment.get(index, ordered[0][0])
            alternatives = tuple((role, probability) for role, probability in ordered[:3])
            guesses.append(
                RoleGuess(
                    champion_id=champion_id,
                    role=chosen,
                    confidence=100.0 * role_mass.get(chosen, 0.0),
                    alternatives=alternatives,
                )
            )

        alternative_payload: list[tuple[tuple[tuple[int, str], ...], float]] = []
        for probability, assignment in weighted[: max(1, top_assignments)]:
            pairs = tuple(
                (int(getattr(clean[index], "champion_id")), role)
                for index, role in sorted(assignment.items())
            )
            alternative_payload.append((pairs, probability))

        return TeamRoleInference(
            guesses=tuple(guesses),
            assignment_confidence=100.0 * best_probability,
            alternatives=tuple(alternative_payload),
        )
