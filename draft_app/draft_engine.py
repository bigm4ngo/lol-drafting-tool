"""Low-latency v3 draft engine with embeddings, role inference and explanations."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from config_manager import PROFILE_PATH, ROLES, load_profile
from data_dragon_maps import ChampionCatalog, normalize_champion_name
from draft_explainer import DraftExplanation, explain_candidate, explain_team
from ml_features import aggregate_team, damage_balance, model_feature_vector, neutral_feature_row
from ml_runtime import DEFAULT_META_PATH, DEFAULT_MODEL_PATH, ModelStatus, NeuralPrediction, PortableDraftModel
from role_inference import RoleGuess, RoleInferenceEngine, TeamRoleInference
from scraper import DEFAULT_DB_PATH
from static_data import StaticDataCatalog

LOGGER = logging.getLogger("draft_engine")


@dataclass(frozen=True, slots=True)
class Pick:
    champion_id: int
    role: str | None = None


@dataclass(frozen=True, slots=True)
class BuildOption:
    kind: str
    ids: tuple[int, ...]
    names: tuple[str, ...]
    games: int
    win_rate: float
    adjusted_win_rate: float
    primary_style_id: int = 0
    sub_style_id: int = 0
    stat_perks: Mapping[str, Any] = field(default_factory=dict)
    recommendation_score: float = 0.0
    context_note: str = ""


@dataclass(frozen=True, slots=True)
class LoadoutOption:
    """One observed item-core + rune-page combination.

    V3.0.4 ranked item cores and rune pages independently, which could present
    two individually popular choices that were rarely used together.  V3.0.6
    stores and ranks the pair as one historical loadout.  Existing analytics
    databases transparently fall back to pairing the independently ranked
    options until analytics are rebuilt.
    """

    item_ids: tuple[int, ...]
    item_names: tuple[str, ...]
    rune_ids: tuple[int, ...]
    rune_names: tuple[str, ...]
    games: int
    win_rate: float
    adjusted_win_rate: float
    primary_style_id: int = 0
    sub_style_id: int = 0
    stat_perks: Mapping[str, Any] = field(default_factory=dict)
    recommendation_score: float = 0.0
    context_note: str = ""


@dataclass(frozen=True, slots=True)
class CompositionSummary:
    score: float
    damage_balance: float
    physical_share: float
    magic_share: float
    true_share: float
    damage_profile: str
    frontline: float
    control: float
    hard_cc: float
    objective: float
    engage: float
    disengage: float
    pick_potential: float
    waveclear: float
    mobility: float
    vision: float
    early_strength: float
    mid_strength: float
    late_strength: float
    feature_confidence: float

    def metric_map(self) -> dict[str, float]:
        return {
            "damage_balance": self.damage_balance,
            "physical_share": self.physical_share,
            "magic_share": self.magic_share,
            "true_share": self.true_share,
            "frontline": self.frontline,
            "control": self.control,
            "hard_cc": self.hard_cc,
            "objective": self.objective,
            "engage": self.engage,
            "disengage": self.disengage,
            "pick_potential": self.pick_potential,
            "waveclear": self.waveclear,
            "mobility": self.mobility,
            "vision": self.vision,
            "early_strength": self.early_strength,
            "mid_strength": self.mid_strength,
            "late_strength": self.late_strength,
            "feature_confidence": self.feature_confidence,
        }


@dataclass(frozen=True, slots=True)
class Recommendation:
    role: str
    champion_id: int
    champion_name: str
    selected: bool
    score: float
    confidence_score: float
    role_confidence: float
    global_win_rate: float
    global_games: int
    weighted_games: float
    patch_freshness: float
    champion_damage_profile: str
    champion_physical_share: float
    champion_magic_share: float
    champion_true_share: float
    synergy_delta: float
    synergy_games: int
    counter_delta: float
    counter_games: int
    composition_score: float
    ml_win_probability: float
    ml_uplift: float
    ml_ensemble_std: float
    personal_multiplier: float
    pool_category: str
    composition: CompositionSummary
    explanation_summary: str = ""
    strengths: tuple[str, ...] = ()
    weaknesses: tuple[str, ...] = ()
    item_builds: tuple[BuildOption, ...] = ()
    rune_pages: tuple[BuildOption, ...] = ()
    summoner_spells: tuple[BuildOption, ...] = ()
    loadouts: tuple[LoadoutOption, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BanRecommendation:
    role: str
    champion_id: int
    champion_name: str
    score: float
    global_win_rate: float
    global_games: int
    matchup_threat: float
    matchup_games: int
    confidence_score: float = 0.0
    target_ally_id: int = 0
    target_ally_name: str = ""
    target_is_hover: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DraftInsights:
    ally_composition: CompositionSummary
    enemy_composition: CompositionSummary
    predicted_win_probability: float
    prediction_confidence: float
    ally_role_inference: TeamRoleInference
    enemy_role_inference: TeamRoleInference
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    summary: str


@dataclass(frozen=True, slots=True)
class EngineSummary:
    matches: int = 0
    participants: int = 0
    champions: int = 0
    analytics_built_at: str = ""
    static_patch: str = ""
    analytics_version: str = ""
    ml_examples: int = 0
    ml_matches: int = 0
    ml_backend: str = ""
    ml_device: str = ""
    ml_validation_accuracy: float = 0.0
    ml_validation_brier: float = 0.25
    ml_reason: str = ""
    loaded_in_ms: float = 0.0


class DraftEngine:
    def __init__(
        self,
        database_path: Path = DEFAULT_DB_PATH,
        profile_path: Path = PROFILE_PATH,
        model_path: Path = DEFAULT_MODEL_PATH,
        catalog: ChampionCatalog | None = None,
        static: StaticDataCatalog | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.profile_path = Path(profile_path)
        self.model_path = Path(model_path)
        self.model_meta_path = self.model_path.with_suffix(".json")
        self.catalog = catalog or ChampionCatalog.load(allow_download=False)
        self.static = static or StaticDataCatalog.load(allow_download=False)
        self._lock = threading.RLock()
        self.profile: dict[str, Any] = {}
        self.summary = EngineSummary()
        self._global: dict[tuple[str, int], dict[str, Any]] = {}
        self._synergy: dict[tuple[str, int, str, int], dict[str, Any]] = {}
        self._counter: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._features: dict[tuple[str, int], dict[str, Any]] = {}
        self._builds: dict[tuple[str, str, int], tuple[BuildOption, ...]] = {}
        self._loadouts: dict[tuple[str, int], tuple[LoadoutOption, ...]] = {}
        self._role_candidates: dict[str, tuple[int, ...]] = {role: () for role in ROLES}
        self._role_games: dict[tuple[int, str], int] = {}
        self._role_priors: dict[tuple[int, str], float] = {}
        self._names: dict[int, str] = {record.champion_id: record.name for record in self.catalog.records}
        self._normalised_names = self._build_name_index(self._names)
        self._pools: dict[tuple[str, str], set[int]] = {}
        self.neural_model = PortableDraftModel(self.model_path, self.model_meta_path)
        self.role_inference = RoleInferenceEngine()
        self.last_insights = self._empty_insights()
        self.reload()

    def _empty_composition(self) -> CompositionSummary:
        return CompositionSummary(
            score=0.5, damage_balance=0.5, physical_share=0.5, magic_share=0.5,
            true_share=0.0, damage_profile="Unknown", frontline=0.5, control=0.5,
            hard_cc=0.5, objective=0.5, engage=0.5, disengage=0.5,
            pick_potential=0.5, waveclear=0.5, mobility=0.5, vision=0.5,
            early_strength=0.5, mid_strength=0.5, late_strength=0.5,
            feature_confidence=0.0,
        )

    def _empty_insights(self) -> DraftInsights:
        empty = self._empty_composition()
        inference = TeamRoleInference((), 0.0, ())
        return DraftInsights(empty, empty, 0.5, 0.0, inference, inference, (), (), "Waiting for draft data.")

    def _build_name_index(self, names: Mapping[int, str]) -> dict[str, int]:
        output = {normalize_champion_name(name): int(champion_id) for champion_id, name in names.items() if name}
        for record in self.catalog.records:
            for value in (record.name, record.alias):
                if value:
                    output[normalize_champion_name(value)] = record.champion_id
        return output

    @property
    def ready(self) -> bool:
        return any(self._role_candidates.values())

    @property
    def model_status(self) -> ModelStatus:
        return self.neural_model.status

    @staticmethod
    def _edit_distance(left: str, right: str) -> int:
        if left == right:
            return 0
        if not left:
            return len(right)
        if not right:
            return len(left)
        previous = list(range(len(right) + 1))
        for row, char_left in enumerate(left, start=1):
            current = [row]
            for column, char_right in enumerate(right, start=1):
                current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (char_left != char_right)))
            previous = current
        return previous[-1]

    def _fuzzy_champion(self, text: str) -> tuple[int | None, str]:
        normalized = normalize_champion_name(text)
        if len(normalized) < 3:
            return None, ""
        ranked: list[tuple[int, int, str]] = []
        for name_key, champion_id in self._normalised_names.items():
            if abs(len(name_key) - len(normalized)) > 2:
                continue
            distance = self._edit_distance(normalized, name_key)
            if distance <= 2:
                ranked.append((distance, abs(len(name_key) - len(normalized)), self._names[champion_id]))
        if not ranked:
            return None, ""
        ranked.sort()
        if len(ranked) > 1 and ranked[1][:2] == ranked[0][:2]:
            return None, ""
        corrected = ranked[0][2]
        return self._normalised_names[normalize_champion_name(corrected)], corrected

    def resolve_champion(self, value: int | str | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        text = str(value).strip()
        if not text:
            return None
        if text.isdigit():
            return int(text) if int(text) > 0 else None
        record = self.catalog.resolve(text)
        exact = record.champion_id if record else self._normalised_names.get(normalize_champion_name(text))
        if exact:
            return exact
        fuzzy, _ = self._fuzzy_champion(text)
        return fuzzy

    def canonical_champion_name(self, value: str) -> tuple[str | None, bool]:
        text = str(value).strip()
        record = self.catalog.resolve(text)
        if record:
            supplied = normalize_champion_name(text)
            exact = {normalize_champion_name(x) for x in (record.name, record.alias) if x}
            return record.name, supplied not in exact
        champion_id = self._normalised_names.get(normalize_champion_name(text))
        if champion_id:
            return self._names[champion_id], False
        champion_id, corrected = self._fuzzy_champion(text)
        return (corrected, True) if champion_id else (None, False)

    def _resolve_pool(self, category: str, role: str) -> set[int]:
        output: set[int] = set()
        for value in self.profile.get(category, {}).get(role, []):
            champion_id = self.resolve_champion(value)
            if champion_id:
                output.add(champion_id)
                canonical, corrected = self.canonical_champion_name(str(value))
                if corrected and canonical:
                    LOGGER.warning("Interpreting profile champion %r as %r.", value, canonical)
            else:
                LOGGER.warning("Unresolved profile champion: %r", value)
        return output

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
        return connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone() is not None

    def _read_table(self, connection: sqlite3.Connection, name: str) -> pd.DataFrame:
        return pd.read_sql_query(f"SELECT * FROM {name}", connection) if self._table_exists(connection, name) else pd.DataFrame()

    def _load_meta(self, connection: sqlite3.Connection) -> dict[str, str]:
        if not self._table_exists(connection, "analytics_meta"):
            return {}
        return {str(key): str(value) for key, value in connection.execute("SELECT key,value FROM analytics_meta")}

    def _load_build_options(self, frame: pd.DataFrame) -> dict[tuple[str, str, int], tuple[BuildOption, ...]]:
        grouped: dict[tuple[str, str, int], list[BuildOption]] = {}
        for row in frame.itertuples(index=False):
            try:
                sample = json.loads(row.sample_json)
            except (TypeError, json.JSONDecodeError):
                sample = {}
            kind = str(row.kind)
            if kind == "loadouts":
                continue
            if kind == "items":
                ids = tuple(int(x) for x in sample.get("ids", []) if int(x or 0) > 0)
                names = tuple(self.static.name("item", identifier) for identifier in ids)
                primary = sub = 0; stat_perks: Mapping[str, Any] = {}
            elif kind == "spells":
                ids = tuple(int(x) for x in sample.get("ids", []) if int(x or 0) > 0)
                names = tuple(self.static.name("spell", identifier) for identifier in ids)
                primary = sub = 0; stat_perks = {}
            else:
                ids = tuple(int(x) for x in sample.get("perk_ids", []) if int(x or 0) > 0)
                names = tuple(self.static.name("rune", identifier) for identifier in ids)
                primary = int(sample.get("primary_style_id", 0) or 0)
                sub = int(sample.get("sub_style_id", 0) or 0)
                stat_perks = sample.get("stat_perks", {}) or {}
            option = BuildOption(
                kind=kind, ids=ids, names=names, games=int(row.games),
                win_rate=float(row.win_rate), adjusted_win_rate=float(row.adjusted_win_rate),
                primary_style_id=primary, sub_style_id=sub, stat_perks=stat_perks,
            )
            grouped.setdefault((kind, str(row.role), int(row.champion_id)), []).append(option)
        return {key: tuple(values) for key, values in grouped.items()}

    def _load_loadout_options(
        self, frame: pd.DataFrame
    ) -> dict[tuple[str, int], tuple[LoadoutOption, ...]]:
        grouped: dict[tuple[str, int], list[LoadoutOption]] = {}
        if frame.empty or "kind" not in frame.columns:
            return {}
        rows = frame[frame["kind"].astype(str) == "loadouts"]
        for row in rows.itertuples(index=False):
            try:
                sample = json.loads(row.sample_json)
            except (TypeError, json.JSONDecodeError):
                sample = {}
            item_ids = tuple(
                int(value)
                for value in sample.get("item_ids", [])
                if int(value or 0) > 0
            )
            rune_ids = tuple(
                int(value)
                for value in sample.get("perk_ids", [])
                if int(value or 0) > 0
            )
            if not item_ids or not rune_ids:
                continue
            option = LoadoutOption(
                item_ids=item_ids,
                item_names=tuple(self.static.name("item", value) for value in item_ids),
                rune_ids=rune_ids,
                rune_names=tuple(self.static.name("rune", value) for value in rune_ids),
                games=int(row.games),
                win_rate=float(row.win_rate),
                adjusted_win_rate=float(row.adjusted_win_rate),
                primary_style_id=int(sample.get("primary_style_id", 0) or 0),
                sub_style_id=int(sample.get("sub_style_id", 0) or 0),
                stat_perks=sample.get("stat_perks", {}) or {},
            )
            grouped.setdefault((str(row.role), int(row.champion_id)), []).append(option)
        return {key: tuple(values) for key, values in grouped.items()}

    def reload(self) -> None:
        started = time.perf_counter()
        profile = load_profile(self.profile_path)

        def apply_empty_state() -> None:
            with self._lock:
                self.profile = profile
                self._role_candidates = {role: () for role in ROLES}
                self.summary = EngineSummary(loaded_in_ms=(time.perf_counter() - started) * 1000)

        if not self.database_path.exists():
            apply_empty_state()
            return
        # Reload is read-only. A bounded timeout prevents a concurrent analytics
        # writer from making the GUI appear to hang indefinitely.
        try:
            with sqlite3.connect(self.database_path, timeout=5.0) as connection:
                connection.execute("PRAGMA query_only = ON")
                global_frame = self._read_table(connection, "analytics_global")
                if global_frame.empty:
                    apply_empty_state()
                    return
                synergy_frame = self._read_table(connection, "analytics_synergy")
                counter_frame = self._read_table(connection, "analytics_counter")
                feature_frame = self._read_table(connection, "analytics_features")
                build_frame = self._read_table(connection, "analytics_builds")
                role_prior_frame = self._read_table(connection, "analytics_role_priors")
                meta = self._load_meta(connection)
        except sqlite3.DatabaseError as exc:
            LOGGER.warning(
                "Draft database is unavailable (%s: %s); running with empty stats.",
                type(exc).__name__, exc,
            )
            apply_empty_state()
            return

        names = dict(self._names)
        for row in global_frame.itertuples(index=False):
            champion_id = int(row.champion_id)
            names[champion_id] = self.catalog.name_for_id(champion_id, str(row.champion_name))
        global_stats: dict[tuple[str, int], dict[str, Any]] = {}
        for row in global_frame.itertuples(index=False):
            global_stats[(str(row.role), int(row.champion_id))] = {
                "games": int(row.games), "wins": float(row.wins),
                "weighted_games": float(getattr(row, "weighted_games", row.games)),
                "win_rate": float(row.win_rate),
                "patch_win_rate": float(getattr(row, "patch_win_rate", row.win_rate)),
                "adjusted_win_rate": float(row.adjusted_win_rate),
                "patch_freshness": float(getattr(row, "patch_freshness", 1.0)),
            }
        synergy_stats = {
            (str(row.candidate_role), int(row.candidate_id), str(row.ally_role), int(row.ally_id)): {
                "games": int(row.games), "weighted_games": float(getattr(row, "weighted_games", row.games)),
                "delta": float(row.delta), "adjusted_delta": float(row.adjusted_delta),
            } for row in synergy_frame.itertuples(index=False)
        }
        counter_stats = {
            (str(row.candidate_role), int(row.candidate_id), int(row.enemy_id)): {
                "games": int(row.games), "weighted_games": float(getattr(row, "weighted_games", row.games)),
                "delta": float(row.delta), "adjusted_delta": float(row.adjusted_delta),
            } for row in counter_frame.itertuples(index=False)
        }
        feature_stats: dict[tuple[str, int], dict[str, Any]] = {}
        for row in feature_frame.itertuples(index=False):
            payload = row._asdict()
            feature_stats[(str(row.role), int(row.champion_id))] = {
                key: payload.get(key, neutral_feature_row().get(key, 0.5))
                for key in payload if key not in {"role", "champion_id"}
            }
        role_candidates: dict[str, tuple[int, ...]] = {}
        role_games: dict[tuple[int, str], int] = {}
        for role in ROLES:
            role_rows = global_frame[global_frame["role"] == role].sort_values(
                "weighted_games" if "weighted_games" in global_frame else "games", ascending=False
            )
            role_candidates[role] = tuple(int(x) for x in role_rows["champion_id"])
            for row in role_rows.itertuples(index=False):
                role_games[(int(row.champion_id), role)] = int(row.games)
        if role_prior_frame.empty:
            total_by_champion: dict[int, int] = {}
            for (champion_id, _), games in role_games.items():
                total_by_champion[champion_id] = total_by_champion.get(champion_id, 0) + games
            role_priors = {
                (champion_id, role): (games + 0.35) / (total_by_champion[champion_id] + 0.35 * len(ROLES))
                for (champion_id, role), games in role_games.items()
            }
        else:
            role_priors = {
                (int(row.champion_id), str(row.role)): float(row.probability)
                for row in role_prior_frame.itertuples(index=False)
            }

        self.neural_model.reload()
        with self._lock:
            self.profile = profile
            self._names = names
            self._normalised_names = self._build_name_index(names)
            self._global = global_stats
            self._synergy = synergy_stats
            self._counter = counter_stats
            self._features = feature_stats
            self._builds = self._load_build_options(build_frame)
            self._loadouts = self._load_loadout_options(build_frame)
            self._role_candidates = role_candidates
            self._role_games = role_games
            self._role_priors = role_priors
            self.role_inference.set_priors(role_priors)
            self._pools = {
                (category, role): self._resolve_pool(category, role)
                for category in ("comfort_picks", "pocket_picks", "general_pool")
                for role in ROLES
            }
            self.summary = EngineSummary(
                matches=int(meta.get("matches", "0") or 0),
                participants=int(meta.get("participants", "0") or 0),
                champions=len({champion for _, champion in global_stats}),
                analytics_built_at=meta.get("built_at", ""),
                static_patch=meta.get("static_patch", ""),
                analytics_version=meta.get("analytics_version", "2.x"),
                ml_examples=int(meta.get("ml_examples", "0") or 0),
                ml_matches=int(meta.get("ml_matches", "0") or 0),
                ml_backend=meta.get("ml_backend", self.neural_model.status.backend),
                ml_device=meta.get("ml_device", self.neural_model.status.device),
                ml_validation_accuracy=float(meta.get("ml_validation_accuracy", "0") or 0),
                ml_validation_brier=float(meta.get("ml_validation_brier", "0.25") or 0.25),
                ml_reason=meta.get("ml_reason", ""),
                loaded_in_ms=(time.perf_counter() - started) * 1000,
            )
        LOGGER.info(
            "Draft engine v3 loaded in %.1fms with %d participants; neural=%s on %s.",
            self.summary.loaded_in_ms, self.summary.participants,
            self.neural_model.status.available, self.neural_model.status.device,
        )

    @staticmethod
    def _canonical_role(value: Any) -> str | None:
        text = str(value or "").upper()
        text = {"MIDDLE": "MID", "BOTTOM": "ADC", "BOT": "ADC", "UTILITY": "SUPPORT"}.get(text, text)
        return text if text in ROLES else None

    def _coerce_picks(self, values: Mapping[str, Any] | Sequence[Any] | None) -> list[Pick]:
        if not values:
            return []
        output: list[Pick] = []
        iterable: Iterable[Any] = values.items() if isinstance(values, Mapping) else values
        for value in iterable:
            role: str | None = None; champion: Any = value
            if isinstance(value, Pick):
                output.append(value); continue
            if isinstance(value, Mapping):
                role = self._canonical_role(value.get("role")); champion = value.get("champion_id", value.get("champion"))
            elif isinstance(value, tuple) and len(value) == 2:
                role = self._canonical_role(value[0]); champion = value[1]
            champion_id = self.resolve_champion(champion)
            if champion_id:
                output.append(Pick(champion_id, role))
        return output

    def _infer_roles_with_confidence(self, picks: list[Pick]) -> tuple[list[Pick], TeamRoleInference, dict[int, float]]:
        inference = self.role_inference.infer(picks)
        role_by_champion = {guess.champion_id: guess.role for guess in inference.guesses}
        confidence = {guess.champion_id: guess.confidence for guess in inference.guesses}
        result = [Pick(pick.champion_id, pick.role or role_by_champion.get(pick.champion_id)) for pick in picks]
        return result, inference, confidence

    def _infer_roles(self, picks: list[Pick]) -> list[Pick]:
        return self._infer_roles_with_confidence(picks)[0]

    def infer_role_for_champion(self, champion_id: int | str | None) -> str | None:
        resolved = self.resolve_champion(champion_id)
        if not resolved:
            return None
        ranked = sorted(((probability, role) for (candidate, role), probability in self._role_priors.items() if candidate == resolved), reverse=True)
        return ranked[0][1] if ranked else None

    def role_probabilities(self, champion_id: int | str | None) -> list[tuple[str, float]]:
        resolved = self.resolve_champion(champion_id)
        if not resolved:
            return []
        values = [(role, self._role_priors.get((resolved, role), 0.002)) for role in ROLES]
        total = sum(value for _, value in values) or 1.0
        return sorted(((role, value / total) for role, value in values), key=lambda item: item[1], reverse=True)

    def _pool_details(self, role: str, champion_id: int) -> tuple[str, float]:
        multipliers = self.profile["personal_multipliers"]
        if champion_id in self._pools[("comfort_picks", role)]:
            return "comfort", float(multipliers["comfort"])
        if champion_id in self._pools[("pocket_picks", role)]:
            return "pocket", float(multipliers["pocket"])
        if champion_id in self._pools[("general_pool", role)]:
            return "general", float(multipliers["general"])
        return "meta", 1.0

    def _feature_for(self, pick: Pick) -> Mapping[str, Any]:
        row = self._features.get((pick.role or "", pick.champion_id))
        if row:
            return row
        candidates = [value for (role, champion), value in self._features.items() if champion == pick.champion_id]
        return max(candidates, key=lambda value: float(value.get("games", 0)), default=neutral_feature_row())

    def champion_profile(self, champion: int | str, role: str | None = None) -> dict[str, Any]:
        """Return one champion's patch-weighted role feature and damage profile."""
        champion_id = self.resolve_champion(champion)
        if not champion_id:
            return {"damage_profile": "Unknown", "physical_share": 0.0, "magic_share": 0.0, "true_share": 0.0, "feature_confidence": 0.0}
        canonical_role = self._canonical_role(role)
        row = self._features.get((canonical_role, champion_id)) if canonical_role else None
        if row is None:
            candidates = [
                value for (candidate_role, candidate_id), value in self._features.items()
                if candidate_id == champion_id
            ]
            row = max(candidates, key=lambda value: float(value.get("games", 0)), default=None)
        if row is None:
            return {"damage_profile": "Unknown", "physical_share": 0.0, "magic_share": 0.0, "true_share": 0.0, "feature_confidence": 0.0}
        return {
            "damage_profile": str(row.get("damage_profile", "Mixed")),
            "physical_share": float(row.get("physical_share", 0.0)),
            "magic_share": float(row.get("magic_share", 0.0)),
            "true_share": float(row.get("true_share", 0.0)),
            "feature_confidence": float(row.get("feature_confidence", 0.0)),
            "control": float(row.get("control", 0.5)),
            "hard_cc": float(row.get("hard_cc", 0.5)),
            "engage": float(row.get("engage", 0.5)),
            "pick_potential": float(row.get("pick_potential", 0.5)),
            "waveclear": float(row.get("waveclear", 0.5)),
            "objective": float(row.get("objective", 0.5)),
            "mobility": float(row.get("mobility", 0.5)),
            "frontline": float(row.get("frontline", 0.5)),
            "early_strength": float(row.get("early_strength", 0.5)),
            "mid_strength": float(row.get("mid_strength", 0.5)),
            "late_strength": float(row.get("late_strength", 0.5)),
        }

    def _composition(self, allies: list[Pick]) -> CompositionSummary:
        if not allies:
            return self._empty_composition()
        role_map = {pick.champion_id: pick.role or "" for pick in allies}
        metrics = aggregate_team([pick.champion_id for pick in allies], role_map, self._features)
        physical = float(metrics.get("physical_share", 0.5)); magic = float(metrics.get("magic_share", 0.5)); true = float(metrics.get("true_share", 0.0))
        balance = damage_balance(metrics)
        if physical >= 0.70:
            profile = "Physical"
        elif magic >= 0.70:
            profile = "Magic"
        elif true >= 0.20:
            profile = "True-heavy"
        else:
            profile = "Mixed"
        targets = self.profile.get("composition_targets", {})
        score_components: list[tuple[float, float]] = [(balance, float(targets.get("damage_balance", 1.0)))]
        for key in ("frontline", "control", "objective", "engage", "disengage", "pick_potential", "waveclear", "mobility", "early_strength", "mid_strength", "late_strength"):
            value = float(metrics.get(key, 0.5))
            target = max(0.05, float(targets.get(key, 0.5)))
            score_components.append((min(1.0, value / target), max(0.05, float(targets.get(key, 0.3)))))
        score = sum(value * weight for value, weight in score_components) / sum(weight for _, weight in score_components)
        return CompositionSummary(
            score=score, damage_balance=balance, physical_share=physical, magic_share=magic,
            true_share=true, damage_profile=profile,
            frontline=float(metrics.get("frontline", 0.5)), control=float(metrics.get("control", 0.5)),
            hard_cc=float(metrics.get("hard_cc", 0.5)), objective=float(metrics.get("objective", 0.5)),
            engage=float(metrics.get("engage", 0.5)), disengage=float(metrics.get("disengage", 0.5)),
            pick_potential=float(metrics.get("pick_potential", 0.5)), waveclear=float(metrics.get("waveclear", 0.5)),
            mobility=float(metrics.get("mobility", 0.5)), vision=float(metrics.get("vision", 0.5)),
            early_strength=float(metrics.get("early_strength", 0.5)), mid_strength=float(metrics.get("mid_strength", 0.5)),
            late_strength=float(metrics.get("late_strength", 0.5)), feature_confidence=float(metrics.get("feature_confidence", 0.0)),
        )

    def _ml_prediction(
        self,
        allies: list[Pick],
        enemies: list[Pick],
        *,
        role_confidence: float = 0.5,
        ally_composition: CompositionSummary | None = None,
        enemy_composition: CompositionSummary | None = None,
    ) -> NeuralPrediction:
        ally_comp = ally_composition or self._composition(allies)
        enemy_comp = enemy_composition or self._composition(enemies)
        ally_map = {pick.role: pick.champion_id for pick in allies if pick.role}
        enemy_map = {pick.role: pick.champion_id for pick in enemies if pick.role}
        features = model_feature_vector(ally_comp.metric_map(), enemy_comp.metric_map())
        coverage = (ally_comp.feature_confidence + enemy_comp.feature_confidence) / 2.0
        return self.neural_model.predict(
            ally_map, enemy_map, features,
            data_coverage=coverage, role_confidence=max(0.0, min(1.0, role_confidence)),
        )

    def _enemy_damage_profile(self, enemies: list[Pick]) -> tuple[float, float, float]:
        comp = self._composition(enemies)
        return comp.physical_share, comp.magic_share, comp.true_share

    def _build_options(
        self,
        kind: str,
        role: str,
        champion_id: int,
        enemies: list[Pick] | None = None,
        *,
        enemy_damage: tuple[float, float, float] | None = None,
    ) -> tuple[BuildOption, ...]:
        minimum = int(self.profile["minimum_samples"].get("build", 5))
        options = [option for option in self._builds.get((kind, role, champion_id), ()) if option.games >= minimum]
        if kind == "items" and enemies:
            physical, magic, _ = enemy_damage or self._enemy_damage_profile(enemies)
        else:
            physical, magic = 0.5, 0.5
        ranked: list[BuildOption] = []
        for option in options:
            score = option.adjusted_win_rate + math.log1p(option.games) * 0.004
            note = "Patch-weighted win rate and sample confidence"
            if kind == "items" and option.ids and enemies:
                tags = set().union(*(self.static.item_tags(identifier) for identifier in option.ids))
                reasons: list[str] = []
                if physical >= 0.58 and "armor" in tags:
                    score += min(0.018, (physical - 0.50) * 0.08); reasons.append("extra value into physical damage")
                if magic >= 0.58 and "spellblock" in tags:
                    score += min(0.018, (magic - 0.50) * 0.08); reasons.append("extra value into magic damage")
                if abs(physical - magic) < 0.12 and "health" in tags:
                    score += 0.006; reasons.append("general durability into mixed damage")
                if reasons:
                    note = "; ".join(reasons)
            ranked.append(replace(option, recommendation_score=score, context_note=note))
        ranked.sort(key=lambda item: (item.recommendation_score, item.games, item.adjusted_win_rate), reverse=True)
        return tuple(ranked[:3])

    def _loadout_options(
        self,
        role: str,
        champion_id: int,
        enemies: list[Pick],
        *,
        enemy_damage: tuple[float, float, float],
        fallback_items: tuple[BuildOption, ...],
        fallback_runes: tuple[BuildOption, ...],
    ) -> tuple[LoadoutOption, ...]:
        """Rank observed item+rune pairs, with an old-database fallback."""
        minimum = int(self.profile["minimum_samples"].get("build", 5))
        physical, magic, _ = enemy_damage
        ranked: list[LoadoutOption] = []
        for option in self._loadouts.get((role, champion_id), ()):
            if option.games < minimum:
                continue
            score = option.adjusted_win_rate + math.log1p(option.games) * 0.004
            note = "Observed item core and rune page from the same games"
            tags = set().union(
                *(self.static.item_tags(identifier) for identifier in option.item_ids)
            ) if option.item_ids else set()
            reasons: list[str] = []
            if physical >= 0.58 and "armor" in tags:
                score += min(0.018, (physical - 0.50) * 0.08)
                reasons.append("item core gains value into physical damage")
            if magic >= 0.58 and "spellblock" in tags:
                score += min(0.018, (magic - 0.50) * 0.08)
                reasons.append("item core gains value into magic damage")
            if abs(physical - magic) < 0.12 and "health" in tags:
                score += 0.006
                reasons.append("general durability into mixed damage")
            if reasons:
                note += "; " + "; ".join(reasons)
            ranked.append(
                replace(
                    option,
                    recommendation_score=score,
                    context_note=note,
                )
            )
        ranked.sort(
            key=lambda item: (
                item.recommendation_score,
                item.games,
                item.adjusted_win_rate,
            ),
            reverse=True,
        )
        if ranked:
            return tuple(ranked[:3])

        # Existing v3.0.4 analytics do not contain combined signatures. Keep the
        # new bundled UI useful immediately, then replace these approximations
        # with truly observed pairs after the next analytics rebuild.
        count = min(3, max(len(fallback_items), len(fallback_runes)))
        output: list[LoadoutOption] = []
        for index in range(count):
            item = fallback_items[min(index, len(fallback_items) - 1)] if fallback_items else None
            rune = fallback_runes[min(index, len(fallback_runes) - 1)] if fallback_runes else None
            if item is None or rune is None:
                continue
            output.append(
                LoadoutOption(
                    item_ids=item.ids,
                    item_names=item.names,
                    rune_ids=rune.ids,
                    rune_names=rune.names,
                    games=min(item.games, rune.games),
                    win_rate=(item.win_rate + rune.win_rate) / 2.0,
                    adjusted_win_rate=(
                        item.adjusted_win_rate + rune.adjusted_win_rate
                    ) / 2.0,
                    primary_style_id=rune.primary_style_id,
                    sub_style_id=rune.sub_style_id,
                    stat_perks=rune.stat_perks,
                    recommendation_score=(
                        item.recommendation_score + rune.recommendation_score
                    ) / 2.0,
                    context_note=(
                        "Temporary pairing of independently ranked options; "
                        "rebuild analytics once to learn observed item+rune bundles"
                    ),
                )
            )
        return tuple(output)

    @staticmethod
    def _sample_confidence(games: float, target: float) -> float:
        return 1.0 - math.exp(-max(0.0, games) / max(1.0, target))

    def _recommendation_for(
        self,
        *,
        role: str,
        champion_id: int,
        allies: list[Pick],
        enemies: list[Pick],
        baseline_ml: NeuralPrediction,
        active_role: str | None,
        selected: bool,
        role_confidence: float,
        enemy_composition: CompositionSummary | None = None,
    ) -> Recommendation | None:
        global_stat = self._global.get((role, champion_id))
        if not global_stat:
            return None
        minimums = self.profile["minimum_samples"]
        if not selected and int(global_stat["games"]) < int(minimums["global"]):
            return None
        other_allies = [ally for ally in allies if not (ally.role == role and ally.champion_id == champion_id)]
        synergy_raw: list[float] = []; synergy_adjusted: list[float] = []; synergy_games = 0; synergy_weighted = 0.0
        for ally in other_allies:
            if not ally.role:
                continue
            stat = self._synergy.get((role, champion_id, ally.role, ally.champion_id))
            if stat and int(stat["games"]) >= int(minimums["synergy"]):
                synergy_raw.append(float(stat["delta"])); synergy_adjusted.append(float(stat["adjusted_delta"]))
                synergy_games += int(stat["games"]); synergy_weighted += float(stat.get("weighted_games", stat["games"]))
        counter_raw: list[float] = []; counter_adjusted: list[float] = []; counter_games = 0; counter_weighted = 0.0
        for enemy in (pick for pick in enemies if pick.role == role):
            stat = self._counter.get((role, champion_id, enemy.champion_id))
            if stat and int(stat["games"]) >= int(minimums["counter"]):
                counter_raw.append(float(stat["delta"])); counter_adjusted.append(float(stat["adjusted_delta"]))
                counter_games += int(stat["games"]); counter_weighted += float(stat.get("weighted_games", stat["games"]))
        raw_synergy = float(np.mean(synergy_raw)) if synergy_raw else 0.0
        adjusted_synergy = float(np.mean(synergy_adjusted)) if synergy_adjusted else 0.0
        raw_counter = float(np.mean(counter_raw)) if counter_raw else 0.0
        adjusted_counter = float(np.mean(counter_adjusted)) if counter_adjusted else 0.0
        before_composition = self._composition(other_allies)
        candidate_allies = other_allies + [Pick(champion_id, role)]
        composition = self._composition(candidate_allies)
        enemy_comp = enemy_composition or self._composition(enemies)
        ml_prediction = self._ml_prediction(
            candidate_allies,
            enemies,
            role_confidence=role_confidence / 100.0,
            ally_composition=composition,
            enemy_composition=enemy_comp,
        )
        ml_uplift = ml_prediction.probability - baseline_ml.probability
        weights = self.profile["weights"]
        confidence_components = [
            self._sample_confidence(float(global_stat.get("weighted_games", global_stat["games"])), 70.0),
            self._sample_confidence(synergy_weighted, 30.0) if other_allies else 0.65,
            self._sample_confidence(counter_weighted, 25.0) if any(x.role == role for x in enemies) else 0.65,
            composition.feature_confidence,
            role_confidence / 100.0,
            ml_prediction.confidence / 100.0 if ml_prediction.available else 0.35,
        ]
        confidence_score = 100.0 * float(np.mean(confidence_components))
        score = (
            float(weights["global_win_rate"]) * float(global_stat["adjusted_win_rate"])
            + float(weights["synergy_delta"]) * adjusted_synergy
            + float(weights["counter_delta"]) * adjusted_counter
            + float(weights.get("composition", 0.0)) * (composition.score - 0.5) * 0.10
            + float(weights.get("machine_learning", 0.0)) * ml_uplift
            + float(weights.get("confidence", 0.0)) * ((confidence_score - 50.0) / 100.0) * 0.01
        )
        if active_role and role == active_role:
            category, multiplier = self._pool_details(role, champion_id)
        else:
            category, multiplier = "meta", 1.0
        score *= multiplier
        champion_profile = self.champion_profile(champion_id, role)
        explanation = explain_candidate(
            champion_name=self._names.get(champion_id, f"Champion {champion_id}"),
            metrics=composition.metric_map(), synergy_delta=raw_synergy,
            counter_delta=raw_counter, ml_uplift=ml_uplift,
            confidence=confidence_score, team_before=before_composition.metric_map(),
        )
        item_builds = self._build_options(
            "items", role, champion_id, enemies,
            enemy_damage=(enemy_comp.physical_share, enemy_comp.magic_share, enemy_comp.true_share),
        )
        rune_pages = self._build_options("runes", role, champion_id)
        spell_options = self._build_options("spells", role, champion_id)
        loadouts = self._loadout_options(
            role,
            champion_id,
            enemies,
            enemy_damage=(
                enemy_comp.physical_share,
                enemy_comp.magic_share,
                enemy_comp.true_share,
            ),
            fallback_items=item_builds,
            fallback_runes=rune_pages,
        )
        return Recommendation(
            role=role, champion_id=champion_id,
            champion_name=self._names.get(champion_id, f"Champion {champion_id}"),
            selected=selected, score=score, confidence_score=confidence_score,
            role_confidence=role_confidence, global_win_rate=float(global_stat["win_rate"]),
            global_games=int(global_stat["games"]), weighted_games=float(global_stat.get("weighted_games", global_stat["games"])),
            patch_freshness=float(global_stat.get("patch_freshness", 1.0)),
            champion_damage_profile=str(champion_profile.get("damage_profile", "Mixed")),
            champion_physical_share=float(champion_profile.get("physical_share", 0.0)),
            champion_magic_share=float(champion_profile.get("magic_share", 0.0)),
            champion_true_share=float(champion_profile.get("true_share", 0.0)),
            synergy_delta=raw_synergy, synergy_games=synergy_games,
            counter_delta=raw_counter, counter_games=counter_games,
            composition_score=composition.score, ml_win_probability=ml_prediction.probability,
            ml_uplift=ml_uplift, ml_ensemble_std=ml_prediction.ensemble_std,
            personal_multiplier=multiplier, pool_category=category, composition=composition,
            explanation_summary=explanation.summary, strengths=explanation.strengths,
            weaknesses=explanation.weaknesses,
            item_builds=item_builds,
            rune_pages=rune_pages,
            summoner_spells=spell_options,
            loadouts=loadouts,
        )

    def analyze_draft(
        self,
        ally_picks: Mapping[str, Any] | Sequence[Any] | None = None,
        enemy_picks: Mapping[str, Any] | Sequence[Any] | None = None,
    ) -> DraftInsights:
        allies_raw = self._coerce_picks(ally_picks); enemies_raw = self._coerce_picks(enemy_picks)
        allies, ally_inference, ally_conf = self._infer_roles_with_confidence(allies_raw)
        enemies, enemy_inference, enemy_conf = self._infer_roles_with_confidence(enemies_raw)
        role_values = list(ally_conf.values()) + list(enemy_conf.values())
        role_confidence = float(np.mean(role_values) / 100.0) if role_values else 0.5
        ally_comp = self._composition(allies); enemy_comp = self._composition(enemies)
        prediction = self._ml_prediction(allies, enemies, role_confidence=role_confidence)
        explanation = explain_team(ally_comp.metric_map())
        insights = DraftInsights(
            ally_composition=ally_comp, enemy_composition=enemy_comp,
            predicted_win_probability=prediction.probability,
            prediction_confidence=prediction.confidence,
            ally_role_inference=ally_inference, enemy_role_inference=enemy_inference,
            strengths=explanation.strengths, weaknesses=explanation.weaknesses,
            summary=explanation.summary,
        )
        self.last_insights = insights
        return insights

    def evaluate_draft(
        self,
        ally_picks: Mapping[str, Any] | Sequence[Any] | None = None,
        enemy_picks: Mapping[str, Any] | Sequence[Any] | None = None,
        bans: Iterable[int | str] | None = None,
        *, top_n: int | None = None, active_role: str | None = None,
    ) -> dict[str, list[Recommendation]]:
        started = time.perf_counter()
        with self._lock:
            if not self.ready:
                return {role: [] for role in ROLES}
            allies, ally_inference, ally_confidence = self._infer_roles_with_confidence(self._coerce_picks(ally_picks))
            enemies, enemy_inference, enemy_confidence = self._infer_roles_with_confidence(self._coerce_picks(enemy_picks))
            active_role = self._canonical_role(active_role)
            banned = {champ for value in (bans or []) if (champ := self.resolve_champion(value))}
            occupied = banned | {pick.champion_id for pick in allies + enemies}
            restrict = bool(self.profile.get("restrict_to_pool", False))
            count = max(1, min(20, int(top_n or self.profile["ui"].get("top_n", 10))))
            all_role_conf = list(ally_confidence.values()) + list(enemy_confidence.values())
            mean_role_conf = float(np.mean(all_role_conf)) if all_role_conf else 50.0
            ally_comp_before = self._composition(allies)
            enemy_comp = self._composition(enemies)
            baseline_ml = self._ml_prediction(
                allies, enemies, role_confidence=mean_role_conf / 100.0,
                ally_composition=ally_comp_before, enemy_composition=enemy_comp,
            )
            results: dict[str, list[Recommendation]] = {}
            locked_by_role = {pick.role: pick for pick in allies if pick.role}
            for role in ROLES:
                selected_pick = locked_by_role.get(role)
                if selected_pick:
                    selected = self._recommendation_for(
                        role=role, champion_id=selected_pick.champion_id,
                        allies=allies, enemies=enemies, baseline_ml=baseline_ml,
                        active_role=active_role, selected=True,
                        role_confidence=ally_confidence.get(selected_pick.champion_id, 100.0),
                        enemy_composition=enemy_comp,
                    )
                    results[role] = [selected] if selected else []
                    continue
                allowed = self._pools[("comfort_picks", role)] | self._pools[("pocket_picks", role)] | self._pools[("general_pool", role)]
                values: list[Recommendation] = []
                for champion_id in self._role_candidates.get(role, ()):
                    if champion_id in occupied:
                        continue
                    if active_role == role and restrict and champion_id not in allowed:
                        continue
                    recommendation = self._recommendation_for(
                        role=role, champion_id=champion_id, allies=allies, enemies=enemies,
                        baseline_ml=baseline_ml, active_role=active_role, selected=False,
                        role_confidence=100.0 * self._role_priors.get((champion_id, role), 0.5),
                        enemy_composition=enemy_comp,
                    )
                    if recommendation:
                        values.append(recommendation)
                values.sort(key=lambda rec: (rec.score, rec.confidence_score, rec.weighted_games), reverse=True)
                results[role] = values[:count]
            ally_comp = self._composition(allies); enemy_comp = self._composition(enemies)
            explanation = explain_team(ally_comp.metric_map())
            self.last_insights = DraftInsights(
                ally_composition=ally_comp, enemy_composition=enemy_comp,
                predicted_win_probability=baseline_ml.probability,
                prediction_confidence=baseline_ml.confidence,
                ally_role_inference=ally_inference, enemy_role_inference=enemy_inference,
                strengths=explanation.strengths, weaknesses=explanation.weaknesses,
                summary=explanation.summary,
            )
        LOGGER.debug("V3 five-role evaluation took %.2fms", (time.perf_counter() - started) * 1000)
        return results

    def _coerce_ally_context(self, values: Mapping[str, Any] | Sequence[Any] | None) -> list[tuple[Pick, bool]]:
        if not values:
            return []
        output: list[tuple[Pick, bool]] = []
        iterable: Iterable[Any] = [
            {"role": role, "champion": champion, "locked": True} for role, champion in values.items()
        ] if isinstance(values, Mapping) else values
        for value in iterable:
            locked = True; role: str | None = None; champion: Any = value
            if isinstance(value, Pick):
                output.append((value, True)); continue
            if isinstance(value, Mapping):
                role = self._canonical_role(value.get("role")); champion = value.get("champion_id", value.get("champion")); locked = bool(value.get("locked", True))
            elif isinstance(value, tuple) and len(value) == 2:
                role = self._canonical_role(value[0]); champion = value[1]
            champion_id = self.resolve_champion(champion)
            if champion_id:
                output.append((Pick(champion_id, role), locked))
        inferred = self._infer_roles([pick for pick, _ in output])
        return [(pick, output[index][1]) for index, pick in enumerate(inferred)]

    def evaluate_bans(
        self,
        ally_context: Mapping[str, Any] | Sequence[Any] | None = None,
        enemy_picks: Mapping[str, Any] | Sequence[Any] | None = None,
        bans: Iterable[int | str] | None = None,
        *, top_n: int = 5,
    ) -> dict[str, list[BanRecommendation]]:
        with self._lock:
            if not self.ready:
                return {role: [] for role in ROLES}
            allies = self._coerce_ally_context(ally_context)
            enemies = self._infer_roles(self._coerce_picks(enemy_picks))
            banned = {champ for value in (bans or []) if (champ := self.resolve_champion(value))}
            occupied = banned | {pick.champion_id for pick, _ in allies} | {pick.champion_id for pick in enemies}
            enemy_filled = {pick.role for pick in enemies if pick.role}
            minimum_counter = int(self.profile["minimum_samples"]["counter"])
            output: dict[str, list[BanRecommendation]] = {}
            for role in ROLES:
                if role in enemy_filled:
                    output[role] = []; continue
                role_allies = [(pick, locked) for pick, locked in allies if pick.role == role]
                values: list[BanRecommendation] = []
                for champion_id in self._role_candidates.get(role, ()):
                    if champion_id in occupied:
                        continue
                    global_stat = self._global.get((role, champion_id))
                    if not global_stat:
                        continue
                    best_threat = 0.0; matchup_games = 0; target_id = 0; target_locked = True
                    for ally, locked in role_allies:
                        # Counter rows are keyed as (candidate role, candidate,
                        # opposing champion). For ban advice the candidate is the
                        # enemy champion we may ban, and the opposing champion is
                        # our ally. v3.0.5 looked this up backwards and negated the
                        # result, suppressing the intended hover-counter influence.
                        stat = self._counter.get((role, champion_id, ally.champion_id))
                        if not stat or int(stat["games"]) < minimum_counter:
                            continue
                        threat = float(stat["adjusted_delta"]) * (1.0 if locked else 0.70)
                        if target_id == 0 or threat > best_threat:
                            best_threat = threat; matchup_games = int(stat["games"]); target_id = ally.champion_id; target_locked = locked
                    confidence = 100.0 * float(np.mean([
                        self._sample_confidence(float(global_stat.get("weighted_games", global_stat["games"])), 70.0),
                        self._sample_confidence(matchup_games, 25.0) if target_id else 0.55,
                        float(global_stat.get("patch_freshness", 1.0)),
                    ]))
                    score = float(global_stat["adjusted_win_rate"]) + 1.65 * best_threat
                    values.append(BanRecommendation(
                        role=role, champion_id=champion_id,
                        champion_name=self._names.get(champion_id, f"Champion {champion_id}"),
                        score=score, global_win_rate=float(global_stat["win_rate"]),
                        global_games=int(global_stat["games"]), matchup_threat=best_threat,
                        matchup_games=matchup_games, confidence_score=confidence,
                        target_ally_id=target_id,
                        target_ally_name=self._names.get(target_id, "") if target_id else "",
                        target_is_hover=bool(target_id and not target_locked),
                    ))
                values.sort(key=lambda rec: (rec.score, rec.confidence_score, rec.global_games), reverse=True)
                output[role] = values[:max(1, min(10, int(top_n)))]
            return output

    def nearest_champions(self, champion: int | str, *, limit: int | None = None) -> list[tuple[str, float]]:
        champion_id = self.resolve_champion(champion)
        if not champion_id:
            return []
        count = int(limit or self.profile.get("ui", {}).get("embedding_neighbors", 8))
        values = self.neural_model.nearest_champions(champion_id, self._names.keys(), limit=count)
        return [(self._names.get(identifier, str(identifier)), similarity) for identifier, similarity in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a v3 League draft.")
    parser.add_argument("--ally", action="append", default=[])
    parser.add_argument("--enemy", action="append", default=[])
    parser.add_argument("--ban", action="append", default=[])
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    engine = DraftEngine()
    output = engine.evaluate_draft(args.ally, args.enemy, args.ban, top_n=args.top)
    print(json.dumps({role: [item.to_dict() for item in items] for role, items in output.items()}, indent=2))


if __name__ == "__main__":
    main()
