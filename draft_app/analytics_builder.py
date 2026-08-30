"""Patch-aware analytics and v3 neural model training.

Heavy Pandas/PyTorch work stays offline.  The live engine reads compact SQLite
analytics tables and a portable NumPy neural ensemble.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config_manager import PROFILE_PATH, load_profile
from ml_features import CHAMPION_FEATURES
from ml_runtime import DEFAULT_META_PATH, DEFAULT_MODEL_PATH, PortableDraftModel
from ml_training import NeuralTrainingReport, train_neural_ensemble
from patch_utils import patch_label, recency_weight
from runtime_paths import PROJECT_ROOT
from scraper import DEFAULT_DB_PATH, DraftDatabase
from static_data import StaticDataCatalog

LOGGER = logging.getLogger("analytics_builder")
ROLES = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")


@dataclass(frozen=True, slots=True)
class BuildReport:
    matches: int
    participants: int
    global_rows: int
    synergy_rows: int
    counter_rows: int
    build_rows: int
    feature_rows: int
    role_prior_rows: int
    patch_rows: int
    ml_examples: int
    neural_trained: bool
    neural_device: str


class AnalyticsBuilder:
    def __init__(
        self,
        database_path: Path = DEFAULT_DB_PATH,
        model_path: Path = DEFAULT_MODEL_PATH,
        profile_path: Path = PROFILE_PATH,
        static: StaticDataCatalog | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.model_path = Path(model_path)
        self.model_meta_path = self.model_path.with_suffix(".json")
        self.profile = load_profile(profile_path)
        migrated = DraftDatabase(self.database_path)
        migrated.close()
        self.static = static or StaticDataCatalog.load(allow_download=True)
        ml = self.profile.get("machine_learning", {})
        self.patch_half_life = float(ml.get("patch_half_life", 3.0))
        self.minimum_patch_weight = float(ml.get("minimum_patch_weight", 0.12))

    def _read(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        columns = (
            "match_id, participant_id, team_id, champion_id, champion_name, role, win, "
            "game_duration_seconds, kills, deaths, assists, gold_earned, "
            "physical_damage_to_champions, magic_damage_to_champions, "
            "true_damage_to_champions, damage_to_objectives, damage_taken, "
            "damage_mitigated, time_cc_dealt, vision_score, total_minions_killed, "
            "neutral_minions_killed, turret_takedowns, inhibitor_takedowns, "
            "time_ccing_others, team_damage_percentage, kill_participation, "
            "lane_minions_first_10, takedowns_first_15, solo_kills, skillshots_dodged, "
            "objectives_stolen, challenges_json, summoner1_id, summoner2_id, "
            "primary_style_id, sub_style_id, rune_page_json, stat_perks_json, items_json"
        )
        with sqlite3.connect(self.database_path) as connection:
            participants = pd.read_sql_query(
                f"SELECT {columns} FROM participants "
                "WHERE role IN ('TOP','JUNGLE','MID','ADC','SUPPORT')",
                connection,
            )
            matches = pd.read_sql_query(
                "SELECT match_id, game_creation, game_duration, game_version, "
                "winning_team_id, target_tier FROM matches",
                connection,
            )
        if participants.empty:
            return participants, matches
        participants["role"] = participants["role"].astype(str).str.upper()
        text_columns = {"match_id", "champion_name", "role", "challenges_json", "rune_page_json", "stat_perks_json", "items_json"}
        for column in participants.columns:
            if column not in text_columns:
                participants[column] = pd.to_numeric(participants[column], errors="coerce").fillna(0)
        participants = participants.merge(
            matches[["match_id", "game_creation", "game_version"]],
            on="match_id",
            how="left",
        )
        participants["patch"] = participants["game_version"].map(patch_label)
        participants["patch_weight"] = participants["game_version"].map(
            lambda value: recency_weight(
                value,
                self.static.version,
                half_life_patches=self.patch_half_life,
                minimum_weight=self.minimum_patch_weight,
            )
        )
        return participants, matches

    @staticmethod
    def _adjusted_rate(wins: pd.Series, games: pd.Series, prior: float) -> pd.Series:
        return (wins + 0.5 * prior) / (games + prior)

    @staticmethod
    def _sklearn_safe_csr(matrix: Any) -> Any:
        """Keep compatibility with scikit-learn builds requiring int32 CSR indices.

        V3 no longer trains the primary model with scikit-learn, but preserving
        this helper keeps old analytics/model migrations and regression tests
        safe when a legacy sparse matrix is encountered.
        """
        converted = matrix.tocsr(copy=True)
        converted.indices = converted.indices.astype(np.int32, copy=False)
        converted.indptr = converted.indptr.astype(np.int32, copy=False)
        return converted

    @staticmethod
    def _weighted_group_mean(
        frame: pd.DataFrame,
        group_columns: list[str],
        value_columns: Iterable[str],
        *,
        weight_column: str = "patch_weight",
    ) -> pd.DataFrame:
        base = frame[group_columns + [weight_column, *value_columns]].copy()
        for column in value_columns:
            base[f"__weighted_{column}"] = base[column].astype(float) * base[weight_column].astype(float)
        aggregations: dict[str, tuple[str, str]] = {
            "weighted_games": (weight_column, "sum"),
            "games": (weight_column, "size"),
        }
        for column in value_columns:
            aggregations[f"__sum_{column}"] = (f"__weighted_{column}", "sum")
        output = base.groupby(group_columns, as_index=False).agg(**aggregations)
        denominator = output["weighted_games"].replace(0, np.nan)
        for column in value_columns:
            output[column] = (output[f"__sum_{column}"] / denominator).fillna(0.0)
            output.drop(columns=[f"__sum_{column}"], inplace=True)
        return output

    def _global(self, p: pd.DataFrame) -> pd.DataFrame:
        prior = float(self.profile["shrinkage"]["global"])
        frame = p.assign(weighted_win=p["win"] * p["patch_weight"]).groupby(
            ["role", "champion_id", "champion_name"], as_index=False
        ).agg(
            games=("win", "size"),
            wins=("win", "sum"),
            weighted_games=("patch_weight", "sum"),
            weighted_wins=("weighted_win", "sum"),
            newest_patch=("patch", "max"),
        )
        frame["win_rate"] = frame["wins"] / frame["games"].clip(lower=1)
        frame["patch_win_rate"] = frame["weighted_wins"] / frame["weighted_games"].clip(lower=1e-6)
        frame["adjusted_win_rate"] = self._adjusted_rate(
            frame["weighted_wins"], frame["weighted_games"], prior
        )
        frame["patch_freshness"] = (frame["weighted_games"] / frame["games"].clip(lower=1)).clip(0, 1)
        return frame

    def _synergy(self, p: pd.DataFrame, global_frame: pd.DataFrame) -> pd.DataFrame:
        candidate = p[["match_id", "team_id", "participant_id", "champion_id", "role", "win", "patch_weight"]].rename(
            columns={"participant_id": "candidate_participant", "champion_id": "candidate_id", "role": "candidate_role", "win": "team_win"}
        )
        ally = p[["match_id", "team_id", "participant_id", "champion_id", "role"]].rename(
            columns={"participant_id": "ally_participant", "champion_id": "ally_id", "role": "ally_role"}
        )
        pairs = candidate.merge(ally, on=["match_id", "team_id"])
        pairs = pairs[pairs["candidate_participant"] != pairs["ally_participant"]].copy()
        pairs["weighted_win"] = pairs["team_win"] * pairs["patch_weight"]
        frame = pairs.groupby(
            ["candidate_role", "candidate_id", "ally_role", "ally_id"], as_index=False
        ).agg(
            games=("team_win", "size"), wins=("team_win", "sum"),
            weighted_games=("patch_weight", "sum"), weighted_wins=("weighted_win", "sum"),
        )
        frame["together_win_rate"] = frame["wins"] / frame["games"].clip(lower=1)
        frame["patch_together_win_rate"] = frame["weighted_wins"] / frame["weighted_games"].clip(lower=1e-6)
        candidate_lookup = global_frame[["role", "champion_id", "adjusted_win_rate"]].rename(
            columns={"role": "candidate_role", "champion_id": "candidate_id", "adjusted_win_rate": "candidate_global"}
        )
        ally_lookup = global_frame[["role", "champion_id", "adjusted_win_rate"]].rename(
            columns={"role": "ally_role", "champion_id": "ally_id", "adjusted_win_rate": "ally_global"}
        )
        frame = frame.merge(candidate_lookup, on=["candidate_role", "candidate_id"], how="left")
        frame = frame.merge(ally_lookup, on=["ally_role", "ally_id"], how="left")
        base = (frame["candidate_global"].fillna(0.5) + frame["ally_global"].fillna(0.5)) / 2
        frame["delta"] = frame["together_win_rate"] - base
        frame["patch_delta"] = frame["patch_together_win_rate"] - base
        prior = float(self.profile["shrinkage"]["synergy"])
        frame["adjusted_delta"] = frame["patch_delta"] * frame["weighted_games"] / (frame["weighted_games"] + prior)
        return frame

    def _counter(self, p: pd.DataFrame, global_frame: pd.DataFrame) -> pd.DataFrame:
        candidate = p[["match_id", "team_id", "champion_id", "role", "win", "patch_weight"]].rename(
            columns={"team_id": "candidate_team", "champion_id": "candidate_id", "role": "candidate_role", "win": "candidate_win"}
        )
        enemy = p[["match_id", "team_id", "champion_id", "role"]].rename(
            columns={"team_id": "enemy_team", "champion_id": "enemy_id", "role": "enemy_role"}
        )
        lane = candidate.merge(enemy, on="match_id")
        lane = lane[(lane["candidate_team"] != lane["enemy_team"]) & (lane["candidate_role"] == lane["enemy_role"])].copy()
        lane["weighted_win"] = lane["candidate_win"] * lane["patch_weight"]
        frame = lane.groupby(["candidate_role", "candidate_id", "enemy_id"], as_index=False).agg(
            games=("candidate_win", "size"), wins=("candidate_win", "sum"),
            weighted_games=("patch_weight", "sum"), weighted_wins=("weighted_win", "sum"),
        )
        frame["head_to_head_win_rate"] = frame["wins"] / frame["games"].clip(lower=1)
        frame["patch_head_to_head_win_rate"] = frame["weighted_wins"] / frame["weighted_games"].clip(lower=1e-6)
        lookup = global_frame[["role", "champion_id", "adjusted_win_rate"]].rename(
            columns={"role": "candidate_role", "champion_id": "candidate_id", "adjusted_win_rate": "candidate_global"}
        )
        frame = frame.merge(lookup, on=["candidate_role", "candidate_id"], how="left")
        frame["delta"] = frame["head_to_head_win_rate"] - frame["candidate_global"].fillna(0.5)
        frame["patch_delta"] = frame["patch_head_to_head_win_rate"] - frame["candidate_global"].fillna(0.5)
        prior = float(self.profile["shrinkage"]["counter"])
        frame["adjusted_delta"] = frame["patch_delta"] * frame["weighted_games"] / (frame["weighted_games"] + prior)
        return frame

    @staticmethod
    def _safe_json(value: Any, default: Any) -> Any:
        try:
            return json.loads(value) if isinstance(value, str) else value
        except (TypeError, json.JSONDecodeError):
            return default

    def _builds(self, p: pd.DataFrame) -> pd.DataFrame:
        completed = self.static.completed_item_ids()
        records: list[dict[str, Any]] = []
        for row in p.itertuples(index=False):
            items = [int(x) for x in self._safe_json(row.items_json, []) if int(x or 0) > 0]
            filtered = [item for item in items if not completed or item in completed]
            core = filtered[:3]
            runes = [int(x) for x in self._safe_json(row.rune_page_json, []) if int(x or 0) > 0]
            stat_perks = self._safe_json(row.stat_perks_json, {})
            spell_pair = sorted([int(row.summoner1_id), int(row.summoner2_id)])
            samples = {
                "items": {"ids": core},
                "runes": {"primary_style_id": int(row.primary_style_id), "sub_style_id": int(row.sub_style_id), "perk_ids": runes, "stat_perks": stat_perks},
                "spells": {"ids": spell_pair},
                "loadouts": {
                    "item_ids": core,
                    "primary_style_id": int(row.primary_style_id),
                    "sub_style_id": int(row.sub_style_id),
                    "perk_ids": runes,
                    "stat_perks": stat_perks,
                },
            }
            for kind, sample in samples.items():
                if kind == "items" and not core:
                    continue
                if kind == "runes" and not runes:
                    continue
                if kind == "spells" and not any(spell_pair):
                    continue
                if kind == "loadouts" and (not core or not runes):
                    continue
                signature = json.dumps(sample, sort_keys=True, separators=(",", ":"))
                records.append({
                    "kind": kind, "role": row.role, "champion_id": int(row.champion_id),
                    "signature": signature, "sample_json": json.dumps(sample),
                    "win": int(row.win), "patch_weight": float(row.patch_weight),
                })
        columns = ["kind", "role", "champion_id", "signature", "games", "wins", "weighted_games", "weighted_wins", "win_rate", "adjusted_win_rate", "sample_json", "rank"]
        if not records:
            return pd.DataFrame(columns=columns)
        raw = pd.DataFrame(records)
        raw["weighted_win"] = raw["win"] * raw["patch_weight"]
        frame = raw.groupby(["kind", "role", "champion_id", "signature", "sample_json"], as_index=False).agg(
            games=("win", "size"), wins=("win", "sum"), weighted_games=("patch_weight", "sum"), weighted_wins=("weighted_win", "sum")
        )
        frame["win_rate"] = frame["wins"] / frame["games"].clip(lower=1)
        prior = float(self.profile["shrinkage"]["build"])
        frame["adjusted_win_rate"] = self._adjusted_rate(frame["weighted_wins"], frame["weighted_games"], prior)
        frame["rank_score"] = frame["adjusted_win_rate"] + np.log1p(frame["weighted_games"]) * 0.004
        frame["rank"] = frame.groupby(["kind", "role", "champion_id"])["rank_score"].rank(method="first", ascending=False).astype(int)
        return frame[frame["rank"] <= 5].drop(columns=["rank_score"])

    def _scaling(self, p: pd.DataFrame, global_frame: pd.DataFrame) -> pd.DataFrame:
        buckets = np.select(
            [p["game_duration_seconds"] < 1500, p["game_duration_seconds"] < 2100],
            ["early", "mid"],
            default="late",
        )
        work = p[["role", "champion_id", "win", "patch_weight"]].copy()
        work["bucket"] = buckets
        work["weighted_win"] = work["win"] * work["patch_weight"]
        grouped = work.groupby(["role", "champion_id", "bucket"], as_index=False).agg(
            games=("win", "size"), weighted_games=("patch_weight", "sum"), weighted_wins=("weighted_win", "sum")
        )
        prior = max(12.0, float(self.profile["shrinkage"]["global"]) * 0.5)
        grouped["adjusted_win_rate"] = self._adjusted_rate(grouped["weighted_wins"], grouped["weighted_games"], prior)
        pivot = grouped.pivot_table(index=["role", "champion_id"], columns="bucket", values="adjusted_win_rate").reset_index()
        sample_pivot = grouped.pivot_table(index=["role", "champion_id"], columns="bucket", values="weighted_games").reset_index()
        sample_pivot = sample_pivot.rename(columns={name: f"{name}_weighted_games" for name in ("early", "mid", "late") if name in sample_pivot})
        pivot = pivot.merge(sample_pivot, on=["role", "champion_id"], how="left")
        lookup = global_frame[["role", "champion_id", "adjusted_win_rate"]].rename(columns={"adjusted_win_rate": "global_adjusted"})
        pivot = pivot.merge(lookup, on=["role", "champion_id"], how="left")
        for bucket in ("early", "mid", "late"):
            if bucket not in pivot:
                pivot[bucket] = pivot["global_adjusted"]
            pivot[bucket] = pivot[bucket].fillna(pivot["global_adjusted"]).fillna(0.5)
            pivot[f"{bucket}_strength"] = pivot.groupby("role")[bucket].rank(pct=True).fillna(0.5)
            sample_name = f"{bucket}_weighted_games"
            if sample_name not in pivot:
                pivot[sample_name] = 0.0
            pivot[sample_name] = pivot[sample_name].fillna(0.0)
        return pivot[["role", "champion_id", "early", "mid", "late", "early_strength", "mid_strength", "late_strength", "early_weighted_games", "mid_weighted_games", "late_weighted_games"]]

    def _features(self, p: pd.DataFrame, scaling: pd.DataFrame) -> pd.DataFrame:
        if p.empty:
            return pd.DataFrame()
        duration = (p["game_duration_seconds"].clip(lower=60) / 60.0).astype(float)
        total_damage = (p["physical_damage_to_champions"] + p["magic_damage_to_champions"] + p["true_damage_to_champions"]).replace(0, 1)
        detail = p[["role", "champion_id", "patch_weight"]].copy()
        detail["physical_share"] = p["physical_damage_to_champions"] / total_damage
        detail["magic_share"] = p["magic_damage_to_champions"] / total_damage
        detail["true_share"] = p["true_damage_to_champions"] / total_damage
        detail["cc_per_min"] = p["time_cc_dealt"] / duration
        detail["hard_cc_per_min"] = p["time_ccing_others"] / duration
        detail["objective_per_min"] = (p["damage_to_objectives"] / 1000.0 + p["turret_takedowns"] * 2.0 + p["inhibitor_takedowns"] * 3.0 + p["objectives_stolen"] * 4.0) / duration
        detail["tank_per_min"] = (p["damage_taken"] + p["damage_mitigated"]) / duration / 1000.0
        detail["vision_per_min"] = p["vision_score"] / duration
        detail["cs_per_min"] = (p["total_minions_killed"] + p["neutral_minions_killed"]) / duration
        detail["gold_per_min"] = p["gold_earned"] / duration / 1000.0
        detail["kda"] = (p["kills"] + p["assists"]) / p["deaths"].clip(lower=1)
        detail["kill_participation"] = p["kill_participation"].clip(0, 1)
        detail["team_damage_percentage"] = p["team_damage_percentage"].clip(0, 1)
        detail["mobility_raw"] = (
            p["skillshots_dodged"] / duration
            + p["kill_participation"].clip(0, 1) * 0.7
            + p["solo_kills"] / duration * 0.4
            + p["takedowns_first_15"] / 15.0 * 0.2
        )
        detail["new_field_coverage"] = (
            (p["time_ccing_others"] > 0).astype(float)
            + (p["total_minions_killed"] + p["neutral_minions_killed"] > 0).astype(float)
            + (p["kill_participation"] > 0).astype(float)
            + (p["team_damage_percentage"] > 0).astype(float)
        ) / 4.0

        values = [
            "physical_share", "magic_share", "true_share", "cc_per_min", "hard_cc_per_min",
            "objective_per_min", "tank_per_min", "vision_per_min", "cs_per_min", "gold_per_min",
            "kda", "kill_participation", "team_damage_percentage", "mobility_raw", "new_field_coverage",
        ]
        frame = self._weighted_group_mean(detail, ["role", "champion_id"], values)
        percentile_sources = {
            "cc_per_min": "control_base",
            "hard_cc_per_min": "hard_cc",
            "objective_per_min": "objective",
            "tank_per_min": "frontline",
            "vision_per_min": "vision",
            "cs_per_min": "waveclear",
            "mobility_raw": "mobility",
        }
        for source, target in percentile_sources.items():
            frame[target] = frame.groupby("role")[source].rank(pct=True).fillna(0.5)
        frame["control"] = (0.55 * frame["hard_cc"] + 0.45 * frame["control_base"]).clip(0, 1)
        frame["pick_potential"] = (0.50 * frame["hard_cc"] + 0.30 * frame["mobility"] + 0.20 * frame["vision"]).clip(0, 1)
        frame["engage"] = (0.38 * frame["control"] + 0.25 * frame["frontline"] + 0.20 * frame["mobility"] + 0.17 * frame["pick_potential"]).clip(0, 1)
        frame["disengage"] = (0.42 * frame["control"] + 0.33 * frame["waveclear"] + 0.25 * frame["mobility"]).clip(0, 1)
        frame = frame.merge(scaling, on=["role", "champion_id"], how="left")
        for column in ("early_strength", "mid_strength", "late_strength"):
            frame[column] = frame[column].fillna(0.5)
        sample_confidence = 1.0 - np.exp(-frame["weighted_games"] / 45.0)
        # Old rows still have strong base coverage. New Match-V5 fields gradually
        # raise feature coverage as the watcher enriches the database.
        coverage = (0.62 + 0.38 * frame["new_field_coverage"].clip(0, 1)).clip(0, 1)
        frame["feature_confidence"] = (sample_confidence * coverage).clip(0, 1)
        frame["damage_profile"] = np.select(
            [
                frame["physical_share"] >= 0.70,
                frame["magic_share"] >= 0.70,
                frame["true_share"] >= 0.20,
                (frame["physical_share"] >= 0.28) & (frame["magic_share"] >= 0.28),
            ],
            ["Physical", "Magic", "True", "Mixed"],
            default="Mixed",
        )
        return frame

    def _role_priors(self, p: pd.DataFrame) -> pd.DataFrame:
        grouped = p.groupby(["champion_id", "role"], as_index=False).agg(
            games=("role", "size"), weighted_games=("patch_weight", "sum")
        )
        totals = grouped.groupby("champion_id", as_index=False)["weighted_games"].sum().rename(columns={"weighted_games": "champion_weighted_games"})
        frame = grouped.merge(totals, on="champion_id", how="left")
        smoothing = 0.35
        frame["probability"] = (frame["weighted_games"] + smoothing) / (frame["champion_weighted_games"] + smoothing * len(ROLES))
        return frame

    def _patch_stats(self, p: pd.DataFrame) -> pd.DataFrame:
        work = p.assign(weighted_win=p["win"] * p["patch_weight"])
        frame = work.groupby(["patch", "role", "champion_id"], as_index=False).agg(
            games=("win", "size"), wins=("win", "sum"), weighted_games=("patch_weight", "sum"), weighted_wins=("weighted_win", "sum")
        )
        frame["win_rate"] = frame["wins"] / frame["games"].clip(lower=1)
        frame["weighted_win_rate"] = frame["weighted_wins"] / frame["weighted_games"].clip(lower=1e-6)
        return frame

    @staticmethod
    def _write_table(connection: sqlite3.Connection, frame: pd.DataFrame, name: str, indexes: list[str]) -> None:
        frame.to_sql(name, connection, if_exists="replace", index=False)
        for number, columns in enumerate(indexes):
            connection.execute(f"CREATE INDEX IF NOT EXISTS idx_{name}_{number} ON {name}({columns})")

    def _embedding_table(self, champion_ids: Iterable[int]) -> pd.DataFrame:
        runtime = PortableDraftModel(self.model_path, self.model_meta_path)
        rows: list[dict[str, Any]] = []
        if not runtime.available:
            return pd.DataFrame(columns=["champion_id", "vector_json", "dimension"])
        for champion_id in champion_ids:
            vector = runtime.embedding(int(champion_id))
            if vector is None:
                continue
            rows.append({
                "champion_id": int(champion_id),
                "vector_json": json.dumps([round(float(x), 7) for x in vector]),
                "dimension": int(vector.size),
            })
        return pd.DataFrame(rows)

    def build_all(self) -> BuildReport:
        p, matches = self._read()
        if p.empty:
            raise RuntimeError("The database has no participants. Run scraper.py first.")
        global_frame = self._global(p)
        synergy = self._synergy(p, global_frame)
        counter = self._counter(p, global_frame)
        builds = self._builds(p)
        scaling = self._scaling(p, global_frame)
        features = self._features(p, scaling)
        role_priors = self._role_priors(p)
        patch_stats = self._patch_stats(p)

        neural_report = NeuralTrainingReport(
            False, "unavailable", "not checked", 0, 0, 0.0, 0.25,
            "Neural training was not attempted.",
        )
        try:
            neural_report = train_neural_ensemble(
                p,
                matches,
                features,
                current_patch=self.static.version,
                settings=self.profile.get("machine_learning", {}),
                model_path=self.model_path,
                meta_path=self.model_meta_path,
            )
        except Exception as exc:
            LOGGER.exception("V3 neural training failed; statistical analytics remain available.")
            neural_report = NeuralTrainingReport(
                False, "failed", "unknown", 0, 0, 0.0, 0.25,
                f"{type(exc).__name__}: {exc}",
            )
        embeddings = self._embedding_table(global_frame["champion_id"].unique())

        with sqlite3.connect(self.database_path) as connection:
            self._write_table(connection, global_frame, "analytics_global", ["role,champion_id"])
            self._write_table(connection, synergy, "analytics_synergy", ["candidate_role,candidate_id,ally_role,ally_id"])
            self._write_table(connection, counter, "analytics_counter", ["candidate_role,candidate_id,enemy_id"])
            self._write_table(connection, builds, "analytics_builds", ["kind,role,champion_id,rank"])
            self._write_table(connection, features, "analytics_features", ["role,champion_id"])
            self._write_table(connection, scaling, "analytics_scaling", ["role,champion_id"])
            self._write_table(connection, role_priors, "analytics_role_priors", ["champion_id,role"])
            self._write_table(connection, patch_stats, "analytics_patch", ["patch,role,champion_id"])
            self._write_table(connection, embeddings, "analytics_embeddings", ["champion_id"])
            connection.execute("DROP TABLE IF EXISTS analytics_meta")
            connection.execute("CREATE TABLE analytics_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            meta = {
                "analytics_version": "3.0.5",
                "built_at": datetime.now(UTC).isoformat(),
                "matches": str(matches["match_id"].nunique()),
                "participants": str(len(p)),
                "target_elo": str(self.profile["target_elo"]),
                "static_patch": self.static.version,
                "patch_half_life": str(self.patch_half_life),
                "minimum_patch_weight": str(self.minimum_patch_weight),
                "ml_examples": str(neural_report.examples),
                "ml_matches": str(neural_report.matches),
                "ml_backend": neural_report.backend,
                "ml_device": neural_report.device,
                "ml_validation_accuracy": str(neural_report.validation_accuracy),
                "ml_validation_brier": str(neural_report.validation_brier),
                "ml_trained": str(bool(neural_report.trained)),
                "ml_reason": neural_report.reason,
                "feature_schema": ",".join(CHAMPION_FEATURES),
            }
            connection.executemany("INSERT INTO analytics_meta(key,value) VALUES(?,?)", meta.items())
            connection.commit()

        report = BuildReport(
            matches=int(matches["match_id"].nunique()), participants=len(p),
            global_rows=len(global_frame), synergy_rows=len(synergy), counter_rows=len(counter),
            build_rows=len(builds), feature_rows=len(features), role_prior_rows=len(role_priors),
            patch_rows=len(patch_stats), ml_examples=neural_report.examples,
            neural_trained=neural_report.trained, neural_device=neural_report.device,
        )
        LOGGER.info("Analytics complete: %s", report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build patch-aware analytics and train the v3 model.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    print(AnalyticsBuilder(args.database, args.model).build_all())


if __name__ == "__main__":
    main()
