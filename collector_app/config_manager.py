"""Atomic profile and secret-file management for the desktop application."""

from __future__ import annotations

import json
import logging
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from dotenv import dotenv_values

from runtime_paths import EXECUTABLE_DIR, PROJECT_ROOT
PROFILE_PATH = PROJECT_ROOT / "config_profile.json"
ENV_PATH = PROJECT_ROOT / "config.env"
LOGGER = logging.getLogger("config_manager")


def _legacy_env_candidates() -> tuple[Path, ...]:
    """Return old per-launch config locations used before v2.4.4."""
    candidates = [
        PROJECT_ROOT / "dist" / "LeagueDraftLab" / "config.env",
        EXECUTABLE_DIR / "config.env",
    ]
    unique: list[Path] = []
    for candidate in candidates:
        try:
            same_as_target = candidate.resolve() == ENV_PATH.resolve()
        except OSError:
            same_as_target = candidate == ENV_PATH
        if not same_as_target and candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _read_key_file(path: Path) -> str:
    return str(dotenv_values(path).get("RIOT_API_KEY", "") or "").strip()


def _retire_legacy_env_files() -> None:
    """Rename obsolete per-EXE key files so they cannot be mistaken as active."""
    for candidate in _legacy_env_candidates():
        if not candidate.is_file():
            continue
        retired = candidate.with_name("config.env.legacy-v2.4.4.bak")
        try:
            if retired.exists():
                candidate.unlink()
            else:
                candidate.replace(retired)
            LOGGER.info("Retired obsolete API key file %s as %s", candidate, retired)
        except OSError:
            LOGGER.warning("Could not retire obsolete API key file %s", candidate, exc_info=True)


def reconcile_legacy_api_key(path: Path = ENV_PATH) -> Path | None:
    """Migrate the newest old EXE/source key into the one canonical file.

    V2.4.1 copied ``config.env`` into ``dist/LeagueDraftLab``. Users could then
    update either copy through the GUI, causing source mode and EXE mode to use
    different credentials. On the first v2.4.4 read/build, compare the canonical
    file with legacy copies and preserve the most recently edited valid-looking
    key. The replaced canonical file is backed up once for recovery.
    """
    candidates = [path, *_legacy_env_candidates()]
    usable: list[tuple[int, int, Path, str]] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        key = _read_key_file(candidate)
        if not key.startswith("RGAPI-"):
            continue
        try:
            modified_ns = candidate.stat().st_mtime_ns
        except OSError:
            modified_ns = 0
        # Prefer the canonical path when timestamps are exactly equal.
        canonical_priority = 1 if candidate == path else 0
        usable.append((modified_ns, canonical_priority, candidate, key))
    if not usable:
        _retire_legacy_env_files()
        return None
    _, _, selected_path, selected_key = max(usable, key=lambda item: (item[0], item[1]))
    current_key = _read_key_file(path) if path.is_file() else ""
    migrated_from: Path | None = None
    if selected_path != path and selected_key != current_key:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            backup = path.with_name("config.env.pre-v2.4.4.bak")
            if not backup.exists():
                try:
                    shutil.copy2(path, backup)
                except OSError:
                    LOGGER.exception("Could not back up the previous canonical config.env.")
        save_api_key(selected_key, path)
        migrated_from = selected_path
        LOGGER.warning(
            "Migrated the newer Riot API key from legacy path %s to shared path %s.",
            selected_path,
            path,
        )
    _retire_legacy_env_files()
    return migrated_from

ROLES = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")
ELO_OPTIONS = (
    "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD",
    "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER",
    "EMERALD+", "DIAMOND+", "MASTER+", "GRANDMASTER+",
)
POOL_CATEGORIES = ("comfort_picks", "pocket_picks", "general_pool")

DEFAULT_PROFILE: dict[str, Any] = {
    "profile_version": "3.0.0",
    "target_elo": "EMERALD",
    "region_platform": "OC1",
    "regional_route": "SEA",
    "queue": "RANKED_SOLO_5x5",
    "queue_id": 420,
    "divisions": ["I", "II", "III", "IV"],
    "restrict_to_pool": False,
    "comfort_picks": {role: [] for role in ROLES},
    "pocket_picks": {role: [] for role in ROLES},
    "general_pool": {role: [] for role in ROLES},
    "weights": {
        "global_win_rate": 1.0,
        "synergy_delta": 1.0,
        "counter_delta": 1.0,
        "composition": 0.65,
        "machine_learning": 0.75,
        "confidence": 0.20,
    },
    "personal_multipliers": {"comfort": 1.25, "pocket": 1.10, "general": 1.0},
    "minimum_samples": {"global": 20, "synergy": 8, "counter": 8, "build": 5},
    "shrinkage": {
        "global": 40.0,
        "synergy": 20.0,
        "counter": 20.0,
        "build": 12.0,
    },
    "composition_targets": {
        "damage_balance": 1.0,
        "frontline": 0.65,
        "control": 0.55,
        "objective": 0.45,
        "engage": 0.45,
        "disengage": 0.30,
        "pick_potential": 0.42,
        "waveclear": 0.32,
        "mobility": 0.20,
        "early_strength": 0.18,
        "mid_strength": 0.22,
        "late_strength": 0.26,
    },
    "machine_learning": {
        "enabled": True,
        "use_gpu": True,
        "mixed_precision": True,
        "ensemble_size": 3,
        "embedding_dimension": 24,
        "role_embedding_dimension": 6,
        "slot_hidden_dimension": 32,
        "hidden_dimension": 64,
        "dropout": 0.12,
        "epochs": 80,
        "early_stopping_patience": 10,
        "batch_size": 256,
        "learning_rate": 0.0025,
        "minimum_training_matches": 250,
        "patch_half_life": 3.0,
        "minimum_patch_weight": 0.12,
        "role_inference_minimum_probability": 0.002,
    },
    "scraper": {
        "players_per_run": 80,
        "matches_per_player": 15,
        "max_matches_per_run": 600,
        "max_concurrent_requests": 4,
        "min_game_duration_seconds": 900,
        "request_timeout_seconds": 20,
        "max_retries": 5,
    },
    "background_collector": {
        "enabled": True,
        "minimum_poll_seconds": 45,
        "maximum_poll_seconds": 300,
        "player_refresh_minutes": 30,
        "players_per_poll": 20,
        "maximum_backoff_minutes": 20,
        "rebuild_analytics_each_batch": True,
    },
    "sync": {
        "outbox_dir": "",
        "inbox_dir": "",
        "poll_interval_seconds": 300,
        "enable_pc_ingest": True,
        "enable_auto_rebuild": True,
        "max_delta_matches_per_ingest": 5000,
    },

    "ui": {
        "poll_interval_ms": 40,
        "top_n": 10,
        "ban_top_n": 5,
        "include_hover_intents_on_board": True,
        "window_width": 1580,
        "window_height": 940,
        "show_confidence": True,
        "show_explanations": True,
        "embedding_neighbors": 8,
    },
}


def _deep_merge(default: Any, supplied: Any) -> Any:
    if isinstance(default, dict) and isinstance(supplied, Mapping):
        merged = deepcopy(default)
        for key, value in supplied.items():
            merged[key] = _deep_merge(default.get(key), value) if key in default else value
        return merged
    return deepcopy(supplied)


def _normalise_pool(value: Any) -> dict[str, list[str]]:
    """Accept legacy flat arrays as well as role-keyed pool dictionaries."""
    result = {role: [] for role in ROLES}
    if isinstance(value, Mapping):
        for role in ROLES:
            entries = value.get(role, [])
            if isinstance(entries, list):
                result[role] = [str(item).strip() for item in entries if str(item).strip()]
    elif isinstance(value, list):
        # Legacy pools had no role information. Keep them available for every role.
        entries = [str(item).strip() for item in value if str(item).strip()]
        for role in ROLES:
            result[role] = list(entries)
    return result


def validate_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    source_version = str(profile.get("profile_version", ""))
    legacy_profile = not source_version
    merged = _deep_merge(DEFAULT_PROFILE, profile)
    # v2.3 shipped with five visible picks. Existing untouched profiles migrate
    # to the v2.4 default of ten; later user choices are preserved.
    if legacy_profile and int(merged.get("ui", {}).get("top_n", 5)) == 5:
        merged["ui"]["top_n"] = 10
    merged["profile_version"] = "3.0.0"
    merged["target_elo"] = str(merged["target_elo"]).strip().upper()
    if merged["target_elo"] not in ELO_OPTIONS:
        raise ValueError(
            f"Unsupported target Elo {merged['target_elo']!r}. "
            f"Choose one of: {', '.join(ELO_OPTIONS)}"
        )
    merged["region_platform"] = str(merged["region_platform"]).upper()
    merged["regional_route"] = str(merged["regional_route"]).upper()
    merged["queue_id"] = int(merged["queue_id"])
    merged["restrict_to_pool"] = bool(merged["restrict_to_pool"])
    for category in POOL_CATEGORIES:
        merged[category] = _normalise_pool(merged.get(category))

    for group in ("weights", "personal_multipliers", "shrinkage", "composition_targets"):
        merged[group] = {key: float(value) for key, value in merged[group].items()}
    ml = merged["machine_learning"]
    ml["enabled"] = bool(ml.get("enabled", True))
    ml["use_gpu"] = bool(ml.get("use_gpu", True))
    ml["mixed_precision"] = bool(ml.get("mixed_precision", True))
    ml["ensemble_size"] = max(1, min(5, int(ml.get("ensemble_size", 3))))
    ml["embedding_dimension"] = max(8, min(96, int(ml.get("embedding_dimension", 24))))
    ml["role_embedding_dimension"] = max(3, min(32, int(ml.get("role_embedding_dimension", 6))))
    ml["slot_hidden_dimension"] = max(16, min(192, int(ml.get("slot_hidden_dimension", 32))))
    ml["hidden_dimension"] = max(24, min(256, int(ml.get("hidden_dimension", 64))))
    ml["dropout"] = max(0.0, min(0.5, float(ml.get("dropout", 0.12))))
    ml["epochs"] = max(10, min(250, int(ml.get("epochs", 80))))
    ml["early_stopping_patience"] = max(3, min(40, int(ml.get("early_stopping_patience", 10))))
    ml["batch_size"] = max(32, min(2048, int(ml.get("batch_size", 256))))
    ml["learning_rate"] = max(1e-5, min(0.05, float(ml.get("learning_rate", 0.0025))))
    ml["minimum_training_matches"] = max(100, int(ml.get("minimum_training_matches", 250)))
    ml["patch_half_life"] = max(0.5, min(24.0, float(ml.get("patch_half_life", 3.0))))
    ml["minimum_patch_weight"] = max(0.01, min(1.0, float(ml.get("minimum_patch_weight", 0.12))))
    ml["role_inference_minimum_probability"] = max(0.0001, min(0.05, float(ml.get("role_inference_minimum_probability", 0.002))))
    for key in ("comfort", "pocket", "general"):
        multiplier = float(merged["personal_multipliers"][key])
        if not 0.1 <= multiplier <= 5.0:
            raise ValueError(f"{key.title()} multiplier must be between 0.1 and 5.0.")
    merged["minimum_samples"] = {
        key: max(1, int(value)) for key, value in merged["minimum_samples"].items()
    }
    collector = merged["background_collector"]
    # V2.5 replaces the old fixed-cooldown batch scheduler with an automatic,
    # adaptive watcher. Existing profiles are enabled on migration so opening
    # the app is sufficient to resume collection; users can still pause it.
    if source_version not in {"2.5.0", "3.0.0"}:
        collector["enabled"] = True
    collector["enabled"] = bool(collector.get("enabled", True))
    collector["minimum_poll_seconds"] = max(
        15, min(300, int(collector.get("minimum_poll_seconds", 45)))
    )
    collector["maximum_poll_seconds"] = max(
        collector["minimum_poll_seconds"],
        min(1800, int(collector.get("maximum_poll_seconds", 300))),
    )
    collector["player_refresh_minutes"] = max(
        5, min(240, int(collector.get("player_refresh_minutes", 30)))
    )
    collector["players_per_poll"] = max(
        5, min(100, int(collector.get("players_per_poll", 20)))
    )
    collector["maximum_backoff_minutes"] = max(
        1, min(120, int(collector.get("maximum_backoff_minutes", 20)))
    )
    collector["rebuild_analytics_each_batch"] = bool(
        collector.get("rebuild_analytics_each_batch", True)
    )
    collector.pop("interval_minutes", None)
    if "sync" in merged and isinstance(merged["sync"], Mapping):
        merged["sync"] = _deep_merge({
            "outbox_dir": "",
            "inbox_dir": "",
            "poll_interval_seconds": 300,
            "enable_pc_ingest": True,
            "enable_auto_rebuild": True,
            "max_delta_matches_per_ingest": 5000,
        }, merged["sync"])
        for key in ("outbox_dir", "inbox_dir"):
            merged["sync"][key] = str(merged["sync"].get(key, "") or "").strip()
        merged["sync"]["poll_interval_seconds"] = max(30, int(merged["sync"].get("poll_interval_seconds", 300)))
        merged["sync"]["enable_pc_ingest"] = bool(merged["sync"].get("enable_pc_ingest", True))
        merged["sync"]["enable_auto_rebuild"] = bool(merged["sync"].get("enable_auto_rebuild", True))
        merged["sync"]["max_delta_matches_per_ingest"] = max(100, int(merged["sync"].get("max_delta_matches_per_ingest", 5000)))
    merged["ui"]["poll_interval_ms"] = max(20, min(250, int(
        merged["ui"].get("poll_interval_ms", 40)
    )))
    merged["ui"]["top_n"] = max(1, min(20, int(merged["ui"].get("top_n", 10))))
    merged["ui"]["ban_top_n"] = max(1, min(10, int(merged["ui"].get("ban_top_n", 5))))
    merged["ui"]["show_confidence"] = bool(merged["ui"].get("show_confidence", True))
    merged["ui"]["show_explanations"] = bool(merged["ui"].get("show_explanations", True))
    merged["ui"]["embedding_neighbors"] = max(3, min(20, int(merged["ui"].get("embedding_neighbors", 8))))
    return merged


def load_profile(path: Path = PROFILE_PATH) -> dict[str, Any]:
    if not path.exists():
        profile = deepcopy(DEFAULT_PROFILE)
        save_profile(profile, path)
        return profile
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("config_profile.json must contain one JSON object.")
    return validate_profile(payload)


def save_profile(profile: Mapping[str, Any], path: Path = PROFILE_PATH) -> dict[str, Any]:
    validated = validate_profile(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(validated, indent=2), encoding="utf-8")
    temporary.replace(path)
    return validated


def read_api_key(path: Path = ENV_PATH) -> str:
    if path == ENV_PATH:
        reconcile_legacy_api_key(path)
    return _read_key_file(path)


def api_key_fingerprint(api_key: str | None = None, path: Path = ENV_PATH) -> str:
    """Return a non-secret identifier for the currently saved Riot key."""
    value = (api_key if api_key is not None else read_api_key(path)).strip()
    if not value:
        return "missing"
    if len(value) <= 10:
        return "saved"
    return f"{value[:6]}…{value[-4:]}"


def save_api_key(api_key: str, path: Path = ENV_PATH) -> Path:
    """Atomically save and verify the Riot key in the canonical config file.

    The returned path is the exact file written. A read-back check prevents the
    GUI from claiming success when Windows permissions, antivirus, or a stale
    packaged path prevented the replacement from taking effect.
    """
    clean = api_key.strip()
    if clean and not clean.startswith("RGAPI-"):
        raise ValueError("Riot development keys normally begin with 'RGAPI-'.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(f"RIOT_API_KEY={clean}\n", encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    temporary.replace(path)
    saved = _read_key_file(path)
    if saved != clean:
        raise OSError(f"API key write verification failed for {path}")
    LOGGER.info("Saved Riot API key %s to %s", api_key_fingerprint(clean), path)
    return path
