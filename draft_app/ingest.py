"""Sync-ingest module for the League Draft Lab v3.1 Windows app.

Watches ``sync_inbox/`` for ``*.sync.zip`` bundles produced by the Ubuntu
collector's ``sync_export.py``. Each bundle contains a ``delta.db`` (new
matches/participants/bans since the last export) plus optional ``static_data.json``
and ``champion_map.json`` snapshots. The app merges the delta into the main
SQLite database, archives the bundle, and (optionally) triggers a full
analytics/model rebuild so the GPU stays on the Windows machine only.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from config_manager import load_profile
from data_dragon_maps import DEFAULT_CACHE_PATH as CHAMPION_CACHE_PATH
from runtime_paths import DATA_DIR, PROJECT_ROOT
from scraper import DEFAULT_DB_PATH, DraftDatabase
from static_data import CACHE_PATH as STATIC_CACHE_PATH

LOGGER = logging.getLogger("ingest")

DEFAULT_INBOX = PROJECT_ROOT / "sync_inbox"
PROCESSED_DIR = PROJECT_ROOT / "sync_inbox_processed"
LAST_INGEST_PATH = DATA_DIR / "ingest_state.json"
_REQUIRED_BUNDLE_MEMBER = "delta.db"
_STALE_BUNDLE_DAYS = 14


@dataclass
class IngestResult:
    ingests: int = 0
    matches_added: int = 0
    skipped: int = 0
    bundle_paths: list[Path] = field(default_factory=list)
    detail: str = ""


class BundleIngestError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_profile_sync() -> dict[str, Any]:
    return load_profile().get("sync", {})


def resolve_inbox() -> Path:
    cfg = _load_profile_sync()
    candidate = str(cfg.get("inbox_dir", "") or "").strip()
    if candidate:
        path = Path(candidate).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    return DEFAULT_INBOX


def resolve_pending_build() -> bool:
    return bool(_load_profile_sync().get("enable_auto_rebuild", True))


def _safe_json(content: str | None) -> dict[str, Any] | None:
    if not content:
        return None
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except ValueError:
        return None


def replacing_static_snapshot(bundle: Path) -> bool:
    """Return True if the bundle embeds static/champion snapshots."""
    try:
        with zipfile.ZipFile(bundle) as archive:
            return (
                "static_data.json" in archive.namelist()
                and "champion_map.json" in archive.namelist()
            )
    except (zipfile.BadZipFile, OSError):
        return False


class SyncIngester:
    """Merge collector delta bundles into the local database."""

    def __init__(
        self,
        *,
        database_path: Path = DEFAULT_DB_PATH,
        inbox: Path | None = None,
        processed_dir: Path = PROCESSED_DIR,
        last_ingest_path: Path = LAST_INGEST_PATH,
        champion_cache_path: Path = CHAMPION_CACHE_PATH,
        static_cache_path: Path = STATIC_CACHE_PATH,
    ) -> None:
        self.database_path = Path(database_path)
        self.inbox = Path(inbox) if inbox else resolve_inbox()
        self.processed_dir = Path(processed_dir)
        self.last_ingest_path = Path(last_ingest_path)
        self.champion_cache_path = Path(champion_cache_path)
        self.static_cache_path = Path(static_cache_path)

    def _read_last_ingest(self) -> dict[str, Any]:
        if not self.last_ingest_path.is_file():
            return {"last_bundle": "", "last_ingested_at": ""}
        try:
            data = json.loads(self.last_ingest_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_last_ingest(self, bundle_name: str) -> None:
        self.last_ingest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_bundle": bundle_name,
            "last_ingested_at": _now(),
        }
        try:
            self.last_ingest_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except OSError:
            LOGGER.exception("Could not write ingest state.")

    def pending_bundles(self) -> list[Path]:
        if not self.inbox.is_dir():
            return []
        return sorted(
            p for p in self.inbox.glob("*.sync.zip") if p.is_file()
        )

    @staticmethod
    def _count_new_in_delta(delta_path: Path, database_path: Path) -> int:
        """Count matches in a delta not already present in the main DB."""
        DraftDatabase(database_path).close()  # ensure the main schema exists
        with sqlite3.connect(database_path, timeout=30) as dst:
            dst.execute("ATTACH DATABASE ? AS src", (str(delta_path),))
            return int(
                dst.execute(
                    "SELECT COUNT(*) FROM src.matches m "
                    "LEFT JOIN main.matches mm ON mm.match_id = m.match_id "
                    "WHERE mm.match_id IS NULL"
                ).fetchone()[0]
            )

    def _merge_delta(self, delta_path: Path) -> int:
        """Merge a delta database into the main DB. Returns matches added."""
        if not delta_path.is_file():
            raise BundleIngestError("delta.db missing in bundle")
        added = self._count_new_in_delta(delta_path, self.database_path)
        DraftDatabase(self.database_path).close()  # ensure schema + indexes
        with sqlite3.connect(self.database_path, timeout=30) as dst:
            dst.execute("PRAGMA journal_mode=WAL")
            dst.execute(f"ATTACH DATABASE ? AS src", (str(delta_path),))
            with dst:
                dst.execute("INSERT OR IGNORE INTO matches SELECT * FROM src.matches")
                dst.execute(
                    "INSERT OR IGNORE INTO participants SELECT * FROM src.participants"
                )
                dst.execute("INSERT OR IGNORE INTO bans SELECT * FROM src.bans")
        return added

    def ingest_bundle(self, bundle: Path) -> IngestResult:
        """Ingest a single bundle atomically into the local DB."""
        result = IngestResult()
        if not bundle.is_file():
            result.detail = "bundle not found"
            return result
        work_dir = self.processed_dir / f".ingest-{bundle.stem}"
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(bundle) as archive:
                members = archive.namelist()
                if _REQUIRED_BUNDLE_MEMBER not in members:
                    result.detail = "bundle missing delta.db"
                    return result
                manifest_raw = archive.read("manifest.json") if "manifest.json" in members else b"{}"
                manifest = _safe_json(manifest_raw.decode("utf-8", errors="replace")) or {}
                archive.extractall(work_dir)
            delta = work_dir / "delta.db"
            added = self._merge_delta(delta)
            # Apply static snapshots if present (current patch champion/static data).
            self._apply_static_snapshots(work_dir)
            self._write_last_ingest(bundle.name)
            self._archive_bundle(bundle, work_dir)
            result.ingests = 1
            result.matches_added = added
            result.bundle_paths.append(bundle)
            result.detail = (
                f"ingested {bundle.name} ({added} new matches)"
            )
            return result
        except (BundleIngestError, zipfile.BadZipFile, sqlite3.Error, OSError) as exc:
            result.detail = f"ingest failed for {bundle.name}: {exc}"
            LOGGER.exception("Bundle ingest failed: %s", bundle)
            return result
        finally:
            try:
                import shutil

                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass

    def _apply_static_snapshots(self, work_dir: Path) -> None:
        static_path = work_dir / "static_data.json"
        champion_path = work_dir / "champion_map.json"
        if static_path.is_file():
            try:
                self.static_cache_path.parent.mkdir(parents=True, exist_ok=True)
                static_path.replace(self.static_cache_path)
                LOGGER.info("Updated static_data.json from bundle snapshot.")
            except OSError:
                LOGGER.exception("Could not replace static_data.json snapshot.")
        if champion_path.is_file():
            try:
                self.champion_cache_path.parent.mkdir(parents=True, exist_ok=True)
                champion_path.replace(self.champion_cache_path)
                LOGGER.info("Updated champion_map.json from bundle snapshot.")
            except OSError:
                LOGGER.exception("Could not replace champion_map.json snapshot.")

    def _archive_bundle(self, bundle: Path, work_dir: Path) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        dest = self.processed_dir / bundle.name
        try:
            if dest.exists():
                dest.unlink()
            bundle.replace(dest)
        except OSError:
            LOGGER.exception("Could not archive processed bundle %s", bundle)

    def ingest_all(self) -> IngestResult:
        """Ingest every pending bundle in order, returning aggregate counts."""
        aggregate = IngestResult()
        for bundle in self.pending_bundles():
            r = self.ingest_bundle(bundle)
            aggregate.ingests += r.ingests
            aggregate.matches_added += r.matches_added
            aggregate.skipped += r.skipped
            if r.bundle_paths:
                aggregate.bundle_paths.extend(r.bundle_paths)
            if r.detail and r.ingests == 0 and r.matches_added == 0:
                LOGGER.warning("%s", r.detail)
        if aggregate.ingests:
            aggregate.detail = (
                f"ingested {aggregate.ingests} bundle(s), "
                f"{aggregate.matches_added} new matches"
            )
        return aggregate
