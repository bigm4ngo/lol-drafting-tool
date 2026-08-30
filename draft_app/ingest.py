"""Sync-ingest module for the League Draft Lab v3.1 Windows app.

Watches ``sync_inbox/`` for ``*.sync.zip`` bundles produced by the Ubuntu
collector's ``sync_export.py``. Each bundle contains a ``delta.db`` (new
matches/participants/bans since the last export) plus optional ``static_data.json``
and ``champion_map.json`` snapshots. The app merges the delta into the main
SQLite database, archives the bundle, and (optionally) triggers a full
analytics/model rebuild so the GPU stays on the Windows machine only.

Broken bundles (truncated transfers, malformed deltas, deltas whose schema
does not fit the local tables) are moved to ``sync_inbox_failed/`` instead of
being retried forever on every app start.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import zipfile
from contextlib import closing
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
FAILED_DIR = PROJECT_ROOT / "sync_inbox_failed"
LAST_INGEST_PATH = DATA_DIR / "ingest_state.json"
_REQUIRED_BUNDLE_MEMBER = "delta.db"
_REQUIRED_DELTA_TABLES = ("matches", "participants", "bans")
_STALE_BUNDLE_DAYS = 14
# A zip younger than this is probably still being copied into the inbox by the
# transfer tool; defer it instead of quarantining on the first failure.
_PARTIAL_TRANSFER_GRACE_SECONDS = 600
_TRANSIENT_MARKERS = ("locked", "busy")


@dataclass
class IngestResult:
    ingests: int = 0
    matches_added: int = 0
    skipped: int = 0
    bundle_paths: list[Path] = field(default_factory=list)
    detail: str = ""


class BundleIngestError(RuntimeError):
    pass


class StructuralBundleError(BundleIngestError):
    """The bundle is permanently unusable and must be quarantined."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _is_transient_sqlite_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


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
        failed_dir: Path = FAILED_DIR,
        last_ingest_path: Path = LAST_INGEST_PATH,
        champion_cache_path: Path = CHAMPION_CACHE_PATH,
        static_cache_path: Path = STATIC_CACHE_PATH,
    ) -> None:
        self.database_path = Path(database_path)
        self.inbox = Path(inbox) if inbox else resolve_inbox()
        self.processed_dir = Path(processed_dir)
        self.failed_dir = Path(failed_dir)
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
    def _bundle_settled(bundle: Path) -> bool:
        """True unless the file was modified very recently (still copying)."""
        try:
            age = time.time() - bundle.stat().st_mtime
        except OSError:
            return True
        return age > _PARTIAL_TRANSFER_GRACE_SECONDS

    @staticmethod
    def _delta_connection(delta_path: Path) -> sqlite3.Connection:
        """Open an extracted delta read-only; an empty file opens as empty DB."""
        return sqlite3.connect(
            f"file:{delta_path}?mode=ro", uri=True, timeout=30
        )

    @classmethod
    def _validate_delta(cls, delta_path: Path, manifest: dict[str, Any]) -> None:
        """Raise StructuralBundleError when the delta cannot be merged.

        Covers truncated/malformed files, deltas without the expected tables
        (0-byte or schema-less files), and deltas that lost their rows because
        a -wal sidecar never made it into the bundle.
        """
        if not delta_path.is_file():
            raise StructuralBundleError("delta.db missing after extraction")
        try:
            with closing(cls._delta_connection(delta_path)) as connection:
                try:
                    row = connection.execute("PRAGMA quick_check").fetchone()
                except sqlite3.DatabaseError as exc:
                    raise StructuralBundleError(
                        f"delta.db is unreadable ({exc})"
                    ) from exc
                check = str(row[0]) if row else "unknown"
                if check != "ok":
                    raise StructuralBundleError(
                        f"delta.db failed integrity check: {check}"
                    )
                names = {
                    str(name)
                    for (name,) in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                missing = [t for t in _REQUIRED_DELTA_TABLES if t not in names]
                if missing:
                    raise StructuralBundleError(
                        f"delta.db is missing table(s): {', '.join(missing)}"
                    )
                expected = int(
                    ((manifest.get("counts") or {}).get("matches")) or 0
                )
                if expected > 0:
                    actual = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM matches"
                        ).fetchone()[0]
                    )
                    if actual == 0:
                        raise StructuralBundleError(
                            "delta.db holds no matches although the manifest "
                            f"lists {expected} (uncheckpointed WAL data was lost)"
                        )
        except sqlite3.OperationalError as exc:
            if _is_transient_sqlite_error(exc):
                raise BundleIngestError(f"delta.db busy: {exc}") from exc
            raise StructuralBundleError(f"delta.db unreadable ({exc})") from exc

    @staticmethod
    def _count_new_in_delta(delta_path: Path, database_path: Path) -> int:
        """Count matches in a delta not already present in the main DB."""
        DraftDatabase(database_path).close()  # ensure the main schema exists
        with closing(sqlite3.connect(database_path, timeout=30)) as dst:
            dst.execute("ATTACH DATABASE ? AS src", (str(delta_path),))
            try:
                return int(
                    dst.execute(
                        "SELECT COUNT(*) FROM src.matches m "
                        "LEFT JOIN main.matches mm ON mm.match_id = m.match_id "
                        "WHERE mm.match_id IS NULL"
                    ).fetchone()[0]
                )
            finally:
                dst.execute("DETACH DATABASE src")

    @staticmethod
    def _common_columns(dst: sqlite3.Connection, table: str) -> list[str]:
        """Columns present in BOTH main and src, in main-table order.

        Copying positionally with ``SELECT *`` silently misaligns values when
        the collector's delta schema and the local database were migrated
        through different app versions (e.g. a REAL stat landing inside
        ``items_json``). Matching by name is version-proof.
        """
        main_columns = [
            str(row[1]) for row in dst.execute(f"PRAGMA main.table_info({table})")
        ]
        src_columns = {
            str(row[1]) for row in dst.execute(f"PRAGMA src.table_info({table})")
        }
        return [column for column in main_columns if column in src_columns]

    def _merge_delta(self, delta_path: Path) -> int:
        """Merge a delta database into the main DB. Returns matches added."""
        if not delta_path.is_file():
            raise StructuralBundleError("delta.db missing in bundle")
        added = self._count_new_in_delta(delta_path, self.database_path)
        DraftDatabase(self.database_path).close()  # ensure schema + indexes
        with closing(sqlite3.connect(self.database_path, timeout=30)) as dst:
            dst.execute("PRAGMA journal_mode=WAL")
            dst.execute("ATTACH DATABASE ? AS src", (str(delta_path),))
            with dst:
                for table in _REQUIRED_DELTA_TABLES:
                    columns = self._common_columns(dst, table)
                    if not columns:
                        raise StructuralBundleError(
                            f"delta.db table '{table}' shares no columns with "
                            "the local schema"
                        )
                    column_sql = ", ".join(f'"{column}"' for column in columns)
                    dst.execute(
                        f"INSERT OR IGNORE INTO {table} ({column_sql}) "
                        f"SELECT {column_sql} FROM src.{table}"
                    )
        return added

    def ingest_bundle(self, bundle: Path) -> IngestResult:
        """Ingest a single bundle atomically into the local DB.

        Permanently broken bundles are quarantined to ``sync_inbox_failed/``
        so they are not retried on every poll; bundles that may still be
        mid-transfer (or that hit a transient SQLite lock) stay in the inbox
        for the next pass. Failures never raise: the aggregate ``detail``
        carries a one-line reason for the Data Watcher console.
        """
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
                    raise StructuralBundleError("bundle missing delta.db")
                manifest_raw = (
                    archive.read("manifest.json")
                    if "manifest.json" in members
                    else b"{}"
                )
                manifest = (
                    _safe_json(manifest_raw.decode("utf-8", errors="replace"))
                    or {}
                )
                archive.extractall(work_dir)
            delta = work_dir / "delta.db"
            self._validate_delta(delta, manifest)
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
        except zipfile.BadZipFile as exc:
            if self._bundle_settled(bundle):
                result.detail = self._quarantine_bundle(
                    bundle, f"corrupt or truncated zip: {exc}"
                )
            else:
                result.detail = (
                    f"deferring {bundle.name}: transfer may still be in "
                    f"progress ({exc})"
                )
            return result
        except StructuralBundleError as exc:
            result.detail = self._quarantine_bundle(bundle, str(exc))
            return result
        except BundleIngestError as exc:
            # Transient by definition (busy delta, mid-transfer file, ...).
            result.detail = f"deferring {bundle.name}: {exc}"
            return result
        except sqlite3.OperationalError as exc:
            if _is_transient_sqlite_error(exc):
                result.detail = f"deferring {bundle.name}: database busy ({exc})"
            else:
                result.detail = self._quarantine_bundle(
                    bundle, f"delta rejected by SQLite: {exc}"
                )
            return result
        except sqlite3.DatabaseError as exc:
            result.detail = self._quarantine_bundle(
                bundle, f"delta rejected by SQLite: {exc}"
            )
            return result
        except OSError as exc:
            result.detail = f"deferring {bundle.name}: {exc}"
            return result
        finally:
            try:
                import shutil

                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass

    def _quarantine_bundle(self, bundle: Path, reason: str) -> str:
        """Move a permanently broken bundle out of the inbox. Returns detail."""
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        dest = self.failed_dir / bundle.name
        try:
            if dest.exists():
                dest.unlink()
            bundle.replace(dest)
            (self.failed_dir / f"{bundle.name}.error.txt").write_text(
                f"quarantined at: {_now()}\nreason: {reason}\n",
                encoding="utf-8",
            )
            LOGGER.error(
                "Quarantined broken bundle %s -> %s (%s)",
                bundle.name,
                self.failed_dir,
                reason,
            )
            return f"quarantined {bundle.name}: {reason}"
        except OSError as exc:
            LOGGER.error(
                "Could not quarantine broken bundle %s (%s); leaving it in "
                "the inbox for a later retry.",
                bundle.name,
                exc,
            )
            return f"quarantine failed for {bundle.name}: {reason}"

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
                # Single line per problem bundle; details come from
                # ingest_bundle (quarantine/deferral) so failures stay readable.
                LOGGER.warning("%s", r.detail)
        if aggregate.ingests:
            aggregate.detail = (
                f"ingested {aggregate.ingests} bundle(s), "
                f"{aggregate.matches_added} new matches"
            )
        return aggregate
