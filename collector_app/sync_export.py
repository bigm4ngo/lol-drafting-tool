"""Incremental data export for the Ubuntu collector -> Windows draft app.

The headless collector stores raw matches into the shared SQLite schema but
never builds analytics or trains the neural model. After each batch of new
matches is stored, this module writes a ZIP bundle into ``outbox/`` containing:

  - ``delta.db``         new matches/participants/bans since the last export
  - ``static_data.json`` current Data Dragon static snapshot (items/runes/spells)
  - ``champion_map.json`` current champion catalog
  - ``manifest.json``    counts, patch distribution, versions, exported_at

The Windows draft app watches its ``sync_inbox/`` directory and ingests these
bundles after a tailscale push (user-managed). Re-export produces deterministic,
incremental deltas keyed on the ``matches`` table rowid, so nothing is ever
re-sent and no analytics/model work runs on the server.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import zipfile
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from data_dragon_maps import DEFAULT_CACHE_PATH as CHAMPION_CACHE_PATH
from patch_utils import patch_label
from runtime_paths import DATA_DIR, PROJECT_ROOT
from scraper import DEFAULT_DB_PATH, DraftDatabase
from static_data import CACHE_PATH as STATIC_CACHE_PATH

LOGGER = logging.getLogger("sync_export")

OUTBOX_DIR = PROJECT_ROOT / "outbox"
SYNC_STATE_PATH = DATA_DIR / "sync_state.json"
BUNDLE_SUFFIX = ".sync.zip"
_BUNDLE_MAX_AGE_DAYS = 30

EMPTY_STATE: dict[str, Any] = {
    "last_match_rowid": 0,
    "last_exported_at": "",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_state(path: Path = SYNC_STATE_PATH) -> dict[str, Any]:
    if not path.is_file():
        return dict(EMPTY_STATE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return dict(EMPTY_STATE)
    except (OSError, ValueError):
        return dict(EMPTY_STATE)
    state = dict(EMPTY_STATE)
    state.update(payload)
    try:
        state["last_match_rowid"] = int(state.get("last_match_rowid", 0))
    except (TypeError, ValueError):
        state["last_match_rowid"] = 0
    return state


def _write_state(state: dict[str, Any], path: Path = SYNC_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _verify_delta(delta_path: Path, expected_matches: int) -> None:
    """Fail loudly instead of shipping an unusable delta in a bundle.

    A broken delta would otherwise be rejected (or silently lose matches) on
    the Windows side while the watermark already advanced past those rows,
    making the data unrecoverable. Checks integrity, required tables, the
    committed row count, and that the file is in rollback-journal mode so all
    copied rows live in the main file that gets zipped.
    """
    connection = sqlite3.connect(delta_path, timeout=30)
    try:
        try:
            if str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower() != "delete":
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                connection.execute("PRAGMA journal_mode=DELETE")
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"could not finalise delta journal state: {exc}"
            ) from exc
    finally:
        connection.close()

    connection = sqlite3.connect(delta_path, timeout=30)
    try:
        check_row = connection.execute("PRAGMA quick_check").fetchone()
        check = str(check_row[0]) if check_row else "unknown"
        if check != "ok":
            raise RuntimeError(f"exported delta failed integrity check: {check}")
        for table in ("matches", "participants", "bans"):
            try:
                connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
            except sqlite3.Error as exc:
                raise RuntimeError(
                    f"exported delta is missing the '{table}' table: {exc}"
                ) from exc
        copied_count = int(
            connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        )
        mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
    finally:
        connection.close()
    if mode.lower() != "delete":
        raise RuntimeError(
            f"exported delta is in journal mode '{mode}'; rows may not be in "
            "the main database file"
        )
    if expected_matches > 0 and copied_count != expected_matches:
        raise RuntimeError(
            f"exported delta holds {copied_count} matches, expected {expected_matches}"
        )


def _copy_delta_slice(src_path: Path, dst_path: Path, after_rowid: int) -> int:
    """Copy matches/participants/bans newer than ``after_rowid`` into ``dst_path``.

    ``dst_path`` is created with the standard schema first, then populated from
    the live database. ``journal_mode=DELETE`` keeps the exported main DB file
    self-contained so the archived copy does not depend on a ``-wal`` sidecar.
    Returns the number of matches copied. The connection is closed explicitly
    (a ``with`` block only ends the transaction, it never closes) and the
    result is verified before the caller zips it.
    """
    DraftDatabase(dst_path).close()  # ensures the full schema exists.
    connection = sqlite3.connect(dst_path, timeout=30)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("ATTACH DATABASE ? AS src", (str(src_path),))
        with connection:
            cursor = connection.execute(
                "INSERT INTO matches SELECT m.* FROM src.matches m "
                "WHERE m.rowid > ?",
                (after_rowid,),
            )
            copied = cursor.rowcount
            connection.execute(
                "INSERT INTO participants "
                "SELECT p.* FROM src.participants p "
                "JOIN src.matches m ON m.match_id = p.match_id "
                "WHERE m.rowid > ?",
                (after_rowid,),
            )
            connection.execute(
                "INSERT INTO bans "
                "SELECT b.* FROM src.bans b "
                "JOIN src.matches m ON m.match_id = b.match_id "
                "WHERE m.rowid > ?",
                (after_rowid,),
            )
    finally:
        connection.close()
    _verify_delta(dst_path, int(copied))
    return int(copied)


def _stats_from_delta(delta_path: Path) -> dict[str, Any]:
    """Collapse counts + patch distribution for the manifest."""
    with closing(sqlite3.connect(delta_path, timeout=30)) as connection:
        def scalar(query: str) -> int:
            return int(connection.execute(query).fetchone()[0] or 0)

        matches = scalar("SELECT COUNT(*) FROM matches")
        participants = scalar("SELECT COUNT(*) FROM participants")
        bans = scalar("SELECT COUNT(*) FROM bans")
        patch_counts: dict[str, int] = {}
        for game_version, count in connection.execute(
            "SELECT game_version, COUNT(*) FROM matches GROUP BY game_version"
        ):
            label = patch_label(game_version)
            patch_counts[label] = patch_counts.get(label, 0) + int(count)
    return {
        "matches": matches,
        "participants": participants,
        "bans": bans,
        "patches": dict(sorted(patch_counts.items(), reverse=True)),
    }


def _safe_json_file(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _stale_bundle_name(exported_at: str) -> str:
    stamp = exported_at.replace(":", "").replace("+", "_").replace(".", "_")
    return f"sync-{stamp}{BUNDLE_SUFFIX}"


@dataclass
class ExportResult:
    exported: bool
    bundle_path: Path | None = None
    matches_exported: int = 0
    detail: str = ""
    state: dict[str, Any] = field(default_factory=dict)


class SyncExporter:
    """Writes incremental sync bundles after the collector stores new matches."""

    def __init__(
        self,
        *,
        database_path: Path = DEFAULT_DB_PATH,
        export_dir: Path = OUTBOX_DIR,
        state_path: Path = SYNC_STATE_PATH,
        static_cache_path: Path = STATIC_CACHE_PATH,
        champion_cache_path: Path = CHAMPION_CACHE_PATH,
        include_static: bool = True,
    ) -> None:
        self.database_path = Path(database_path)
        self.export_dir = Path(export_dir)
        self.state_path = Path(state_path)
        self.static_cache_path = Path(static_cache_path)
        self.champion_cache_path = Path(champion_cache_path)
        self.include_static = include_static

    def _read_state(self) -> dict[str, Any]:
        return _read_state(self.state_path)

    def _max_match_rowid(self) -> int:
        if not self.database_path.is_file():
            return 0
        try:
            with closing(sqlite3.connect(self.database_path, timeout=30)) as connection:
                return int(
                    connection.execute(
                        "SELECT COALESCE(MAX(rowid), 0) FROM matches"
                    ).fetchone()[0]
                )
        except sqlite3.Error:
            LOGGER.exception("Could not read match rowid watermark.")
            return 0

    def export(self) -> ExportResult:
        """Build and export a delta bundle if new matches exist."""
        state = self._read_state()
        last_rowid = int(state.get("last_match_rowid", 0))
        if not self.database_path.is_file():
            return ExportResult(exported=False, state=state, detail="no local database yet")
        max_rowid = self._max_match_rowid()
        if max_rowid <= last_rowid:
            return ExportResult(exported=False, state=state, detail="no new matches since last export")

        self.export_dir.mkdir(parents=True, exist_ok=True)
        exported_at = _now()
        delta_db = self.export_dir / f".delta-{exported_at.replace(':', '').replace('.', '')}.tmp.db"
        try:
            copied = _copy_delta_slice(self.database_path, delta_db, last_rowid)
            if copied <= 0:
                return ExportResult(exported=False, state=state, detail="export produced no matches")
            stats = _stats_from_delta(delta_db)

            manifest = {
                "format_version": 1,
                "kind": "lol-draft-delta",
                "exported_at": exported_at,
                "after_match_rowid": last_rowid,
                "through_match_rowid": max_rowid,
                "counts": stats,
                "static_version": self._static_version(),
                "champion_count": self._champion_count(),
            }

            bundle_name = _stale_bundle_name(exported_at)
            bundle_path = self.export_dir / bundle_name
            with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.write(delta_db, arcname="delta.db")
                if self.include_static:
                    self._archive_json(archive, "static_data.json", self.static_cache_path)
                    self._archive_json(archive, "champion_map.json", self.champion_cache_path)
                archive.writestr(
                    "manifest.json",
                    json.dumps(manifest, indent=2),
                )
            # A partially written zip must never leave the outbox: verify the
            # archive before the watermark advances past these matches.
            with closing(zipfile.ZipFile(bundle_path)) as archive:
                corrupt_member = archive.testzip()
                if corrupt_member is not None:
                    raise RuntimeError(f"bundle member failed CRC check: {corrupt_member}")
                if "delta.db" not in archive.namelist():
                    raise RuntimeError("bundle was written without delta.db")

            state["last_match_rowid"] = max_rowid
            state["last_exported_at"] = exported_at
            _write_state(state, self.state_path)
            return ExportResult(
                exported=True,
                bundle_path=bundle_path,
                matches_exported=copied,
                detail=(f"exported {copied} new matches to {bundle_path.name}"),
                state=state,
            )
        except Exception:
            LOGGER.exception("Incremental export failed.")
            # Never leave a half-written bundle behind for the transfer to pick up.
            try:
                bundle_path.unlink(missing_ok=True)
            except (OSError, NameError, UnboundLocalError):
                pass
            return ExportResult(exported=False, state=state, detail="export failed")
        finally:
            try:
                delta_db.unlink(missing_ok=True)
            except OSError:
                pass

    def _static_version(self) -> str:
        content = _safe_json_file(self.static_cache_path)
        if not content:
            return ""
        try:
            return str(json.loads(content).get("version", ""))
        except (ValueError, TypeError):
            return ""

    def _champion_count(self) -> int:
        content = _safe_json_file(self.champion_cache_path)
        if not content:
            return 0
        try:
            data = json.loads(content)
        except (ValueError, TypeError):
            return 0
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            return len(data.get("champions", {})) if isinstance(data.get("champions"), dict) else 0
        return 0

    @staticmethod
    def _archive_json(archive: zipfile.ZipFile, arcname: str, path: Path) -> None:
        content = _safe_json_file(path)
        if content:
            archive.writestr(arcname, content)
