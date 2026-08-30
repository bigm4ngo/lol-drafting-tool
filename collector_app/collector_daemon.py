"""Headless Ubuntu data collector daemon for League Draft Lab v3.1.

Separate concern: this process collects raw Ranked Solo matches from the Riot
API and writes them to SQLite. It does NOT build analytics, train a neural
model, or run any CUDA / numpy / pandas heavy work — that stays on the Windows
machine. After each stored batch it writes an incremental ``*.sync.zip`` bundle
into ``outbox/`` for the Windows app to pull.

Run modes:
  python collector_daemon.py            # long-lived watcher (systemd)
  python collector_daemon.py --once     # single batch, then exit
  python collector_daemon.py --export   # export-only (no collection)
  python collector_daemon.py --status   # print current DB count / patch info
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from config_manager import PROFILE_PATH, load_profile
from data_dragon_maps import ChampionCatalog
from runtime_paths import PROJECT_ROOT
from scraper import (
    DEFAULT_DB_PATH,
    DEFAULT_LOG_PATH,
    run_background_match_watcher,
    run_scrape,
)
from sync_export import SyncExporter

LOGGER = logging.getLogger("collector_daemon")

# Static-data auto-refresh interval (the server is rarely checked on, so keep
# the current patch + champion map current automatically).
STATIC_REFRESH_INTERVAL_SECONDS = 6 * 60 * 60  # 6 hours
# Export timing is driven by the watcher callback, which runs after each batch.


def _resolve_export_dir(profile: dict[str, Any]) -> Path:
    sync = profile.get("sync", {}) or {}
    raw = str(sync.get("outbox_dir", "")).strip()
    if not raw:
        return PROJECT_ROOT / "outbox"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def refresh_static_data(force: bool = False) -> None:
    """Keep the current patch + champion map fresh with no user interaction."""
    try:
        StaticDataCatalog = _import_static_data().StaticDataCatalog
        StaticDataCatalog.load(refresh=force)
        catalog = ChampionCatalog.load(refresh=force, allow_download=True)
        LOGGER.info(
            "Static data refreshed (champions=%d).", len(catalog)
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Static-data refresh failed: %s", exc)


def _import_static_data():
    import static_data as module
    return module


def _watcher_callback_factory(export_dir: Path, data_dir: Path) -> Callable[[dict[str, Any]], None]:
    """Build a callback that exports a delta bundle after each stored batch."""
    exporter = SyncExporter(export_dir=export_dir)

    def callback(payload: dict[str, Any]) -> None:
        state = str(payload.get("state", "watching"))
        stored = int(payload.get("stored", 0) or 0)
        LOGGER.info("Watcher status: %s (stored=%d)", state, stored)
        if stored > 0 or state == "scanning":
            result = exporter.export()
            if result.exported:
                LOGGER.info(
                    "Incremental export ready: %s (%d matches)",
                    result.bundle_path,
                    result.matches_exported,
                )

    return callback


def _log_exporter_callback(exporter: SyncExporter) -> Callable[[dict[str, Any]], None]:
    def callback(payload: dict[str, Any]) -> None:
        result = exporter.export()
        if result.exported:
            LOGGER.info("Exported %d matches -> %s", result.matches_exported, result.bundle_path)
        else:
            LOGGER.debug("No export needed: %s", result.detail)
    return callback


async def _run_once(export_dir: Path) -> None:
    stored, skipped = await run_scrape()
    LOGGER.info("Batch complete: stored=%d skipped=%d", stored, skipped)
    if stored:
        SyncExporter(export_dir=export_dir).export()


async def _export_only(export_dir: Path) -> None:
    result = SyncExporter(export_dir=export_dir).export()
    if result.exported:
        LOGGER.info("Exported %d matches -> %s", result.matches_exported, result.bundle_path)
    else:
        LOGGER.info("No new data to export: %s", result.detail)


def _static_refresh_loop(stop_event: threading.Event) -> None:
    """Periodically refresh the current-patch static data + champion map.

    The server is rarely checked on, so keeping static data current automatically
    ensures new patches are captured without any human interaction.
    """
    refresh_static_data(force=True)  # initial refresh on start
    while not stop_event.is_set():
        if stop_event.wait(STATIC_REFRESH_INTERVAL_SECONDS):
            break
        refresh_static_data(force=True)


def _run_watcher(export_dir: Path) -> None:
    """Long-lived adaptive watcher that stores matches and exports bundles."""
    stop_event = threading.Event()
    pause_event = threading.Event()
    wake_event = threading.Event()
    profile = load_profile(PROFILE_PATH)
    exporter = SyncExporter(export_dir=export_dir)
    callback = _log_exporter_callback(exporter)

    # Keep the current patch/champion data fresh in the background.
    static_thread = threading.Thread(
        target=_static_refresh_loop,
        args=(stop_event,),
        daemon=True,
        name="collector-static-refresh",
    )
    static_thread.start()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # The collector never builds analytics — it exports data only. The callback
    # that v3's watcher would use for analytics is replaced with a no-op so the
    # shared watcher never imports analytics_builder / numpy / pandas on Ubuntu.
    def noop_analytics() -> None:
        pass

    try:
        loop.run_until_complete(
            run_background_match_watcher(
                stop_event,
                database_path=DEFAULT_DB_PATH,
                pause_event=pause_event,
                wake_event=wake_event,
                status_callback=callback,
                analytics_rebuild_callback=noop_analytics,
            )
        )
    except KeyboardInterrupt:
        LOGGER.info("Collector stopped by user.")
    finally:
        stop_event.set()
        loop.close()
        static_thread.join(timeout=5)


def _print_status() -> None:
    from scraper import DraftDatabase

    database = DraftDatabase(DEFAULT_DB_PATH)
    try:
        matches, participants, bans = database.counts()
        exporter = SyncExporter()
        state = exporter._read_state()
        print(f"database      : {DEFAULT_DB_PATH}")
        print(f"matches       : {matches:,}")
        print(f"participants  : {participants:,}")
        print(f"bans          : {bans:,}")
        print(f"last_exported : {state.get('last_exported_at', 'never')}")
        print(f"outbox        : {exporter.export_dir}")
        if exporter.export_dir.is_dir():
            bundles = sorted(exporter.export_dir.glob("*.sync.zip"))
            print(f"pending bundles: {len(bundles)}")
            for bundle in bundles[-5:]:
                print(f"  - {bundle.name}")
    finally:
        database.close()


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    while root.handlers:
        root.removeHandler(root.handlers[0])
    # Under pythonw.exe (the Windows auto-start task), sys.stdout/stderr are
    # None; adding a StreamHandler then would make every log emit fail, so it
    # is only attached when a console stream exists. The file log always works.
    if sys.stdout is not None and sys.stderr is not None:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        root.addHandler(stream)
    file_handler = logging.FileHandler(DEFAULT_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="Run a single batch then exit.")
    parser.add_argument("--export", action="store_true", help="Export pending data without collecting.")
    parser.add_argument("--status", action="store_true", help="Print DB/export status and exit.")
    parser.add_argument("--refresh-static", action="store_true", help="Force-refresh static data and exit.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    profile = load_profile(PROFILE_PATH)
    export_dir = _resolve_export_dir(profile)

    if args.status:
        _print_status()
        return
    if args.refresh_static:
        refresh_static_data(force=True)
        return
    if args.export:
        asyncio.run(_export_only(export_dir))
        return
    if args.once:
        asyncio.run(_run_once(export_dir))
        return
    _run_watcher(export_dir)


if __name__ == "__main__":
    main()
