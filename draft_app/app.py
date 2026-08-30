"""CustomTkinter desktop app for live LCU drafting, profiles and data jobs."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import queue
import sqlite3
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.messagebox as messagebox
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import customtkinter as ctk
import requests
from PIL import Image
from requests.adapters import HTTPAdapter

from config_manager import (
    ELO_OPTIONS,
    ENV_PATH,
    api_key_fingerprint,
    POOL_CATEGORIES,
    ROLES,
    load_profile,
    read_api_key,
    save_api_key,
    save_profile,
)
from data_dragon_maps import ChampionCatalog
from draft_engine import (
    BanRecommendation,
    BuildOption,
    DraftEngine,
    DraftInsights,
    LoadoutOption,
    Recommendation,
)
from lcu_state import BoardChampion, DraftSnapshot, EMPTY_SNAPSHOT, parse_lcu_session
from patch_utils import patch_label
from runtime_paths import EXECUTABLE_DIR, PROJECT_ROOT
from ingest import SyncIngester, resolve_inbox, resolve_pending_build
from scraper import (
    DEFAULT_DB_PATH,
    RiotAuthenticationError,
    RiotForbiddenError,
    run_background_match_watcher,
)
from static_data import CACHE_PATH as STATIC_CACHE_PATH

try:
    from lcu_driver import Connector
except ImportError:
    Connector = None  # type: ignore[assignment]

LOGGER = logging.getLogger("draft_app")
APP_VERSION = "3.2.1"
ROLE_LABELS = {
    "TOP": "Top",
    "JUNGLE": "Jungle",
    "MID": "Mid",
    "ADC": "ADC",
    "SUPPORT": "Support",
}
CATEGORY_LABELS = {
    "comfort_picks": "Comfort picks",
    "pocket_picks": "Pocket picks",
    "general_pool": "General pool",
}
IMAGE_CACHE_DIR = PROJECT_ROOT / "data" / "images"
STATIC_IMAGE_CACHE_DIR = PROJECT_ROOT / "data" / "static_images"
MAX_UI_EVENTS_PER_TICK = 80
MAX_CONSOLE_LINES = 3500
INGEST_POLL_INTERVAL_MS = 30_000  # 30s default; overridden by profile sync config
# Only images this small are decoded synchronously on the UI thread; bigger or
# unreadable files fall back to the background pipeline.
MAX_SYNC_DECODE_BYTES = 3 * 1024 * 1024
DECODER_CACHE_LIMIT = 600
PREFETCH_THREADS = 12


_HTTP_SESSION_LOCK = threading.Lock()
_HTTP_SESSION: requests.Session | None = None


def _shared_http_session() -> requests.Session:
    """Return one process-wide session so HTTP calls reuse pooled connections.

    Downloading ~170 champion portraits through ``requests.get`` opened a fresh
    TLS handshake per icon. A shared, connection-pooled session makes the same
    bulk download several times faster and keeps the app responsive afterwards.
    """
    global _HTTP_SESSION
    with _HTTP_SESSION_LOCK:
        if _HTTP_SESSION is None:
            session = requests.Session()
            session.headers.update({"User-Agent": "LeagueDraftLab/3.1"})
            adapter = HTTPAdapter(
                pool_connections=8, pool_maxsize=PREFETCH_THREADS + 4
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            _HTTP_SESSION = session
        return _HTTP_SESSION


def _atomic_save_image(path: Path, content: bytes) -> bool:
    """Persist *content* as PNG only when Pillow can decode it, atomically.

    Verifying before replacing the target prevents a transient CDN error page
    from poisoning the disk cache forever (a corrupt file would previously be
    skipped for re-download because only ``path.exists()`` was checked).
    """
    temporary_path = path.with_suffix(path.suffix + ".part")
    try:
        with Image.open(io.BytesIO(content)) as probe:
            probe.load()  # force full decode before trusting the payload
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(io.BytesIO(content)) as source:
            source.convert("RGBA").save(temporary_path, format="PNG")
        temporary_path.replace(path)
        return True
    except Exception:
        LOGGER.debug("Discarded invalid image payload for %s", path, exc_info=True)
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _fetch_image_file(url: str, path: Path) -> bool:
    """Download *url* into *path* (validated PNG), returning success."""
    try:
        response = _shared_http_session().get(url, timeout=12)
        response.raise_for_status()
        return _atomic_save_image(path, response.content)
    except Exception:
        LOGGER.debug("Image download failed: %s", url, exc_info=True)
        return False


def _prefetch_image_files(targets: Sequence[tuple[str, str, Path]]) -> tuple[int, int]:
    """Bulk-download ``(url, fallback_url, path)`` triples in parallel."""
    done = 0
    failed = 0

    def work(entry: tuple[str, str, Path]) -> bool:
        primary_url, fallback_url, path = entry
        if path.exists():
            return True
        if _fetch_image_file(primary_url, path):
            return True
        if fallback_url != primary_url and not path.exists():
            return _fetch_image_file(fallback_url, path)
        return path.exists()

    if not targets:
        return 0, 0
    with ThreadPoolExecutor(
        max_workers=PREFETCH_THREADS, thread_name_prefix="image-prefetch"
    ) as pool:
        for ok in pool.map(work, targets):
            if ok:
                done += 1
            else:
                failed += 1
    return done, failed


def _background_popen_kwargs() -> dict[str, Any]:
    """Return Windows flags that keep background Python children invisible.

    The data watcher captures stdout/stderr itself, so a separate console window
    has no purpose.  On non-Windows platforms no extra flags are required.
    """
    if os.name != "nt":
        return {}
    creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= int(getattr(subprocess, "STARTF_USESHOWWINDOW", 0x00000001))
    startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0))
    return {"creationflags": creationflags, "startupinfo": startupinfo}


class DualAxisScrollableFrame(ctk.CTkFrame):
    """CustomTkinter-compatible viewport with vertical and horizontal bars."""

    _instances: "weakref.WeakSet[DualAxisScrollableFrame]" = weakref.WeakSet()
    _bound_roots: "weakref.WeakKeyDictionary[Any, bool]" = weakref.WeakKeyDictionary()

    def __init__(self, master: Any, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        background = self._apply_appearance_mode(self.cget("fg_color"))
        self.canvas = tk.Canvas(
            self, background=background, highlightthickness=0, borderwidth=0
        )
        self.vertical_scrollbar = ctk.CTkScrollbar(
            self, orientation="vertical", command=self.canvas.yview
        )
        self.horizontal_scrollbar = ctk.CTkScrollbar(
            self, orientation="horizontal", command=self.canvas.xview
        )
        self.canvas.configure(
            yscrollcommand=self.vertical_scrollbar.set,
            xscrollcommand=self.horizontal_scrollbar.set,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        self.horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
        self.content = ctk.CTkFrame(self.canvas, fg_color="transparent")
        self._window_id = self.canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )
        self.content.bind("<Configure>", self._update_scroll_region, add="+")
        self._instances.add(self)
        root = self.winfo_toplevel()
        if root not in self._bound_roots:
            root.bind_all(
                "<MouseWheel>",
                lambda event: DualAxisScrollableFrame._route_wheel(event),
                add="+",
            )
            root.bind_all(
                "<Shift-MouseWheel>",
                lambda event: DualAxisScrollableFrame._route_wheel(
                    event, force_horizontal=True
                ),
                add="+",
            )
            root.bind_all(
                "<Button-4>",
                lambda event: DualAxisScrollableFrame._route_wheel(event),
                add="+",
            )
            root.bind_all(
                "<Button-5>",
                lambda event: DualAxisScrollableFrame._route_wheel(event),
                add="+",
            )
            self._bound_roots[root] = True

    def _update_scroll_region(self, _event: Any = None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    @staticmethod
    def _wheel_units(event: Any) -> int:
        delta = int(getattr(event, "delta", 0) or 0)
        if delta:
            return -1 if delta > 0 else 1
        number = int(getattr(event, "num", 0) or 0)
        return -1 if number == 4 else 1

    def _pointer_inside(self) -> bool:
        try:
            widget = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
            while widget is not None:
                if widget == self:
                    return True
                widget = getattr(widget, "master", None)
        except tk.TclError:
            return False
        return False

    @classmethod
    def _route_wheel(
        cls, event: Any, *, force_horizontal: bool = False
    ) -> str | None:
        for frame in list(cls._instances):
            try:
                if not frame.winfo_exists() or not frame._pointer_inside():
                    continue
                units = frame._wheel_units(event)
                horizontal = force_horizontal or bool(
                    int(getattr(event, "state", 0) or 0) & 0x0001
                )
                if horizontal:
                    frame.canvas.xview_scroll(units, "units")
                else:
                    frame.canvas.yview_scroll(units, "units")
                return "break"
            except tk.TclError:
                continue
        return None


class EventQueueLogHandler(logging.Handler):
    """Forward background-job log records into the Data Watcher console."""

    def __init__(self, output: queue.Queue[tuple[str, Any]]) -> None:
        super().__init__()
        self.output = output
        self.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.output.put(("console", self.format(record) + "\n"))
        except Exception:
            self.handleError(record)


class LCUBridge:
    """LCU websocket bridge with a low-frequency session polling fallback.

    The websocket path intentionally mirrors the last stable v2 implementation.
    Polling is independent insurance against missed/partial websocket updates and
    is deduplicated later by ``DraftSnapshot.draft_key``.
    """

    POLL_INTERVAL_SECONDS = 0.75

    def __init__(self, output: queue.Queue[tuple[str, Any]]) -> None:
        self.output = output

    def emit(self, kind: str, payload: Any) -> None:
        self.output.put((kind, payload))

    def run(self) -> None:
        if Connector is None:
            self.emit("lcu_status", "lcu-driver is not installed")
            return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        poll_task: asyncio.Task[Any] | None = None

        async def read_session(connection: Any) -> Any:
            response = await connection.request("get", "/lol-champ-select/v1/session")
            if response.status == 200:
                return await response.json()
            if response.status == 404:
                return None
            return None

        async def poll_session(connection: Any) -> None:
            # Websocket events remain the fast path. This modest fallback poll
            # makes the live board self-healing if a client build misses an
            # UPDATE event or emits a partial payload.
            while True:
                try:
                    await asyncio.sleep(self.POLL_INTERVAL_SECONDS)
                    self.emit("lcu_session", await read_session(connection))
                except asyncio.CancelledError:
                    raise
                except Exception:
                    LOGGER.debug("LCU session fallback poll failed.", exc_info=True)
                    await asyncio.sleep(1.5)

        try:
            connector = Connector(loop=loop)

            @connector.ready
            async def on_ready(connection: Any) -> None:
                nonlocal poll_task
                self.emit("lcu_status", "League Client connected")
                try:
                    self.emit("lcu_session", await read_session(connection))
                except Exception:
                    LOGGER.debug("No active Champion Select session at connect time.")
                if poll_task is not None and not poll_task.done():
                    poll_task.cancel()
                poll_task = asyncio.create_task(poll_session(connection))

            @connector.close
            async def on_close(_: Any) -> None:
                nonlocal poll_task
                if poll_task is not None and not poll_task.done():
                    poll_task.cancel()
                poll_task = None
                self.emit("lcu_status", "League Client disconnected")
                self.emit("lcu_session", None)

            @connector.ws.register(
                "/lol-champ-select/v1/session",
                event_types=("CREATE", "UPDATE", "DELETE"),
            )
            async def on_champ_select(_: Any, event: Any) -> None:
                # This is deliberately the simple, stable v2 callback. Do not
                # perform another LCU request from inside the websocket handler;
                # the independent poll task handles incomplete payload recovery.
                if str(getattr(event, "type", "UPDATE")).upper() == "DELETE":
                    self.emit("lcu_session", None)
                else:
                    data = getattr(event, "data", None)
                    if isinstance(data, Mapping):
                        self.emit("lcu_session", data)

            connector.start()
        except Exception as exc:
            LOGGER.exception("LCU connector stopped.")
            self.emit("lcu_status", f"LCU error: {exc}")
        finally:
            if poll_task is not None and not poll_task.done():
                poll_task.cancel()
                if not loop.is_closed():
                    try:
                        loop.run_until_complete(
                            asyncio.gather(poll_task, return_exceptions=True)
                        )
                    except Exception:
                        pass
            asyncio.set_event_loop(None)
            if not loop.is_closed():
                loop.close()


class DraftApp(ctk.CTk):
    def __init__(self) -> None:
        self.profile = load_profile()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        super().__init__()
        self.title(f"Local League Draft Lab v{APP_VERSION}")
        self.geometry(
            f"{self.profile['ui']['window_width']}x{self.profile['ui']['window_height']}"
        )
        self.minsize(1120, 720)

        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        # ONE shared handler forwards log records into the Data Watcher
        # console. Background threads previously attached their own handlers,
        # so every record was duplicated (once per attached handler).
        self._console_log_handler = EventQueueLogHandler(self.events)
        logging.getLogger().addHandler(self._console_log_handler)
        self.worker = ThreadPoolExecutor(max_workers=8, thread_name_prefix="draft-app")
        # Live scoring must never wait behind portrait/item downloads or other
        # general background tasks. v3 previously shared one executor for both,
        # so a burst of image requests could make later draft changes appear dead.
        self.analysis_worker = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="draft-analysis"
        )
        # Analytics reload has a dedicated worker. It cannot be starved behind
        # image downloads, API validation, or long-running data subprocesses.
        self.reload_worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="draft-reload"
        )
        # Ban scoring is intentionally isolated from pick scoring. In v3.0.5
        # every live generation performed picks and bans in one task, so rapid
        # websocket/poll updates could queue many obsolete ban passes behind the
        # current picks. The newest ban request now gets its own single-worker
        # lane; queued obsolete requests exit before doing any heavy work.
        self.ban_worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="draft-bans"
        )
        self.catalog = ChampionCatalog.load(allow_download=True)
        self.engine = DraftEngine(catalog=self.catalog)
        self._repair_profile_champion_names()
        self.snapshot = EMPTY_SNAPSHOT
        self.analysis_generations: dict[str, int] = {"live": 0, "manual": 0}
        self.ban_generation = 0
        self.ban_phase_over = False
        self.ban_watchdog_after: str | None = None
        self.last_live_render_generation = 0
        self.last_draft_key: tuple[Any, ...] | None = None
        self.last_ban_key: tuple[Any, ...] | None = None
        self.last_submitted_ban_key: tuple[Any, ...] | None = None
        self.pending_evaluation_after: str | None = None
        self.engine_reload_running = False
        self.engine_reload_pending = False
        self.engine_reload_watchdog_after: str | None = None
        self.data_job_lock = threading.Lock()
        self.champ_select_active = threading.Event()
        self.shutdown_event = threading.Event()
        self.collector_wake_event = threading.Event()
        self.collector_status_text = "starting"
        self.collector_busy = False
        self.ingest_status_text = "idle"
        self.ingest_busy = False
        self.current_data_job: tuple[str, bool] | None = None
        self.api_key_validation_state = "not checked"
        self.pool_boxes: dict[tuple[str, str], ctk.CTkTextbox] = {}
        self.weight_entries: dict[str, ctk.CTkEntry] = {}
        self.multiplier_entries: dict[str, ctk.CTkEntry] = {}
        self.ml_entries: dict[str, ctk.CTkEntry] = {}
        self.category_header_labels: dict[str, ctk.CTkLabel] = {}
        self.data_buttons: list[ctk.CTkButton] = []
        self.role_titles: dict[str, ctk.CTkLabel] = {}
        self.role_ban_frames: dict[str, ctk.CTkFrame] = {}
        self.role_pick_frames: dict[str, ctk.CTkFrame] = {}
        self.role_fingerprints: dict[str, tuple[Any, ...]] = {}
        self.ban_fingerprints: dict[str, tuple[Any, ...]] = {}
        self.current_recommendations: dict[str, list[Recommendation]] = {
            role: [] for role in ROLES
        }
        self.current_bans: dict[str, list[BanRecommendation]] = {role: [] for role in ROLES}
        self.open_detail_windows: set[ctk.CTkToplevel] = set()
        self.pending_role_renders: dict[
            str, tuple[Sequence[Recommendation] | None, Sequence[BanRecommendation] | None]
        ] = {}
        self.role_render_scheduled = False
        self.image_cache: dict[tuple[str, tuple[int, int]], ctk.CTkImage] = {}
        self.image_waiters: dict[
            tuple[str, tuple[int, int]], list[Any]
        ] = {}
        self.image_loads_inflight: set[tuple[str, tuple[int, int]]] = set()
        # Decoded PIL sources shared across every display size of one file.
        # 170 champion squares decode once instead of once per widget size.
        self.decoded_image_cache_lock = threading.Lock()
        self.decoded_image_cache: dict[Path, tuple[int, Image.Image]] = {}
        self._portrait_prefetch_started = False

        self._build_ui()
        self._start_image_cache_prefetch()
        self._populate_profile_fields()
        self._refresh_data_status()
        self.after(20, self._safe_drain_events)
        threading.Thread(
            target=LCUBridge(self.events).run,
            daemon=True,
            name="lcu-bridge",
        ).start()
        threading.Thread(
            target=self._run_background_match_watcher_thread,
            daemon=True,
            name="background-match-watcher",
        ).start()
        threading.Thread(
            target=self._run_ingest_watcher_thread,
            daemon=True,
            name="sync-ingest-watcher",
        ).start()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._show_empty_state()

    def _start_image_cache_prefetch(self) -> None:
        """Warm the on-disk portrait cache once per launch.

        Missing champion squares are downloaded in parallel through one pooled
        HTTP session while the user does something else. Portrait-heavy
        windows (pool editor, recommendation cards) afterwards hit only the
        local disk, which renders them effectively instantly.
        """
        if getattr(self, "_portrait_prefetch_started", False):
            return
        self._portrait_prefetch_started = True

        missing: list[tuple[str, str, Path]] = []
        valid_ids = {record.champion_id for record in self.catalog.records}
        stale_files: list[Path] = []
        for path in IMAGE_CACHE_DIR.glob("*.png"):
            stem = path.stem
            if stem.isdigit() and int(stem) not in valid_ids:
                # Leftover from removed catalog entries such as CDragon's
                # duplicated Jade-variant ids; keep the cache consistent.
                stale_files.append(path)
        for record in self.catalog.records:
            path = IMAGE_CACHE_DIR / f"{record.champion_id}.png"
            if path.exists():
                continue
            primary = self.catalog.square_url(record.champion_id)
            fallback = self.catalog.fallback_square_url(record.champion_id)
            missing.append((primary, fallback, path))
        if stale_files:
            LOGGER.info("Pruning %d stale portrait file(s).", len(stale_files))
            for path in stale_files:
                try:
                    path.unlink()
                except OSError:
                    LOGGER.debug("Could not delete stale portrait %s", path)
        if not missing:
            LOGGER.info("Portrait cache complete: %d icons present.", len(self.catalog.records))
            return
        LOGGER.info("Portrait prefetch started: %d icons missing.", len(missing))

        def work() -> None:
            done, failed = _prefetch_image_files(missing)
            LOGGER.info(
                "Portrait prefetch finished: %d downloaded, %d failed.", done, failed
            )

        self.worker.submit(work)

    def _repair_profile_champion_names(self) -> None:
        """Persist safe, unique typo corrections such as ``Jihn`` -> ``Jhin``."""
        updated = json.loads(json.dumps(self.profile))
        corrections: list[str] = []
        changed = False
        for category in POOL_CATEGORIES:
            for role in ROLES:
                repaired: list[str] = []
                for supplied in updated[category][role]:
                    canonical, corrected = self.engine.canonical_champion_name(supplied)
                    final_name = canonical or supplied
                    if canonical and canonical.casefold() != supplied.casefold():
                        changed = True
                        if corrected:
                            corrections.append(f"{supplied} -> {canonical}")
                    if final_name not in repaired:
                        repaired.append(final_name)
                updated[category][role] = repaired
        if changed:
            self.profile = save_profile(updated)
            self.engine.reload()
            if corrections:
                LOGGER.info("Corrected profile champion names: %s", "; ".join(corrections))
            else:
                LOGGER.info("Normalised internal champion aliases in the profile.")

    def _build_ui(self) -> None:
        header = ctk.CTkFrame(self, corner_radius=0)
        header.pack(fill="x")
        ctk.CTkLabel(
            header,
            text=f"LOCAL LEAGUE DRAFT LAB · v{APP_VERSION}",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(side="left", padx=14, pady=9)
        self.status_label = ctk.CTkLabel(header, text="Starting…")
        self.status_label.pack(side="right", padx=14)

        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=(5, 8))
        self.live_tab = self.tabs.add("Live Draft")
        self.manual_tab = self.tabs.add("Manual Lab")
        self.insights_tab = self.tabs.add("Draft Insights")
        self.profile_tab = self.tabs.add("Champion Pools")
        self.model_tab = self.tabs.add("Model & Features")
        self.data_tab = self.tabs.add("Data Watcher")
        self.settings_tab = self.tabs.add("Settings")
        self._build_live_tab()
        self._build_manual_tab()
        self._build_insights_tab()
        self._build_profile_tab()
        self._build_model_tab()
        self._build_data_tab()
        self._build_settings_tab()

    def _build_live_tab(self) -> None:
        controls = ctk.CTkFrame(self.live_tab)
        controls.pack(fill="x", padx=5, pady=5)
        self.phase_label = ctk.CTkLabel(
            controls,
            text=EMPTY_SNAPSHOT.phase,
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.phase_label.pack(side="left", padx=9, pady=6)
        self.active_role_label = ctk.CTkLabel(controls, text="Your role: unknown")
        self.active_role_label.pack(side="left", padx=14)
        ctk.CTkButton(
            controls,
            text="Reload analytics",
            width=126,
            height=29,
            command=self._reload_engine_async,
        ).pack(side="right", padx=6)

        self.board_label = ctk.CTkLabel(
            self.live_tab,
            text="Allies: —     Enemies: —     Bans: —",
            anchor="w",
        )
        self.board_label.pack(fill="x", padx=11, pady=(0, 2))

        self.recommendation_frame = DualAxisScrollableFrame(self.live_tab)
        self.recommendation_frame.pack(fill="both", expand=True, padx=5, pady=5)
        recommendation_content = self.recommendation_frame.content
        for column, role in enumerate(ROLES):
            recommendation_content.grid_columnconfigure(column, weight=1, minsize=232)
            role_frame = ctk.CTkFrame(recommendation_content)
            role_frame.grid(row=0, column=column, padx=3, pady=3, sticky="nsew")
            title = ctk.CTkLabel(
                role_frame,
                text=ROLE_LABELS[role],
                font=ctk.CTkFont(size=16, weight="bold"),
            )
            title.pack(pady=(7, 2))
            self.role_titles[role] = title
            ctk.CTkLabel(
                role_frame,
                text="BAN PRIORITY",
                text_color="gray70",
                font=ctk.CTkFont(size=10, weight="bold"),
            ).pack(anchor="w", padx=7, pady=(1, 0))
            ban_frame = ctk.CTkFrame(role_frame, fg_color="transparent")
            ban_frame.pack(fill="x", padx=5, pady=(0, 4))
            self.role_ban_frames[role] = ban_frame
            ctk.CTkLabel(
                role_frame,
                text="PICKS / LOCKED BUILD",
                text_color="gray70",
                font=ctk.CTkFont(size=10, weight="bold"),
            ).pack(anchor="w", padx=7, pady=(1, 0))
            pick_frame = ctk.CTkFrame(role_frame, fg_color="transparent")
            pick_frame.pack(fill="x", padx=2, pady=(0, 5))
            self.role_pick_frames[role] = pick_frame

    def _build_manual_tab(self) -> None:
        controls = ctk.CTkFrame(self.manual_tab)
        controls.pack(fill="x", padx=7, pady=7)
        ctk.CTkLabel(
            controls,
            text="Manual draft test · use ROLE:Champion when a role is known",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, columnspan=5, sticky="w", padx=8, pady=(6, 2))
        self.manual_ally = ctk.CTkEntry(controls, placeholder_text="TOP:Ornn, JUNGLE:Vi")
        self.manual_enemy = ctk.CTkEntry(controls, placeholder_text="TOP:Darius, Ahri")
        self.manual_bans = ctk.CTkEntry(controls, placeholder_text="Yone, Smolder")
        self.manual_role_menu = ctk.CTkOptionMenu(controls, values=list(ROLES), width=110)
        self.manual_role_menu.set("MID")
        self.manual_ally.grid(row=1, column=0, sticky="ew", padx=4, pady=6)
        self.manual_enemy.grid(row=1, column=1, sticky="ew", padx=4, pady=6)
        self.manual_bans.grid(row=1, column=2, sticky="ew", padx=4, pady=6)
        self.manual_role_menu.grid(row=1, column=3, padx=4)
        ctk.CTkButton(controls, text="Evaluate", command=self._manual_evaluate).grid(row=1, column=4, padx=5)
        for column in range(3):
            controls.grid_columnconfigure(column, weight=1)
        self.manual_output = ctk.CTkTextbox(
            self.manual_tab, wrap="none", font=("Consolas", 11)
        )
        self.manual_output.pack(fill="both", expand=True, padx=7, pady=(0, 7))
        self.manual_output.insert(
            "1.0",
            "Enter a draft above. Results, confidence, inferred roles and explanations will appear here.\n",
        )
        self.manual_output.configure(state="disabled")

    def _build_insights_tab(self) -> None:
        scroll = DualAxisScrollableFrame(self.insights_tab)
        scroll.pack(fill="both", expand=True, padx=7, pady=7)
        body = scroll.content
        top = ctk.CTkFrame(body)
        top.pack(fill="x", pady=4)
        self.insight_prediction_label = ctk.CTkLabel(
            top, text="Predicted win probability: waiting for draft",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.insight_prediction_label.pack(anchor="w", padx=12, pady=(10, 4))
        self.insight_damage_label = ctk.CTkLabel(top, text="Damage profile: —", justify="left")
        self.insight_damage_label.pack(anchor="w", padx=12, pady=(0, 10))

        columns = ctk.CTkFrame(body)
        columns.pack(fill="both", expand=True, pady=4)
        columns.grid_columnconfigure(0, weight=1)
        columns.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(columns, text="Strengths / weaknesses", font=ctk.CTkFont(size=15, weight="bold")).grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ctk.CTkLabel(columns, text="Role inference", font=ctk.CTkFont(size=15, weight="bold")).grid(row=0, column=1, sticky="w", padx=8, pady=4)
        self.insight_strengths_text = ctk.CTkTextbox(columns, height=310, wrap="word")
        self.insight_role_text = ctk.CTkTextbox(columns, height=310, wrap="word")
        self.insight_strengths_text.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        self.insight_role_text.grid(row=1, column=1, sticky="nsew", padx=6, pady=4)

        embedding = ctk.CTkFrame(body)
        embedding.pack(fill="x", pady=8)
        ctk.CTkLabel(embedding, text="Champion embedding neighbours", font=ctk.CTkFont(size=15, weight="bold")).pack(side="left", padx=10, pady=8)
        self.embedding_entry = ctk.CTkEntry(embedding, placeholder_text="Champion name", width=220)
        self.embedding_entry.pack(side="left", padx=6)
        ctk.CTkButton(embedding, text="Find similar", command=self._show_embedding_neighbors, width=120).pack(side="left", padx=6)
        self.embedding_output = ctk.CTkLabel(embedding, text="", justify="left", anchor="w")
        self.embedding_output.pack(side="left", fill="x", expand=True, padx=10)

    def _build_profile_tab(self) -> None:
        profile_scroll = DualAxisScrollableFrame(self.profile_tab)
        profile_scroll.pack(fill="both", expand=True, padx=6, pady=6)
        scroll = profile_scroll.content
        top = ctk.CTkFrame(scroll)
        top.pack(fill="x", pady=4)
        self.restrict_var = ctk.BooleanVar()
        ctk.CTkCheckBox(
            top,
            text="Restrict only my active role to its configured pool",
            variable=self.restrict_var,
        ).pack(side="left", padx=12, pady=9)
        ctk.CTkLabel(
            top,
            text="Other roles remain unrestricted meta recommendations.",
            text_color="gray70",
        ).pack(side="left", padx=8)

        ctk.CTkLabel(
            scroll,
            text="Champion pools by role · one name per line or comma-separated",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(anchor="w", padx=5, pady=(10, 3))
        pool_grid = ctk.CTkFrame(scroll)
        pool_grid.pack(fill="x")
        ctk.CTkLabel(pool_grid, text="Role").grid(row=0, column=0, padx=4, pady=5)
        for column, category in enumerate(POOL_CATEGORIES, start=1):
            label = ctk.CTkLabel(pool_grid, text="")
            label.grid(row=0, column=column, padx=4)
            self.category_header_labels[category] = label
        for row, role in enumerate(ROLES, start=1):
            ctk.CTkLabel(pool_grid, text=ROLE_LABELS[role]).grid(row=row, column=0, padx=8, sticky="n")
            for column, category in enumerate(POOL_CATEGORIES, start=1):
                cell = ctk.CTkFrame(pool_grid)
                cell.grid(row=row, column=column, padx=4, pady=4, sticky="ew")
                box = ctk.CTkTextbox(cell, height=66, width=245)
                box.pack(fill="x", expand=True)
                self.pool_boxes[(category, role)] = box
                ctk.CTkButton(
                    cell,
                    text="Browse champions…",
                    height=26,
                    command=lambda c=category, r=role: self._open_champion_pool_editor(c, r),
                ).pack(anchor="e", pady=(2, 0))
        for column in range(1, 4):
            pool_grid.grid_columnconfigure(column, weight=1)

        multipliers = ctk.CTkFrame(scroll)
        multipliers.pack(fill="x", pady=8)
        for column, key in enumerate(("comfort", "pocket")):
            ctk.CTkLabel(multipliers, text=f"{key.title()} multiplier").grid(row=0, column=column, padx=8, pady=(7, 2))
            entry = ctk.CTkEntry(multipliers, width=120)
            entry.grid(row=1, column=column, padx=8, pady=(0, 8))
            self.multiplier_entries[key] = entry
        ctk.CTkButton(scroll, text="Save pools and reload engine", command=self._save_profile_from_gui, height=34).pack(anchor="e", padx=7, pady=10)

    def _build_model_tab(self) -> None:
        scroll = DualAxisScrollableFrame(self.model_tab)
        scroll.pack(fill="both", expand=True, padx=7, pady=7)
        body = scroll.content
        self.model_status_label = ctk.CTkLabel(
            body, text="Model status loading…", justify="left",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.model_status_label.pack(fill="x", padx=9, pady=9)
        actions = ctk.CTkFrame(body)
        actions.pack(fill="x", pady=4)
        ctk.CTkButton(actions, text="Rebuild analytics + model", command=lambda: self._start_data_job("build"), width=190).pack(side="left", padx=7, pady=8)
        ctk.CTkButton(actions, text="Check GPU (.venv)", command=self._check_gpu_async, width=145).pack(side="left", padx=7)
        ctk.CTkLabel(actions, text="GPU training is optional; live inference always uses the small NumPy export.", text_color="gray70").pack(side="left", padx=10)

        ctk.CTkLabel(body, text="Recommendation scoring weights", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=6, pady=(12, 3))
        weights = ctk.CTkFrame(body)
        weights.pack(fill="x")
        score_keys = ("global_win_rate", "synergy_delta", "counter_delta", "composition", "machine_learning", "confidence")
        for index, key in enumerate(score_keys):
            ctk.CTkLabel(weights, text=key.replace("_", " ").title()).grid(row=0, column=index, padx=4, pady=(5, 2))
            entry = ctk.CTkEntry(weights, width=110)
            entry.grid(row=1, column=index, padx=4, pady=(0, 7))
            self.weight_entries[key] = entry
            weights.grid_columnconfigure(index, weight=1)

        ctk.CTkLabel(body, text="Neural model and patch-awareness settings", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=6, pady=(14, 3))
        grid = ctk.CTkFrame(body)
        grid.pack(fill="x")
        fields = (
            ("ensemble_size", "Ensemble models"),
            ("embedding_dimension", "Embedding dimensions"),
            ("epochs", "Maximum epochs"),
            ("batch_size", "Batch size"),
            ("patch_half_life", "Patch half-life"),
            ("minimum_patch_weight", "Old-game minimum weight"),
            ("minimum_training_matches", "Minimum complete matches"),
        )
        for index, (key, label) in enumerate(fields):
            row, column = divmod(index, 4)
            ctk.CTkLabel(grid, text=label).grid(row=row * 2, column=column, padx=7, pady=(6, 2))
            entry = ctk.CTkEntry(grid, width=155)
            entry.grid(row=row * 2 + 1, column=column, padx=7, pady=(0, 7))
            self.ml_entries[key] = entry
            grid.grid_columnconfigure(column, weight=1)
        self.use_gpu_var = ctk.BooleanVar(value=True)
        self.mixed_precision_var = ctk.BooleanVar(value=True)
        toggles = ctk.CTkFrame(body)
        toggles.pack(fill="x", pady=7)
        ctk.CTkCheckBox(toggles, text="Use CUDA GPU when available", variable=self.use_gpu_var).pack(side="left", padx=10, pady=8)
        ctk.CTkCheckBox(toggles, text="Mixed-precision GPU training", variable=self.mixed_precision_var).pack(side="left", padx=10)
        ctk.CTkButton(body, text="Save model settings", command=self._save_profile_from_gui, height=34).pack(anchor="e", padx=7, pady=10)

    def _build_data_tab(self) -> None:
        status = ctk.CTkFrame(self.data_tab)
        status.pack(fill="x", padx=8, pady=8)
        status_text = ctk.CTkFrame(status, fg_color="transparent")
        status_text.pack(side="left", fill="x", expand=True, padx=12, pady=10)
        self.data_status_label = ctk.CTkLabel(
            status_text, text="", anchor="w", justify="left"
        )
        self.data_status_label.pack(fill="x", anchor="w")
        self.patch_status_label = ctk.CTkLabel(
            status_text, text="", anchor="w", justify="left",
            text_color="gray70", font=ctk.CTkFont(size=11),
        )
        self.patch_status_label.pack(fill="x", anchor="w", pady=(3, 0))
        ctk.CTkButton(
            status, text="Refresh status", command=self._refresh_data_status
        ).pack(side="right", padx=8)

        buttons = ctk.CTkFrame(self.data_tab)
        buttons.pack(fill="x", padx=8, pady=(0, 8))
        # Legacy single-app controls removed: match scraping and the automatic
        # Riot watcher live on the collector server now, so this tab only
        # offers local analytics/static maintenance and ingest helpers.
        build_button = ctk.CTkButton(
            buttons,
            text="Rebuild analytics only",
            command=lambda: self._start_data_job("build"),
            width=180,
        )
        build_button.pack(side="left", padx=6, pady=10)
        static_button = ctk.CTkButton(
            buttons,
            text="Refresh static data",
            command=lambda: self._start_data_job("static"),
            width=170,
        )
        static_button.pack(side="left", padx=6)
        self.data_buttons = [build_button, static_button]
        ctk.CTkButton(
            buttons,
            text="Open data folder",
            command=self._open_data_folder,
            width=145,
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            buttons,
            text="Open sync inbox",
            command=self._open_sync_inbox,
            width=145,
        ).pack(side="left", padx=6)
        # The local watcher is off by default (collection runs on the collector
        # server), so the watcher pill starts empty and only appears when the
        # local watcher is actually enabled and doing something.
        if self.profile["background_collector"].get("enabled", False):
            initial_watcher = "Watcher: starting…"
        else:
            initial_watcher = ""
        self.job_label = ctk.CTkLabel(buttons, text=initial_watcher)
        self.job_label.pack(side="right", padx=12)
        self.console = ctk.CTkTextbox(
            self.data_tab, font=("Consolas", 12), wrap="none"
        )
        self.console.pack(fill="both", expand=True, padx=8, pady=8)
        self._append_console(
            "Match collection runs on your collector server (Riot watcher + scraper). "
            "This PC ingests the sync bundles it sends and rebuilds analytics automatically; "
            "use the buttons above only for local maintenance.\n"
        )

    def _build_settings_tab(self) -> None:
        scroll = DualAxisScrollableFrame(self.settings_tab)
        scroll.pack(fill="both", expand=True, padx=7, pady=7)
        body = scroll.content
        general = ctk.CTkFrame(body)
        general.pack(fill="x", pady=4)
        ctk.CTkLabel(general, text="Target Elo").grid(row=0, column=0, padx=8, pady=8)
        self.elo_menu = ctk.CTkOptionMenu(general, values=list(ELO_OPTIONS))
        self.elo_menu.grid(row=0, column=1, padx=8)
        self.background_var = ctk.BooleanVar()
        ctk.CTkCheckBox(general, text="Automatic background match watcher", variable=self.background_var).grid(row=0, column=2, padx=14)
        self.ingest_var = ctk.BooleanVar()
        ctk.CTkCheckBox(general, text="Auto-ingest server data + rebuild", variable=self.ingest_var).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=4)
        ctk.CTkLabel(general, text="Open-role picks").grid(row=0, column=3, padx=(18, 4))
        self.top_n_entry = ctk.CTkEntry(general, width=80)
        self.top_n_entry.grid(row=0, column=4, padx=6)

        secret = ctk.CTkFrame(body)
        secret.pack(fill="x", pady=7)
        self.key_status = ctk.CTkLabel(secret, text="")
        self.key_status.pack(side="left", padx=8, pady=8)
        self.api_key_entry = ctk.CTkEntry(secret, show="•", width=390, placeholder_text="Paste a fresh RGAPI key")
        self.api_key_entry.pack(side="left", padx=8, fill="x", expand=True)
        ctk.CTkButton(secret, text="Save & validate entered key", width=190, command=self._save_entered_api_key).pack(side="left", padx=(2, 4))
        ctk.CTkButton(secret, text="Validate saved", width=110, command=self._validate_api_key_async).pack(side="left", padx=(0, 8))
        self.key_path_label = ctk.CTkLabel(body, text=f"Key file: {ENV_PATH}", anchor="w", font=ctk.CTkFont(size=11))
        self.key_path_label.pack(fill="x", padx=12, pady=(0, 6))
        ctk.CTkButton(body, text="Save all settings and reload", command=self._save_profile_from_gui, height=36).pack(anchor="e", padx=8, pady=10)
        self.profile_message = ctk.CTkLabel(body, text="", justify="left")
        self.profile_message.pack(anchor="e", padx=8)

    def _populate_profile_fields(self) -> None:
        self.elo_menu.set(self.profile["target_elo"])
        self.restrict_var.set(bool(self.profile["restrict_to_pool"]))
        self.background_var.set(bool(self.profile["background_collector"]["enabled"]))
        self.ingest_var.set(bool(self.profile.get("sync", {}).get("enable_pc_ingest", True)))
        self._update_continuous_controls()
        self._update_api_key_status()
        for (category, role), box in self.pool_boxes.items():
            box.delete("1.0", "end")
            box.insert("1.0", "\n".join(self.profile[category][role]))
        for key, entry in self.weight_entries.items():
            entry.delete(0, "end")
            entry.insert(0, str(self.profile["weights"][key]))
        for key, entry in self.multiplier_entries.items():
            entry.delete(0, "end")
            entry.insert(0, str(self.profile["personal_multipliers"][key]))
        ml = self.profile.get("machine_learning", {})
        for key, entry in self.ml_entries.items():
            entry.delete(0, "end")
            entry.insert(0, str(ml.get(key, "")))
        self.use_gpu_var.set(bool(ml.get("use_gpu", True)))
        self.mixed_precision_var.set(bool(ml.get("mixed_precision", True)))
        self.top_n_entry.delete(0, "end")
        self.top_n_entry.insert(0, str(self.profile["ui"].get("top_n", 10)))
        self._refresh_pool_header_labels()
        self._refresh_model_status()

    def _refresh_pool_header_labels(self) -> None:
        multipliers = self.profile["personal_multipliers"]
        labels = {
            "comfort_picks": f"Comfort · {float(multipliers['comfort']):g}×",
            "pocket_picks": f"Pocket · {float(multipliers['pocket']):g}×",
            "general_pool": "General",
        }
        for category, label in self.category_header_labels.items():
            label.configure(text=labels[category])


    def _open_champion_pool_editor(self, category: str, role: str) -> None:
        """Open an alphabetically-sorted champion icon picker for one pool.

        Clicking an icon adds/removes that champion from the given role's pool
        by editing the shared textbox, so the normal save/reload flow is reused.
        Comfort and Pocket pickers are restricted to the champions present in
        that role's General pool; when it is empty, the user is routed there
        first because those two lists only have meaning inside it.
        """
        box = self.pool_boxes[(category, role)]
        allowed_ids: set[int] | None = None
        if category != "general_pool":
            general_names = self._split_names(
                self.pool_boxes[("general_pool", role)].get("1.0", "end")
            )
            if not general_names:
                self._prompt_for_general_pool_first(category, role)
                return
            allowed_ids = {
                champion_id
                for name in general_names
                if (champion_id := self.catalog.id_for_name(name)) is not None
            }
            if not allowed_ids:
                messagebox.showwarning(
                    "Unrecognised General pool",
                    f"None of the names in your {ROLE_LABELS[role]} General pool "
                    "match known champions. Fix that list before editing "
                    f"{CATEGORY_LABELS[category].lower()}.",
                )
                return

        window = ctk.CTkToplevel(self)
        self.open_detail_windows.add(window)
        window.title(f"{ROLE_LABELS[role]} · {CATEGORY_LABELS[category]} editor")
        window.geometry("940x660")
        window.minsize(700, 480)
        window.transient(self)
        window.protocol("WM_DELETE_WINDOW", lambda: self._close_detail_window(window))

        explanation = (
            "Only champions in this role's General pool are shown."
            if allowed_ids is not None
            else "Click a champion icon to add or remove it from the pool. Close when done."
        )
        label = ctk.CTkLabel(window, text=explanation, anchor="w")
        label.pack(fill="x", padx=10, pady=(10, 2))
        hint = ctk.CTkLabel(
            window,
            text="Selected: 0 in pool · click icons to toggle",
            text_color="gray70",
            anchor="w",
        )
        hint.pack(fill="x", padx=10, pady=(0, 4))

        scroll = DualAxisScrollableFrame(window)
        scroll.pack(fill="both", expand=True, padx=8, pady=8)
        grid = scroll.content
        icon_size = 52
        per_row = 12
        # Defensive per-id dedupe: even if a stale catalog cache still contains
        # duplicated champion rows, never render the same icon twice.
        records: list[Any] = []
        seen_ids: set[int] = set()
        for record in sorted(self.catalog.records, key=lambda r: r.name.casefold()):
            if record.champion_id in seen_ids:
                continue
            if allowed_ids is not None and record.champion_id not in allowed_ids:
                continue
            seen_ids.add(record.champion_id)
            records.append(record)

        def current_names() -> list[str]:
            return self._split_names(box.get("1.0", "end"))

        buttons: dict[int, ctk.CTkButton] = {}

        def refresh_window() -> None:
            selected = set(current_names())
            hint.configure(
                text=(
                    f"Selected: {len(selected)} in pool · "
                    f"{len(records)} candidates shown from General pool"
                    if allowed_ids is not None
                    else f"Selected: {len(selected)} in pool · click icons to toggle"
                )
            )
            for champion_id, icon in buttons.items():
                name = self.catalog.name_for_id(champion_id)
                icon.configure(
                    fg_color=("gray45" if name in selected else "transparent"),
                    text=name if name not in selected else f"✓ {name}",
                )

        for index, record in enumerate(records):
            row = index // per_row
            col = index % per_row
            champion_id = record.champion_id
            name = record.name

            icon = ctk.CTkButton(
                grid,
                text=name,
                width=icon_size,
                height=icon_size,
                corner_radius=8,
                hover_color="gray30",
                command=lambda cid=champion_id, cname=name: self._toggle_pool_champion(
                    category, role, box, cname, buttons, refresh_window
                ),
            )
            icon.grid(row=row, column=col, padx=3, pady=3)
            buttons[champion_id] = icon
            self._load_champion_portrait(champion_id, icon, (icon_size - 8, icon_size - 8))

        for col in range(per_row):
            grid.grid_columnconfigure(col, weight=1)

        refresh_window()

    def _prompt_for_general_pool_first(self, category: str, role: str) -> bool:
        """Explain that Comfort/Pocket live inside the General pool.

        Returns True when a General-pool editor was opened as a follow-up so
        the user can fill it immediately.
        """
        opened = messagebox.askyesno(
            "Select the General pool first",
            f"{CATEGORY_LABELS[category]} picks are chosen from your "
            f"{ROLE_LABELS[role]} General pool.\n\n"
            "That pool is empty right now, so there is nothing to pick from here.",
        )
        if opened:
            self._open_champion_pool_editor("general_pool", role)
        return opened

    def _toggle_pool_champion(
        self,
        category: str,
        role: str,
        box: Any,
        champion_name: str,
        buttons: dict[int, ctk.CTkButton] | None = None,
        refresh: Any = None,
    ) -> None:
        names = self._split_names(box.get("1.0", "end"))
        was_removed = champion_name in names
        if was_removed:
            names.remove(champion_name)
        else:
            names.append(champion_name)
        names.sort(key=lambda n: n.casefold())
        box.delete("1.0", "end")
        box.insert("1.0", "\n".join(names))
        if refresh is not None:
            refresh()
        other_category = {
            "comfort_picks": "pocket_picks",
            "pocket_picks": "comfort_picks",
        }.get(category)
        # Overlap guard: the engine resolves a dual membership as Comfort
        # (comfort is checked before pocket), so surface that consequence at
        # the moment of the click instead of letting the multiplier silently
        # disappear.
        if other_category and not was_removed:
            other_names = self._split_names(
                self.pool_boxes[(other_category, role)].get("1.0", "end")
            )
            if champion_name in other_names:
                messagebox.showwarning(
                    "In both Comfort and Pocket picks",
                    f"{champion_name} is now in both pools for "
                    f"{ROLE_LABELS[role]}.\n\n"
                    "Until you remove one of them, Comfort wins: the Comfort "
                    "multiplier applies and the Pocket multiplier is ignored.",
                )


    @staticmethod
    def _split_names(text: str) -> list[str]:
        values = text.replace(",", "\n").splitlines()
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    def _canonicalize_pool_names(self, names: Sequence[str]) -> tuple[list[str], list[str]]:
        canonical: list[str] = []
        corrections: list[str] = []
        unknown: list[str] = []
        for name in names:
            resolved, corrected = self.engine.canonical_champion_name(name)
            if not resolved:
                unknown.append(name)
                continue
            if corrected and resolved.casefold() != name.casefold():
                corrections.append(f"{name} → {resolved}")
            if resolved not in canonical:
                canonical.append(resolved)
        if unknown:
            raise ValueError("Unknown champion name(s): " + ", ".join(unknown))
        return canonical, corrections

    def _update_api_key_status(self) -> None:
        key = read_api_key()
        if not key:
            self.key_status.configure(text="No API key saved")
            return
        state = self.api_key_validation_state
        self.key_status.configure(
            text=f"API key {api_key_fingerprint(key)} · {state}"
        )

    def _save_entered_api_key(self, *, validate: bool = True) -> bool:
        """Save the text currently in the masked field to the canonical file."""
        key = self.api_key_entry.get().strip()
        if not key:
            self.profile_message.configure(
                text="Paste a new RGAPI key into the field before saving."
            )
            return False
        try:
            written_path = save_api_key(key)
            self.api_key_entry.delete(0, "end")
            self.api_key_validation_state = "saved; not checked"
            self._update_api_key_status()
            self.profile_message.configure(
                text=f"Saved API key to {written_path}."
            )
            self._append_console(
                f"\n> saved Riot API key {api_key_fingerprint(key)} to {written_path}\n"
            )
            self.collector_wake_event.set()
        except Exception as exc:
            self.profile_message.configure(text=f"Could not save API key: {exc}")
            return False
        if validate:
            self._validate_api_key_async()
        return True

    def _validate_api_key_async(self) -> None:
        if not read_api_key():
            self.api_key_validation_state = "missing"
            self._update_api_key_status()
            self.profile_message.configure(
                text="Paste a fresh RGAPI key, then save the profile first."
            )
            return
        self.api_key_validation_state = "checking…"
        self._update_api_key_status()
        self._append_console(
            f"\n> validating saved Riot API key from {ENV_PATH}\n"
        )

        def work() -> None:
            try:
                from scraper import validate_saved_api_key

                result = asyncio.run(validate_saved_api_key())
                self.events.put(("api_key_valid", result))
            except (RiotAuthenticationError, RiotForbiddenError) as exc:
                self.events.put(("api_key_invalid", str(exc)))
            except Exception as exc:
                self.events.put(("api_key_error", str(exc)))

        self.worker.submit(work)

    def _save_profile_from_gui(self) -> None:
        try:
            updated = json.loads(json.dumps(self.profile))
            updated["target_elo"] = self.elo_menu.get()
            updated["restrict_to_pool"] = bool(self.restrict_var.get())
            updated["background_collector"]["enabled"] = bool(self.background_var.get())
            updated["sync"]["enable_pc_ingest"] = bool(self.ingest_var.get())
            corrections: list[str] = []
            for (category, role), box in self.pool_boxes.items():
                names, fixed = self._canonicalize_pool_names(
                    self._split_names(box.get("1.0", "end"))
                )
                updated[category][role] = names
                corrections.extend(fixed)
            for key, entry in self.weight_entries.items():
                updated["weights"][key] = float(entry.get())
            for key, entry in self.multiplier_entries.items():
                updated["personal_multipliers"][key] = float(entry.get())
            integer_ml = {
                "ensemble_size", "embedding_dimension", "epochs", "batch_size",
                "minimum_training_matches",
            }
            for key, entry in self.ml_entries.items():
                raw = entry.get().strip()
                updated["machine_learning"][key] = int(raw) if key in integer_ml else float(raw)
            updated["machine_learning"]["use_gpu"] = bool(self.use_gpu_var.get())
            updated["machine_learning"]["mixed_precision"] = bool(self.mixed_precision_var.get())
            updated["ui"]["top_n"] = int(self.top_n_entry.get())
            key_changed = bool(self.api_key_entry.get().strip())
            if key_changed and not self._save_entered_api_key(validate=False):
                return
            self.profile = save_profile(updated)
            self.collector_wake_event.set()
            self._populate_profile_fields()
            self._reload_engine_async()
            message = "Saved v3 settings. Personal pool rules apply only to your active role."
            if corrections:
                message += " Corrected: " + "; ".join(corrections)
            self.profile_message.configure(text=message)
            if key_changed:
                self._validate_api_key_async()
        except Exception as exc:
            self.profile_message.configure(text=f"Could not save: {exc}")

    def _update_continuous_controls(self) -> None:
        """No-op: watcher pause/resume buttons moved to the collector server.

        The Settings-tab checkbox still decides whether the local watcher loop
        sleeps or works, so nothing else had to change for the removal of the
        legacy Data Watcher controls.
        """
        return

    def _reload_engine_async(self) -> None:
        """Load analytics into a fresh engine and atomically swap it in.

        Reloading the existing engine can block behind live evaluations because
        both operations share its internal lock. Building a replacement engine
        off-thread leaves the currently displayed engine usable and guarantees
        that the Reload button always reaches a completion event.
        """
        if self.engine_reload_running:
            self.engine_reload_pending = True
            self.status_label.configure(text="Analytics reload queued…")
            return

        self.engine_reload_running = True
        self.engine_reload_pending = False
        self.status_label.configure(text="Reloading analytics…")
        current = self.engine

        watchdog_after = getattr(self, "engine_reload_watchdog_after", None)
        if watchdog_after:
            try:
                self.after_cancel(watchdog_after)
            except Exception:
                pass
        self.engine_reload_watchdog_after = None
        if hasattr(self, "after"):
            try:
                self.engine_reload_watchdog_after = self.after(
                    15000, self._engine_reload_watchdog
                )
            except Exception:
                self.engine_reload_watchdog_after = None

        def work() -> None:
            try:
                replacement = DraftEngine(
                    database_path=current.database_path,
                    profile_path=current.profile_path,
                    model_path=current.model_path,
                    catalog=self.catalog,
                )
                self.events.put(("engine_reloaded", replacement))
            except Exception as exc:
                LOGGER.exception("Analytics replacement engine failed to load.")
                self.events.put(("engine_reload_failed", f"Engine reload failed: {exc}"))

        getattr(self, "reload_worker", self.worker).submit(work)

    def _engine_reload_watchdog(self) -> None:
        self.engine_reload_watchdog_after = None
        if not self.engine_reload_running:
            return
        message = (
            "Analytics reload is taking unusually long. The UI is still alive; "
            "the database may be busy rebuilding. Completion will be applied "
            "automatically when the read finishes."
        )
        self.status_label.configure(text="Analytics reload waiting for database…")
        self._append_console(message + "\n")

    @staticmethod
    def _parse_manual(text: str) -> list[Any]:
        output: list[Any] = []
        for token in text.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                role, champion = token.split(":", 1)
                output.append((role.strip(), champion.strip()))
            else:
                output.append(token)
        return output

    def _manual_evaluate(self) -> None:
        allies = self._parse_manual(self.manual_ally.get())
        enemies = self._parse_manual(self.manual_enemy.get())
        bans = self._parse_manual(self.manual_bans.get())
        active_role = self.manual_role_menu.get()
        ally_context = [
            {"role": value[0], "champion": value[1], "locked": True}
            if isinstance(value, tuple) else {"champion": value, "locked": True}
            for value in allies
        ]
        self._submit_evaluation(
            allies,
            enemies,
            bans,
            active_role=active_role,
            ally_context=ally_context,
            target="manual",
        )

    def _submit_evaluation(
        self,
        allies: Any,
        enemies: Any,
        bans: Any,
        *,
        active_role: str | None,
        ally_context: Any,
        target: str = "live",
        ban_state_key: tuple[Any, ...] | None = None,
    ) -> None:
        """Use the proven v2 generation model: every meaningful state can run.

        Results are accepted only when their generation is still current. This is
        simpler and more reliable than the v3 single-in-flight state machine, which
        could lose its release event and permanently stop processing later drafts.
        """
        target = "manual" if target == "manual" else "live"
        self.analysis_generations[target] += 1
        generation = self.analysis_generations[target]
        engine = self.engine
        request = (
            generation,
            allies,
            enemies,
            bans,
            active_role,
            ally_context,
            target,
            engine,
            ban_state_key,
        )
        if target == "live":
            self.status_label.configure(text="Updating live recommendations…")
        executor = getattr(self, "analysis_worker", self.worker)
        executor.submit(self._run_evaluation_request, request)

    def _run_evaluation_request(self, request: tuple[Any, ...]) -> None:
        if len(request) == 8:  # compatibility with older tests/callers
            (
                generation,
                allies,
                enemies,
                bans,
                active_role,
                ally_context,
                target,
                engine,
            ) = request
        else:
            (
                generation,
                allies,
                enemies,
                bans,
                active_role,
                ally_context,
                target,
                engine,
                _ban_state_key,
            ) = request
        try:
            # Hold the selected engine's re-entrant lock through the insights copy
            # so another evaluation cannot replace ``last_insights`` between the
            # score result and the event payload. Reloads use a different engine.
            with engine._lock:
                picks = engine.evaluate_draft(
                    allies, enemies, bans, active_role=active_role
                )
                insights = engine.last_insights
            if target == "live":
                # Live bans are deliberately handled by the dedicated latest-state
                # ban worker. Returning here prevents old pick generations from
                # creating a serialized backlog of obsolete ban calculations.
                self.events.put((
                    "live_analysis_picks",
                    (generation, picks, active_role, insights),
                ))
                return

            ban_output = engine.evaluate_bans(
                ally_context,
                enemies,
                bans,
                top_n=int(self.profile["ui"].get("ban_top_n", 5)),
            )
            self.events.put((
                "analysis",
                (generation, picks, ban_output, active_role, target, insights),
            ))
        except Exception as exc:
            self.events.put((
                "analysis_failed",
                (target, generation, f"Evaluation failed: {exc}"),
            ))

    def _submit_live_ban_evaluation(self, snapshot: DraftSnapshot) -> None:
        if not snapshot.bans_pending or self.ban_phase_over:
            return
        """Queue only the newest ban state on a dedicated worker.

        The executor has one worker so SQLite/model state is never read by many
        simultaneous ban jobs. Every queued job checks its generation before it
        starts; obsolete jobs therefore return immediately instead of delaying
        the newest Champion Select state.
        """
        self.ban_generation += 1
        generation = self.ban_generation
        request = (
            generation,
            snapshot.allied_context,
            snapshot.locked_enemies,
            snapshot.all_bans,
            snapshot.active_role,
            snapshot.ban_key,
            self.engine,
            int(self.profile["ui"].get("ban_top_n", 5)),
        )
        if self.ban_watchdog_after:
            try:
                self.after_cancel(self.ban_watchdog_after)
            except Exception:
                pass
        try:
            self.ban_watchdog_after = self.after(
                8000, lambda g=generation: self._ban_watchdog(g)
            )
        except Exception:
            self.ban_watchdog_after = None
        getattr(self, "ban_worker", getattr(self, "analysis_worker", self.worker)).submit(
            self._run_live_ban_request, request
        )

    def _run_live_ban_request(self, request: tuple[Any, ...]) -> None:
        (
            generation,
            ally_context,
            enemies,
            bans,
            active_role,
            ban_state_key,
            engine,
            top_n,
        ) = request
        # A newer request may have been queued while this one waited for the
        # single ban worker. Skip obsolete work before acquiring the engine lock.
        if generation != self.ban_generation or self.shutdown_event.is_set():
            return
        started = time.perf_counter()
        try:
            output = engine.evaluate_bans(
                ally_context, enemies, bans, top_n=top_n
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.events.put((
                "live_bans_ready",
                (generation, output, active_role, ban_state_key, elapsed_ms),
            ))
        except Exception as exc:
            LOGGER.exception("Live ban evaluation failed.")
            self.events.put((
                "live_bans_failed",
                (generation, ban_state_key, f"Ban evaluation failed: {exc}"),
            ))

    def _ban_watchdog(self, generation: int) -> None:
        self.ban_watchdog_after = None
        if (
            generation != self.ban_generation
            or self.ban_phase_over
            or not self.champ_select_active.is_set()
        ):
            return
        message = "Ban calculation is taking longer than expected…"
        for role in ROLES:
            frame = self.role_ban_frames[role]
            self._clear_frame(frame)
            ctk.CTkLabel(frame, text=message, text_color="gray60").pack(
                anchor="w", padx=5, pady=2
            )
        self._append_console(
            "Ban evaluation exceeded 8 seconds. The newest result will still be "
            "shown when ready; see app.log if this repeats.\n"
        )

    def _finish_ban_watchdog(self, generation: int) -> None:
        if generation != self.ban_generation:
            return
        if self.ban_watchdog_after:
            try:
                self.after_cancel(self.ban_watchdog_after)
            except Exception:
                pass
            self.ban_watchdog_after = None

    def _handle_lcu_session(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            # The fallback poll returns 404/None every 750 ms while the user is
            # not in Champion Select. Rebuilding every waiting label on each
            # poll caused the idle screen to visibly flash. Render the inactive
            # state only on the actual active -> inactive transition.
            was_active = (
                self.champ_select_active.is_set()
                or self.snapshot is not EMPTY_SNAPSHOT
                or self.last_draft_key is not None
            )
            if not was_active:
                return
            self.snapshot = EMPTY_SNAPSHOT
            self.last_draft_key = None
            self.last_ban_key = None
            self.last_submitted_ban_key = None
            self.champ_select_active.clear()
            self.analysis_generations["live"] += 1
            self.ban_generation += 1
            self.ban_phase_over = False
            self._finish_ban_watchdog(self.ban_generation)
            self.phase_label.configure(text=EMPTY_SNAPSHOT.phase)
            self.active_role_label.configure(text="Your role: unknown")
            self._show_empty_state()
            return

        # A websocket UPDATE can occasionally contain only a fragment of the
        # session. Never let such a fragment erase a valid board; the LCU bridge
        # fallback poll will supply the canonical full session within 750 ms.
        if not isinstance(payload.get("myTeam"), list) or not isinstance(
            payload.get("theirTeam"), list
        ):
            timer = payload.get("timer")
            if isinstance(timer, Mapping):
                phase = str(timer.get("phase", "Champion Select")).replace("_", " ").title()
                self.phase_label.configure(text=phase)
            return

        self.champ_select_active.set()
        include_hovers = bool(
            self.profile["ui"].get("include_hover_intents_on_board", True)
        )
        snapshot = parse_lcu_session(
            payload,
            include_hover_intents=include_hovers,
            previous_snapshot=self.snapshot if self.snapshot is not EMPTY_SNAPSHOT else None,
        )
        if not snapshot.active_role and snapshot.local_champion_id:
            inferred = self.engine.infer_role_for_champion(snapshot.local_champion_id)
            if inferred:
                snapshot = replace(snapshot, active_role=inferred)
        self.snapshot = snapshot
        self.phase_label.configure(text=snapshot.phase)
        active_text = ROLE_LABELS.get(snapshot.active_role or "", "unknown")
        self.active_role_label.configure(text=f"Your role: {active_text}")

        # The LCU broadcasts timer updates frequently. Only draft-key changes
        # trigger scoring/rendering, and bursts are coalesced by a short debounce.
        if snapshot.draft_key == self.last_draft_key:
            return
        # Ban advice is only meaningful while a ban action is still pending.
        # Once all bans have been completed (no pending ban actions and at
        # least one ban or locked pick observed), permanently stop recomputing
        # and clear the panels so stale advice cannot linger or churn through
        # the rest of the draft.
        if not snapshot.bans_pending and (
            snapshot.all_bans or snapshot.locked_allies or snapshot.locked_enemies
        ):
            if not self.ban_phase_over:
                # Transition to the post-ban state exactly once: invalidate any
                # in-flight ban result and hide the panels so later picks cannot
                # resurrect stale ban advice.
                self.ban_phase_over = True
                self.ban_generation += 1
                self.last_submitted_ban_key = snapshot.ban_key
                self.last_ban_key = snapshot.ban_key
                self.current_bans = {role: [] for role in ROLES}
                self.ban_fingerprints.clear()
                for role in ROLES:
                    frame = self.role_ban_frames[role]
                    self._clear_frame(frame)
                    ctk.CTkLabel(
                        frame,
                        text="Ban phase over",
                        text_color="gray60",
                    ).pack(anchor="w", padx=5, pady=2)
        elif snapshot.ban_key != self.last_ban_key:
            self.last_ban_key = snapshot.ban_key
            if snapshot.bans_pending:
                self.current_bans = {role: [] for role in ROLES}
                self.ban_fingerprints.clear()
                for role in ROLES:
                    frame = self.role_ban_frames[role]
                    self._clear_frame(frame)
                    ctk.CTkLabel(
                        frame,
                        text="Calculating ban priorities…",
                        text_color="gray60",
                    ).pack(anchor="w", padx=5, pady=2)
        self.last_draft_key = snapshot.draft_key
        LOGGER.info(
            "Live draft changed: allies=%d locked=%d enemies=%d locked=%d bans=%d role=%s",
            len(snapshot.allies), len(snapshot.locked_allies),
            len(snapshot.enemies), len(snapshot.locked_enemies),
            len(snapshot.all_bans), snapshot.active_role or "unknown",
        )
        self._update_board_label()
        if self.pending_evaluation_after:
            try:
                self.after_cancel(self.pending_evaluation_after)
            except Exception:
                pass
        self.pending_evaluation_after = self.after(90, self._evaluate_current_snapshot)

    def _evaluate_current_snapshot(self) -> None:
        self.pending_evaluation_after = None
        snapshot = self.snapshot
        self._submit_evaluation(
            snapshot.locked_allies,
            snapshot.locked_enemies,
            snapshot.all_bans,
            active_role=snapshot.active_role,
            ally_context=snapshot.allied_context,
            target="live",
            ban_state_key=snapshot.ban_key,
        )
        if (
            snapshot.bans_pending
            and not self.ban_phase_over
            and snapshot.ban_key != self.last_submitted_ban_key
        ):
            self.last_submitted_ban_key = snapshot.ban_key
            self._submit_live_ban_evaluation(snapshot)

    def _update_board_label(self) -> None:
        def names(values: tuple[BoardChampion, ...]) -> str:
            labels: list[str] = []
            for value in values:
                name = self.catalog.name_for_id(value.champion_id)
                suffix = "" if value.locked else " (hover)"
                labels.append(name + suffix)
            return ", ".join(labels) or "—"

        bans = ", ".join(
            self.catalog.name_for_id(x) for x in self.snapshot.all_bans
        ) or "—"
        self.board_label.configure(
            text=(
                f"Allies: {names(self.snapshot.allies)}     "
                f"Enemies: {names(self.snapshot.enemies)}     Bans: {bans}"
            )
        )

    @staticmethod
    def _recommendation_fingerprint(values: Sequence[Recommendation]) -> tuple[Any, ...]:
        def options_fingerprint(options: Sequence[BuildOption]) -> tuple[Any, ...]:
            return tuple(
                (
                    option.ids,
                    option.games,
                    round(option.win_rate, 6),
                    round(option.adjusted_win_rate, 6),
                    round(option.recommendation_score, 6),
                    option.context_note,
                )
                for option in options
            )

        def loadouts_fingerprint(options: Sequence[LoadoutOption]) -> tuple[Any, ...]:
            return tuple(
                (
                    option.item_ids,
                    option.rune_ids,
                    option.games,
                    round(option.win_rate, 6),
                    round(option.adjusted_win_rate, 6),
                    round(option.recommendation_score, 6),
                    option.context_note,
                )
                for option in options
            )

        return tuple(
            (
                rec.champion_id,
                rec.selected,
                round(rec.score, 6),
                round(rec.confidence_score, 3),
                round(rec.role_confidence, 3),
                round(rec.patch_freshness, 6),
                rec.champion_damage_profile,
                round(rec.champion_physical_share, 6),
                round(rec.champion_magic_share, 6),
                round(rec.champion_true_share, 6),
                round(rec.synergy_delta, 6),
                round(rec.counter_delta, 6),
                round(rec.composition_score, 6),
                round(rec.ml_uplift, 6),
                round(rec.ml_ensemble_std, 6),
                round(rec.personal_multiplier, 6),
                rec.explanation_summary,
                rec.strengths,
                rec.weaknesses,
                options_fingerprint(rec.item_builds),
                options_fingerprint(rec.rune_pages),
                options_fingerprint(rec.summoner_spells),
                loadouts_fingerprint(rec.loadouts),
            )
            for rec in values
        )

    @staticmethod
    def _ban_fingerprint(values: Sequence[BanRecommendation]) -> tuple[Any, ...]:
        return tuple(
            (
                rec.champion_id,
                round(rec.score, 6),
                rec.target_ally_id,
                rec.target_is_hover,
            )
            for rec in values
        )

    @staticmethod
    def _clear_frame(frame: ctk.CTkFrame) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def _show_empty_state(self) -> None:
        self.role_fingerprints.clear()
        self.ban_fingerprints.clear()
        self.pending_role_renders.clear()
        self.role_render_scheduled = False
        self.current_recommendations = {role: [] for role in ROLES}
        self.current_bans = {role: [] for role in ROLES}
        self.last_ban_key = None
        self.last_submitted_ban_key = None
        self.ban_phase_over = False
        self.last_live_render_generation = 0
        message = "No precomputed analytics yet. Use Data Watcher → Rebuild analytics only."
        if self.engine.ready:
            message = "Waiting for Champion Select, or use the manual test fields above."
        for role in ROLES:
            self._clear_frame(self.role_ban_frames[role])
            self._clear_frame(self.role_pick_frames[role])
            ctk.CTkLabel(
                self.role_ban_frames[role], text="Waiting for draft", text_color="gray60"
            ).pack(anchor="w", padx=5, pady=2)
            ctk.CTkLabel(
                self.role_pick_frames[role],
                text=message,
                wraplength=245,
                justify="left",
            ).pack(padx=9, pady=14)

    @staticmethod
    def _option_text(options: tuple[BuildOption, ...], empty: str) -> str:
        if not options:
            return empty
        first = options[0]
        return " · ".join(first.names) + f"  ({first.games}g, {first.win_rate:.0%})"

    @staticmethod
    def _loadout_text(options: tuple[LoadoutOption, ...], empty: str) -> str:
        if not options:
            return empty
        first = options[0]
        items = " · ".join(first.item_names) or "Unknown items"
        runes = " · ".join(first.rune_names[:4]) or "Unknown runes"
        return f"{items}  +  {runes}  ({first.games}g, {first.win_rate:.0%})"

    def _render_manual_analysis(
        self,
        picks: dict[str, list[Recommendation]],
        bans: dict[str, list[BanRecommendation]],
        active_role: str | None,
        insights: DraftInsights,
    ) -> None:
        lines: list[str] = []
        lines.append(
            f"Predicted allied win probability: {insights.predicted_win_probability:.1%} "
            f"(confidence {insights.prediction_confidence:.0f}%)"
        )
        lines.append(f"Draft summary: {insights.summary}")
        lines.append("")
        for role in ROLES:
            role_values = picks.get(role, [])
            lines.append(f"{ROLE_LABELS[role]}{' · YOUR ROLE' if role == active_role else ''}")
            if role_values:
                for rank, rec in enumerate(role_values, start=1):
                    locked = " [LOCKED]" if rec.selected else ""
                    lines.append(
                        f"  {rank:>2}. {rec.champion_name}{locked}  score={rec.score:.3f}  "
                        f"confidence={rec.confidence_score:.0f}%  role={rec.role_confidence:.0f}%  "
                        f"WR={rec.global_win_rate:.1%}  ML={rec.ml_win_probability:.1%}"
                    )
                    if rec.explanation_summary:
                        lines.append(f"      {rec.explanation_summary}")
            else:
                lines.append("  No eligible recommendation data.")
            ban_values = bans.get(role, [])
            if ban_values:
                lines.append("  Bans: " + ", ".join(
                    f"{value.champion_name} ({value.confidence_score:.0f}%)"
                    for value in ban_values
                ))
            lines.append("")
        self.manual_output.configure(state="normal")
        self.manual_output.delete("1.0", "end")
        self.manual_output.insert("1.0", "\n".join(lines))
        self.manual_output.configure(state="disabled")

    def _render_insights(self, insights: DraftInsights) -> None:
        self.insight_prediction_label.configure(
            text=(
                f"Predicted allied win probability: {insights.predicted_win_probability:.1%} "
                f"· confidence {insights.prediction_confidence:.0f}%"
            )
        )
        ally = insights.ally_composition
        enemy = insights.enemy_composition
        self.insight_damage_label.configure(
            text=(
                f"Allied damage: {ally.damage_profile} · {ally.physical_share:.0%} physical / "
                f"{ally.magic_share:.0%} magic / {ally.true_share:.0%} true · "
                f"feature coverage {ally.feature_confidence:.0%}\n"
                f"Enemy damage: {enemy.damage_profile} · {enemy.physical_share:.0%} physical / "
                f"{enemy.magic_share:.0%} magic / {enemy.true_share:.0%} true · "
                f"feature coverage {enemy.feature_confidence:.0%}"
            )
        )
        strength_lines = ["STRENGTHS", *[f"• {value}" for value in insights.strengths], "", "WEAKNESSES", *[f"• {value}" for value in insights.weaknesses]]
        self.insight_strengths_text.configure(state="normal")
        self.insight_strengths_text.delete("1.0", "end")
        self.insight_strengths_text.insert("1.0", "\n".join(strength_lines))
        self.insight_strengths_text.configure(state="disabled")

        def inference_lines(label: str, inference: Any) -> list[str]:
            values = [f"{label} · assignment confidence {inference.assignment_confidence:.0f}%"]
            if not inference.guesses:
                values.append("  No champions selected.")
                return values
            for guess in inference.guesses:
                name = self.catalog.name_for_id(guess.champion_id)
                alternatives = ", ".join(
                    f"{ROLE_LABELS.get(role, role)} {probability:.0%}"
                    for role, probability in guess.alternatives
                )
                profile = self.engine.champion_profile(guess.champion_id, guess.role)
                values.append(
                    f"  {name}: {ROLE_LABELS.get(guess.role, guess.role)} "
                    f"({guess.confidence:.0f}%) · {alternatives}\n"
                    f"      Damage {profile['damage_profile']}: {profile['physical_share']:.0%} physical / "
                    f"{profile['magic_share']:.0%} magic / {profile['true_share']:.0%} true"
                )
            return values

        role_lines = inference_lines("ALLIES", insights.ally_role_inference)
        role_lines.extend(["", *inference_lines("ENEMIES", insights.enemy_role_inference)])
        self.insight_role_text.configure(state="normal")
        self.insight_role_text.delete("1.0", "end")
        self.insight_role_text.insert("1.0", "\n".join(role_lines))
        self.insight_role_text.configure(state="disabled")

    def _show_embedding_neighbors(self) -> None:
        supplied = self.embedding_entry.get().strip()
        if not supplied:
            self.embedding_output.configure(text="Enter a champion name.")
            return
        values = self.engine.nearest_champions(supplied)
        if not values:
            self.embedding_output.configure(
                text="No embedding is available yet. Build the v3 neural model first."
            )
            return
        self.embedding_output.configure(
            text=" · ".join(f"{name} {similarity:.2f}" for name, similarity in values)
        )

    def _refresh_model_status(self) -> None:
        status = self.engine.model_status
        summary = self.engine.summary
        available = "available" if status.available else "not trained"
        backend = status.backend if status.available else (summary.ml_backend or status.backend)
        device = status.device if status.available else (summary.ml_device or status.device or "not checked")
        trained_matches = status.trained_matches if status.available else summary.ml_matches
        validation_accuracy = status.validation_accuracy if status.available else summary.ml_validation_accuracy
        validation_brier = status.validation_brier if status.available else summary.ml_validation_brier
        reason = ""
        if not status.available:
            reason = summary.ml_reason or (
                "No exported neural model exists yet. Run setup_gpu_ml.bat, verify with Check GPU, "
                "then rebuild analytics + model."
            )
        reason_line = f"\nLast training result: {reason}" if reason else ""
        self.model_status_label.configure(
            text=(
                f"Neural ensemble: {available} · backend {backend or 'not checked'} · training device {device or 'not checked'}\n"
                f"Trained matches: {trained_matches:,} · embedding dimensions: {status.embedding_dimension} · "
                f"validation accuracy {validation_accuracy:.1%} · Brier {validation_brier:.4f}\n"
                f"Analytics version: {summary.analytics_version or 'not built'} · current weighted patch: "
                f"{status.current_patch or summary.static_patch or 'unknown'} · built {status.built_at or summary.analytics_built_at or 'never'}"
                f"{reason_line}"
            )
        )

    def _check_gpu_async(self) -> None:
        self.model_status_label.configure(
            text="Checking PyTorch/CUDA in the shared project virtual environment…"
        )

        def work() -> None:
            python_path = self._project_python() or Path(sys.executable)
            probe_script = PROJECT_ROOT / "gpu_probe.py"
            if not probe_script.is_file():
                self.events.put((
                    "gpu_status",
                    {
                        "available": False,
                        "error": f"GPU probe is missing: {probe_script}",
                        "python": str(python_path),
                    },
                ))
                return
            try:
                completed = subprocess.run(
                    [str(python_path), str(probe_script), "--json"],
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=90,
                    **_background_popen_kwargs(),
                )
                lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
                payload: dict[str, Any] = {}
                for line in reversed(lines):
                    try:
                        payload = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
                if not payload:
                    payload = {
                        "available": False,
                        "error": completed.stdout.strip() or f"GPU probe exited with code {completed.returncode}",
                    }
                else:
                    payload["available"] = bool(payload.get("cuda_available", False))
                    payload["torch"] = str(payload.get("torch_version", "none"))
                    payload["cuda"] = str(payload.get("cuda_runtime", "none"))
                    payload["count"] = int(payload.get("device_count", 0) or 0)
                self.events.put(("gpu_status", payload))
            except Exception as exc:
                self.events.put((
                    "gpu_status",
                    {
                        "available": False,
                        "error": str(exc),
                        "python": str(python_path),
                    },
                ))

        self.worker.submit(work)

    def _render_analysis(
        self,
        picks: dict[str, list[Recommendation]] | None,
        bans: dict[str, list[BanRecommendation]] | None,
        active_role: str | None,
    ) -> None:
        if picks is not None:
            self.current_recommendations = picks
        if bans is not None:
            self.current_bans = bans
        for role in ROLES:
            values = self.current_recommendations.get(role, [])
            ban_values = self.current_bans.get(role, [])
            title = ROLE_LABELS[role]
            if role == active_role:
                title += " · YOU"
            if values and values[0].selected:
                title += " · LOCKED"
            self.role_titles[role].configure(text=title)

            pending_picks: Sequence[Recommendation] | None = None
            pending_bans: Sequence[BanRecommendation] | None = None
            if bans is not None:
                ban_fp = self._ban_fingerprint(ban_values)
                if self.ban_fingerprints.get(role) != ban_fp:
                    self.ban_fingerprints[role] = ban_fp
                    pending_bans = tuple(ban_values)

            if picks is not None:
                pick_fp = self._recommendation_fingerprint(values)
                if self.role_fingerprints.get(role) != pick_fp:
                    self.role_fingerprints[role] = pick_fp
                    pending_picks = tuple(values)

            if pending_picks is not None or pending_bans is not None:
                previous = self.pending_role_renders.get(role, (None, None))
                self.pending_role_renders[role] = (
                    pending_picks if pending_picks is not None else previous[0],
                    pending_bans if pending_bans is not None else previous[1],
                )

        if self.pending_role_renders and not self.role_render_scheduled:
            self.role_render_scheduled = True
            self.after_idle(self._flush_one_role_render)

    def _flush_one_role_render(self) -> None:
        """Render one role per idle slice so widget creation never blocks typing."""
        if not self.pending_role_renders:
            self.role_render_scheduled = False
            return
        role = next(iter(self.pending_role_renders))
        picks, bans = self.pending_role_renders.pop(role)
        if bans is not None:
            self._render_bans(role, bans)
        if picks is not None:
            self._render_role_picks(role, picks)
        if self.pending_role_renders:
            self.after_idle(self._flush_one_role_render)
        else:
            self.role_render_scheduled = False

    def _render_bans(self, role: str, values: Sequence[BanRecommendation]) -> None:
        frame = self.role_ban_frames[role]
        self._clear_frame(frame)
        if not values:
            enemy_filled = any(
                champion.locked and champion.role == role for champion in self.snapshot.enemies
            )
            text = "Enemy role already picked" if enemy_filled else "No eligible data"
            ctk.CTkLabel(frame, text=text, text_color="gray60").pack(
                anchor="w", padx=5, pady=2
            )
            return
        for rank, rec in enumerate(values, start=1):
            detail = ""
            if rec.target_ally_name:
                hover = "hovered " if rec.target_is_hover else ""
                detail = f" · vs {hover}{rec.target_ally_name}"
            ctk.CTkLabel(
                frame,
                text=f"{rank}. {rec.champion_name}{detail} · {rec.confidence_score:.0f}% conf",
                anchor="w",
                justify="left",
                font=ctk.CTkFont(size=10),
            ).pack(fill="x", padx=5, pady=0)

    def _render_role_picks(self, role: str, values: Sequence[Recommendation]) -> None:
        frame = self.role_pick_frames[role]
        self._clear_frame(frame)
        if not values:
            ctk.CTkLabel(frame, text="No eligible data", text_color="gray70").pack(pady=18)
            return
        for rank, rec in enumerate(values, start=1):
            self._create_recommendation_card(frame, rank, rec)

    def _create_recommendation_card(
        self, frame: ctk.CTkFrame, rank: int, rec: Recommendation
    ) -> None:
        card = ctk.CTkFrame(frame)
        card.pack(fill="x", padx=3, pady=3)
        portrait = ctk.CTkButton(
            card,
            text="LOCK" if rec.selected else str(rank),
            width=48,
            height=48,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=lambda value=rec: self._open_champion_details(value),
        )
        portrait.grid(row=0, column=0, rowspan=3, padx=6, pady=6)
        title = ("Locked: " if rec.selected else f"{rank}. ") + rec.champion_name
        ctk.CTkButton(
            card,
            text=title,
            anchor="w",
            fg_color="transparent",
            hover_color=("gray78", "gray25"),
            font=ctk.CTkFont(size=13, weight="bold"),
            command=lambda value=rec: self._open_champion_details(value),
        ).grid(row=0, column=1, sticky="ew", padx=1, pady=(3, 0))
        pool_text = ""
        if rec.personal_multiplier != 1.0:
            pool_text = f"  {rec.pool_category.title()} ×{rec.personal_multiplier:g}"
        ctk.CTkLabel(
            card,
            text=(
                f"Score {rec.score:.3f}{pool_text}  Confidence {rec.confidence_score:.0f}%  "
                f"Role {rec.role_confidence:.0f}%\n"
                f"WR {rec.global_win_rate:.1%}/{rec.global_games}g  Patch {rec.patch_freshness:.0%}  "
                f"{rec.champion_damage_profile} "
                f"{rec.champion_physical_share:.0%}P/{rec.champion_magic_share:.0%}M/{rec.champion_true_share:.0%}T\n"
                f"Syn {rec.synergy_delta:+.1%}  Ctr {rec.counter_delta:+.1%}\n"
                f"Comp {rec.composition_score:.0%}  ML {rec.ml_win_probability:.1%} "
                f"(Δ{rec.ml_uplift:+.1%}, σ{rec.ml_ensemble_std:.2%})\n"
                f"Loadout: {self._loadout_text(rec.loadouts, 'insufficient sample')}\n"
                f"Spells: {self._option_text(rec.summoner_spells, 'insufficient sample')}"
            ),
            justify="left",
            anchor="w",
            wraplength=225,
            font=ctk.CTkFont(size=10),
        ).grid(row=1, column=1, sticky="ew", padx=4, pady=(0, 2))
        ctk.CTkButton(
            card,
            text="Visual loadout details",
            height=24,
            command=lambda value=rec: self._open_champion_details(value),
        ).grid(row=2, column=1, sticky="ew", padx=4, pady=(0, 5))
        card.grid_columnconfigure(1, weight=1)
        self._load_champion_portrait(rec.champion_id, portrait, (48, 48))

    def _open_champion_details(self, rec: Recommendation) -> None:
        window = ctk.CTkToplevel(self)
        self.open_detail_windows.add(window)
        window.title(f"{rec.champion_name} · {ROLE_LABELS[rec.role]} details")
        window.geometry("920x760")
        window.minsize(720, 600)
        window.transient(self)
        window.protocol("WM_DELETE_WINDOW", lambda: self._close_detail_window(window))

        detail_scroll = DualAxisScrollableFrame(window)
        detail_scroll.pack(fill="both", expand=True, padx=8, pady=8)
        scroll = detail_scroll.content
        header = ctk.CTkFrame(scroll)
        header.pack(fill="x", pady=(0, 8))
        portrait = ctk.CTkLabel(header, text=rec.champion_name, width=96, height=96)
        portrait.pack(side="left", padx=12, pady=12)
        self._load_champion_portrait(rec.champion_id, portrait, (96, 96))
        title_text = f"{rec.champion_name} · {ROLE_LABELS[rec.role]}"
        if rec.selected:
            title_text += " · LOCKED"
        ctk.CTkLabel(
            header,
            text=title_text,
            font=ctk.CTkFont(size=24, weight="bold"),
            anchor="w",
        ).pack(anchor="w", padx=8, pady=(15, 3))
        ctk.CTkLabel(
            header,
            text=(
                f"Score {rec.score:.4f}   Global WR {rec.global_win_rate:.1%} ({rec.global_games:,} games)\n"
                f"Synergy {rec.synergy_delta:+.1%}   Lane counter {rec.counter_delta:+.1%}   "
                f"Composition {rec.composition_score:.0%}   ML probability {rec.ml_win_probability:.1%}\n"
                f"Confidence {rec.confidence_score:.0f}%   Role confidence {rec.role_confidence:.0f}%   "
                f"Patch freshness {rec.patch_freshness:.0%}   Ensemble σ {rec.ml_ensemble_std:.2%}\n"
                f"Champion damage: {rec.champion_damage_profile} · "
                f"{rec.champion_physical_share:.0%} physical / {rec.champion_magic_share:.0%} magic / "
                f"{rec.champion_true_share:.0%} true"
            ),
            justify="left",
            anchor="w",
        ).pack(anchor="w", padx=8, pady=(0, 12))

        explanation = ctk.CTkFrame(scroll)
        explanation.pack(fill="x", padx=4, pady=(0, 8))
        ctk.CTkLabel(
            explanation,
            text="WHY THIS PICK",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(anchor="w", padx=9, pady=(8, 3))
        explanation_text = [rec.explanation_summary]
        explanation_text.extend(f"✓ {value}" for value in rec.strengths)
        explanation_text.extend(f"△ {value}" for value in rec.weaknesses)
        ctk.CTkLabel(
            explanation,
            text="\n".join(value for value in explanation_text if value),
            justify="left",
            anchor="w",
            wraplength=820,
        ).pack(fill="x", padx=10, pady=(0, 10))

        self._detail_loadout_section(
            scroll,
            "RECOMMENDED LOADOUTS · item core + rune page",
            rec.loadouts,
        )
        self._detail_option_section(
            scroll, "SUMMONER SPELLS", rec.summoner_spells, "spell"
        )

        composition = rec.composition
        ctk.CTkLabel(
            scroll,
            text="TEAM COMPOSITION AFTER THIS PICK",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(anchor="w", padx=4, pady=(12, 3))
        ctk.CTkLabel(
            scroll,
            text=(
                f"Damage balance {composition.damage_balance:.0%} · Physical {composition.physical_share:.0%} · "
                f"Magic {composition.magic_share:.0%} · True {composition.true_share:.0%}\n"
                f"Profile {composition.damage_profile} · Frontline {composition.frontline:.0%} · "
                f"Control {composition.control:.0%} · Hard CC {composition.hard_cc:.0%} · "
                f"Engage {composition.engage:.0%} · Pick {composition.pick_potential:.0%}\n"
                f"Wave clear {composition.waveclear:.0%} · Objective {composition.objective:.0%} · "
                f"Mobility {composition.mobility:.0%} · Vision {composition.vision:.0%} · "
                f"Early/Mid/Late {composition.early_strength:.0%}/{composition.mid_strength:.0%}/{composition.late_strength:.0%} · "
                f"feature coverage {composition.feature_confidence:.0%}"
            ),
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=8, pady=(0, 12))

    def _close_detail_window(self, window: ctk.CTkToplevel) -> None:
        self.open_detail_windows.discard(window)
        window.destroy()

    def _detail_option_section(
        self,
        parent: Any,
        heading: str,
        options: Sequence[BuildOption],
        icon_kind: str,
    ) -> None:
        ctk.CTkLabel(
            parent,
            text=heading,
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(anchor="w", padx=4, pady=(10, 3))
        if not options:
            ctk.CTkLabel(
                parent,
                text="Insufficient sample size in the local database.",
                text_color="gray65",
            ).pack(anchor="w", padx=8, pady=5)
            return
        for rank, option in enumerate(options[:3], start=1):
            card = ctk.CTkFrame(parent)
            card.pack(fill="x", padx=4, pady=4)
            ctk.CTkLabel(
                card,
                text=f"#{rank}",
                width=42,
                font=ctk.CTkFont(size=16, weight="bold"),
            ).pack(side="left", padx=8, pady=10)
            body = ctk.CTkFrame(card, fg_color="transparent")
            body.pack(side="left", fill="both", expand=True, padx=3, pady=5)
            icon_row = ctk.CTkFrame(body, fg_color="transparent")
            icon_row.pack(anchor="w")
            identifiers = list(option.ids)
            if icon_kind == "rune":
                styles = [x for x in (option.primary_style_id, option.sub_style_id) if x > 0]
                identifiers = styles + identifiers
            for identifier in identifiers[:8]:
                label = ctk.CTkLabel(icon_row, text=str(identifier), width=46, height=46)
                label.pack(side="left", padx=3, pady=2)
                self._load_static_icon(icon_kind, int(identifier), label, (42, 42))
            names = " · ".join(option.names) or "Unknown option"
            note = option.context_note or "Ranked by adjusted win rate and sample size"
            ctk.CTkLabel(
                body,
                text=(
                    f"{names}\n{option.games:,} games · {option.win_rate:.1%} raw WR · "
                    f"{option.adjusted_win_rate:.1%} adjusted WR\n{note}"
                ),
                justify="left",
                anchor="w",
                wraplength=730,
            ).pack(fill="x", padx=3, pady=(2, 5))

    def _detail_loadout_section(
        self,
        parent: Any,
        heading: str,
        options: Sequence[LoadoutOption],
    ) -> None:
        ctk.CTkLabel(
            parent,
            text=heading,
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(anchor="w", padx=4, pady=(10, 3))
        if not options:
            ctk.CTkLabel(
                parent,
                text="Insufficient paired item-and-rune samples in the local database.",
                text_color="gray65",
            ).pack(anchor="w", padx=8, pady=5)
            return
        for rank, option in enumerate(options[:3], start=1):
            card = ctk.CTkFrame(parent)
            card.pack(fill="x", padx=4, pady=4)
            ctk.CTkLabel(
                card,
                text=f"#{rank}",
                width=42,
                font=ctk.CTkFont(size=16, weight="bold"),
            ).pack(side="left", padx=8, pady=10)
            body = ctk.CTkFrame(card, fg_color="transparent")
            body.pack(side="left", fill="both", expand=True, padx=3, pady=5)

            item_row = ctk.CTkFrame(body, fg_color="transparent")
            item_row.pack(anchor="w")
            ctk.CTkLabel(item_row, text="Items", width=52, anchor="w").pack(
                side="left", padx=(0, 4)
            )
            for identifier in option.item_ids[:6]:
                label = ctk.CTkLabel(item_row, text=str(identifier), width=46, height=46)
                label.pack(side="left", padx=3, pady=2)
                self._load_static_icon("item", int(identifier), label, (42, 42))

            rune_row = ctk.CTkFrame(body, fg_color="transparent")
            rune_row.pack(anchor="w")
            ctk.CTkLabel(rune_row, text="Runes", width=52, anchor="w").pack(
                side="left", padx=(0, 4)
            )
            rune_identifiers = [
                value
                for value in (option.primary_style_id, option.sub_style_id)
                if value > 0
            ] + list(option.rune_ids)
            for identifier in rune_identifiers[:8]:
                label = ctk.CTkLabel(rune_row, text=str(identifier), width=46, height=46)
                label.pack(side="left", padx=3, pady=2)
                self._load_static_icon("rune", int(identifier), label, (42, 42))

            item_names = " · ".join(option.item_names) or "Unknown item core"
            rune_names = " · ".join(option.rune_names) or "Unknown rune page"
            note = option.context_note or "Observed paired loadout ranked by patch-weighted performance"
            ctk.CTkLabel(
                body,
                text=(
                    f"{item_names}\n{rune_names}\n"
                    f"{option.games:,} games · {option.win_rate:.1%} raw WR · "
                    f"{option.adjusted_win_rate:.1%} adjusted WR\n{note}"
                ),
                justify="left",
                anchor="w",
                wraplength=730,
            ).pack(fill="x", padx=3, pady=(2, 5))

    def _load_champion_portrait(
        self, champion_id: int, widget: Any, size: tuple[int, int]
    ) -> None:
        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = IMAGE_CACHE_DIR / f"{champion_id}.png"
        self._load_image_async(
            self.catalog.square_url(champion_id), path, widget, size
        )

    def _load_static_icon(
        self,
        kind: str,
        identifier: int,
        widget: Any,
        size: tuple[int, int],
    ) -> None:
        STATIC_IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = STATIC_IMAGE_CACHE_DIR / f"{kind}_{identifier}.png"
        url = self.engine.static.icon_url(kind, identifier)
        if url:
            self._load_image_async(url, path, widget, size)

    def _attach_cached_image(self, widget: Any, cache_key: tuple[str, tuple[int, int]]) -> bool:
        """Attach the CTkImage for *cache_key* to *widget*; True when found."""
        cached = self.image_cache.get(cache_key)
        if cached is None:
            return False
        try:
            if widget.winfo_exists():
                setattr(widget, "_draft_image", cached)
                widget.configure(image=cached, text="")
        except tk.TclError:
            pass
        return True

    def _decoded_source(self, path: Path) -> Image.Image | None:
        """Decode *path* once per file version and share it across all sizes.

        Previously every widget size re-opened and re-decoded the same PNG in a
        worker thread. This cache keeps one RGBA source per (path, mtime), so a
        champion shown at 44px, 48px, and 96px decodes exactly once per session.
        """
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return None
        entry = self.decoded_image_cache.get(path)
        if entry is not None and entry[0] == mtime_ns:
            return entry[1]
        try:
            with Image.open(path) as source:
                decoded = source.convert("RGBA").copy()
        except Exception:
            LOGGER.debug("Image decode failed: %s", path, exc_info=True)
            return None
        with self.decoded_image_cache_lock:
            self.decoded_image_cache[path] = (mtime_ns, decoded)
            while len(self.decoded_image_cache) > DECODER_CACHE_LIMIT:
                self.decoded_image_cache.pop(next(iter(self.decoded_image_cache)))
        return decoded

    def _load_image_async(
        self,
        url: str,
        path: Path,
        widget: Any,
        size: tuple[int, int],
    ) -> None:
        cache_key = (str(path), size)
        if self._attach_cached_image(widget, cache_key):
            return

        # Instant fast-path for icons already cached on disk. Champion squares
        # are tiny; decoding them on the calling thread removes the whole
        # worker->event-queue round trip that previously made icon grids fill
        # in visibly over several seconds.
        try:
            disk_cached = path.exists() and path.stat().st_size <= MAX_SYNC_DECODE_BYTES
        except OSError:
            disk_cached = False
        if disk_cached:
            decoded = self._decoded_source(path)
            if decoded is not None:
                self._apply_ready_image(cache_key, decoded, size)
                self._attach_cached_image(widget, cache_key)
                return

        self.image_waiters.setdefault(cache_key, []).append(widget)
        if cache_key in self.image_loads_inflight:
            return
        self.image_loads_inflight.add(cache_key)

        def work() -> None:
            try:
                if not path.exists():
                    response = _shared_http_session().get(url, timeout=12)
                    response.raise_for_status()
                    _atomic_save_image(path, response.content)
                    if not path.exists():
                        raise RuntimeError("image payload rejected")
                # One retry: a legacy corrupt cache entry is deleted and
                # re-downloaded instead of failing forever.
                for attempt in range(2):
                    decoded_source = self._decoded_source(path)
                    if decoded_source is not None:
                        # Shared cache entry handed straight to the UI; nobody
                        # mutates decoded sources afterwards.
                        self.events.put(("image_ready", (cache_key, decoded_source, size)))
                        return
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        break
                    if attempt or not url:
                        break
                    try:
                        response = _shared_http_session().get(url, timeout=12)
                        response.raise_for_status()
                        _atomic_save_image(path, response.content)
                    except Exception:
                        LOGGER.debug("Image re-download failed: %s", url, exc_info=True)
                        break
                self.events.put(("image_failed", cache_key))
            except Exception:
                LOGGER.debug("Image load failed: %s", url, exc_info=True)
                self.events.put(("image_failed", cache_key))

        self.worker.submit(work)

    def _start_data_job(self, kind: str, *, background: bool = False) -> None:
        # Manual jobs share one lock with the automatic watcher, preventing
        # concurrent analytics/model writes or duplicate scraper sessions.
        if kind == "scrape" and self.api_key_entry.get().strip():
            if not self._save_entered_api_key(validate=False):
                return
        if not self.data_job_lock.acquire(blocking=False):
            self._append_console(
                "A data job or watcher scan is already running. Pause the watcher before starting a manual job.\n"
            )
            return

        profile = load_profile()
        rebuild_after_scrape = bool(
            profile["background_collector"].get(
                "rebuild_analytics_each_batch", True
            )
        )
        self.current_data_job = (kind, False)
        self.job_label.configure(text="Running manual data job…")
        for button in self.data_buttons:
            button.configure(state="disabled")

        project_python = self._project_python()
        command_python = str(project_python or Path(sys.executable))
        if getattr(sys, "frozen", False) and project_python is None:
            self._append_console(f"\n> packaged fallback job: {kind}\n")
            self.worker.submit(
                self._run_packaged_data_job, kind, False, rebuild_after_scrape
            )
            return

        scrape_command = [command_python, "scraper.py"]
        if rebuild_after_scrape:
            scrape_command.append("--build-analytics")
        commands = {
            "scrape": scrape_command,
            "build": [command_python, "analytics_builder.py"],
            "static": [
                command_python,
                "-c",
                "from static_data import StaticDataCatalog; from data_dragon_maps import ChampionCatalog; "
                "StaticDataCatalog.load(refresh=True); ChampionCatalog.load(refresh=True); "
                "print('Static data refreshed')",
            ],
        }
        command = commands[kind]
        self._append_console(f"\n> {' '.join(command)}\n")

        def work() -> None:
            return_code = -1
            try:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    **_background_popen_kwargs(),
                )
                assert process.stdout is not None
                for line in process.stdout:
                    self.events.put(("console", line))
                return_code = process.wait()
            except Exception as exc:
                self.events.put(("console", f"Job failed to start: {exc}\n"))
            finally:
                self.data_job_lock.release()
                self.events.put((
                    "job_done",
                    {
                        "code": return_code,
                        "kind": kind,
                        "background": False,
                        "rebuilt": kind != "scrape" or rebuild_after_scrape,
                    },
                ))

        self.worker.submit(work)

    def _run_packaged_data_job(
        self, kind: str, background: bool, rebuild_after_scrape: bool
    ) -> None:
        return_code = -1
        try:
            if kind == "scrape":
                from scraper import run_scrape

                asyncio.run(run_scrape())
                if rebuild_after_scrape:
                    from analytics_builder import AnalyticsBuilder

                    AnalyticsBuilder().build_all()
            elif kind == "build":
                from analytics_builder import AnalyticsBuilder

                AnalyticsBuilder().build_all()
            elif kind == "static":
                from static_data import StaticDataCatalog

                StaticDataCatalog.load(refresh=True)
                ChampionCatalog.load(refresh=True)
                self.events.put(("console", "Static data refreshed\n"))
            else:
                raise ValueError(f"Unknown data job: {kind}")
            return_code = 0
        except (RiotAuthenticationError, RiotForbiddenError) as exc:
            return_code = 3
            LOGGER.error("%s", exc)
            self.events.put(("console", str(exc) + "\n"))
        except Exception:
            LOGGER.exception("Packaged data job failed.")
        finally:
            self.data_job_lock.release()
            self.events.put((
                "job_done",
                {
                    "code": return_code,
                    "kind": kind,
                    "background": background,
                    "rebuilt": kind != "scrape" or rebuild_after_scrape,
                },
            ))

    def _project_python(self) -> Path | None:
        """Return the shared project's venv Python when available."""
        candidates = (
            PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
            PROJECT_ROOT / ".venv" / "bin" / "python",
        )
        return next((path for path in candidates if path.is_file()), None)

    def _run_analytics_subprocess_for_watcher(self) -> None:
        """Rebuild v3 analytics using the source venv, including optional CUDA.

        The EXE deliberately excludes PyTorch to remain reasonably small. Since
        it shares the project root, its watcher delegates heavy training to the
        same .venv used by launch_app.bat whenever that interpreter exists.
        """
        python_path = self._project_python()
        script = PROJECT_ROOT / "analytics_builder.py"
        if python_path is None or not script.is_file():
            from analytics_builder import AnalyticsBuilder

            AnalyticsBuilder(DEFAULT_DB_PATH).build_all()
            return
        process = subprocess.Popen(
            [str(python_path), str(script), "--database", str(DEFAULT_DB_PATH)],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **_background_popen_kwargs(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            self.events.put(("console", line))
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"Analytics/model rebuild exited with code {return_code}.")

    def _run_ingest_watcher_thread(self) -> None:
        """Poll the sync inbox and merge collector delta bundles.

        Runs on a daemon thread so the Tk UI never blocks. When new matches are
        ingested and auto-rebuild is enabled, it runs the same analytics/model
        rebuild subprocess the background watcher uses (GPU stays on this PC),
        then signals the UI to reload the engine.
        """
        ingester = SyncIngester()
        try:
            while not self.shutdown_event.is_set():
                profile = load_profile()
                sync = profile.get("sync", {}) or {}
                if not bool(sync.get("enable_pc_ingest", True)):
                    self.events.put((
                        "ingest_status",
                        {"state": "paused", "message": "paused in Settings", "busy": False},
                    ))
                    time.sleep(5)
                    continue
                poll_seconds = max(15, int(sync.get("poll_interval_seconds", 300)))
                self.ingester = ingester
                try:
                    pending = ingester.pending_bundles()
                except OSError as exc:
                    LOGGER.exception("Could not list sync inbox.")
                    self.events.put((
                        "ingest_status",
                        {"state": "error", "message": f"inbox error: {exc}", "busy": False},
                    ))
                    time.sleep(60)
                    continue
                if pending:
                    self.events.put((
                        "ingest_status",
                        {"state": "ingesting", "message": f"found {len(pending)} pending bundle(s)", "busy": True},
                    ))
                    try:
                        result = ingester.ingest_all()
                    except Exception:
                        # One unexpected bundle problem must never kill the
                        # ingest watcher; report and retry on the next poll.
                        LOGGER.exception("Sync ingest run failed unexpectedly.")
                        self.events.put((
                            "ingest_status",
                            {
                                "state": "error",
                                "message": "ingest run failed; will retry next poll",
                                "busy": False,
                            },
                        ))
                        time.sleep(poll_seconds)
                        continue
                    rebuilt = False
                    if result.matches_added > 0 and bool(sync.get("enable_auto_rebuild", True)):
                        self.events.put((
                            "ingest_status",
                            {"state": "rebuilding", "message": "ingested new matches; rebuilding analytics + model", "busy": True},
                        ))
                        try:
                            self._run_analytics_subprocess_for_watcher()
                            rebuilt = True
                        except Exception as exc:
                            LOGGER.exception("Auto-rebuild after ingest failed.")
                            self.events.put((
                                "ingest_status",
                                {"state": "error", "message": f"rebuild failed: {exc}", "busy": False},
                            ))
                    self.events.put((
                        "ingest_status",
                        {
                            "state": "idle",
                            "message": result.detail or "ingest complete",
                            "busy": False,
                            "matches": result.matches_added,
                            "analytics_rebuilt": rebuilt,
                        },
                    ))
                time.sleep(poll_seconds)
        except Exception as exc:
            LOGGER.exception("Sync ingest watcher stopped unexpectedly.")
            self.events.put((
                "ingest_status",
                {"state": "error", "message": f"ingest watcher stopped: {exc}", "busy": False},
            ))

    def _run_background_match_watcher_thread(self) -> None:
        """Run the adaptive unseen-match watcher outside Tkinter's UI thread."""
        try:
            asyncio.run(
                run_background_match_watcher(
                    self.shutdown_event,
                    pause_event=self.champ_select_active,
                    wake_event=self.collector_wake_event,
                    database_path=DEFAULT_DB_PATH,
                    job_lock=self.data_job_lock,
                    status_callback=lambda payload: self.events.put(
                        ("collector_status", payload)
                    ),
                    catalog=self.catalog,
                    analytics_rebuild_callback=self._run_analytics_subprocess_for_watcher,
                )
            )
        except Exception as exc:
            LOGGER.exception("Background match watcher stopped unexpectedly.")
            self.events.put((
                "collector_status",
                {
                    "state": "failed",
                    "message": f"watcher stopped unexpectedly: {exc}",
                    "busy": False,
                },
            ))

    @staticmethod
    def _read_json_version(path: Path) -> str:
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("version", ""))
        except (OSError, ValueError, json.JSONDecodeError):
            return ""

    def _raw_database_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "matches": 0,
            "participants": 0,
            "bans": 0,
            "built_at": "",
            "ml_examples": 0,
            "size_mb": (
                DEFAULT_DB_PATH.stat().st_size / (1024 * 1024)
                if DEFAULT_DB_PATH.exists() else 0.0
            ),
            "patch_counts": {},
        }
        if not DEFAULT_DB_PATH.exists():
            return status
        try:
            with sqlite3.connect(DEFAULT_DB_PATH, timeout=2.0) as connection:
                for table in ("matches", "participants", "bans"):
                    exists = connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,),
                    ).fetchone()
                    if exists:
                        status[table] = int(
                            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                        )
                matches_columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(matches)").fetchall()
                }
                if "game_version" in matches_columns:
                    patch_counts: dict[str, int] = {}
                    for game_version, count in connection.execute(
                        "SELECT game_version, COUNT(*) FROM matches GROUP BY game_version"
                    ):
                        label = patch_label(game_version)
                        patch_counts[label] = patch_counts.get(label, 0) + int(count)
                    status["patch_counts"] = patch_counts
                meta_exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='analytics_meta'"
                ).fetchone()
                if meta_exists:
                    meta = dict(connection.execute("SELECT key,value FROM analytics_meta"))
                    status["built_at"] = str(meta.get("built_at", ""))
                    status["ml_examples"] = int(meta.get("ml_examples", "0") or 0)
        except sqlite3.Error:
            LOGGER.exception("Could not read local database status.")
        return status

    def _open_data_folder(self) -> None:
        folder = PROJECT_ROOT / "data"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except OSError as exc:
            self._append_console(f"Could not open data folder: {exc}\n")

    def _open_sync_inbox(self) -> None:
        folder = resolve_inbox()
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except OSError as exc:
            self._append_console(f"Could not open sync inbox: {exc}\n")

    def _refresh_data_status(self) -> None:
        database = self._raw_database_status()
        matches = int(database["matches"])
        participants = int(database["participants"])
        bans = int(database["bans"])
        built_at = str(database["built_at"])
        ml_examples = int(database["ml_examples"])
        size_mb = float(database["size_mb"])
        patch_counts = dict(database.get("patch_counts", {}))
        key_value = read_api_key()
        key_state = (
            f"{api_key_fingerprint(key_value)} / {self.api_key_validation_state}"
            if key_value else "missing"
        )
        static_version = self._read_json_version(STATIC_CACHE_PATH)
        current_patch = patch_label(static_version) if static_version else "unknown"
        current_patch_games = int(patch_counts.get(current_patch, 0))
        current_patch_share = current_patch_games / matches if matches else 0.0
        analytics_state = "built" if built_at else "not built"
        collector_config = load_profile()["background_collector"]
        collector_state = (
            self.collector_status_text
            if collector_config.get("enabled", False)
            else "server-side (Riot collection on Ubuntu)"
        )
        sync_cfg = load_profile().get("sync", {}) or {}
        ingest_state = (
            self.ingest_status_text
            if bool(sync_cfg.get("enable_pc_ingest", True))
            else "paused in Settings"
        )
        self.data_status_label.configure(
            text=(
                f"Shared runtime root: {PROJECT_ROOT}\n"
                f"Collector: {collector_state}    API key: {key_state}\n"
                f"Server sync ingest: {ingest_state}    "
                f"Sync inbox: {resolve_inbox()}\n"
                f"Downloaded games: {matches:,} total · current patch {current_patch}: "
                f"{current_patch_games:,} ({current_patch_share:.1%})   "
                f"Participants: {participants:,} · bans: {bans:,} · database: {size_mb:.1f} MB\n"
                f"Champion map: {len(self.catalog.records)} cached   "
                f"Static patch: {static_version or 'not cached'}   "
                f"Analytics: {analytics_state}   ML examples: {ml_examples:,}"
            )
        )
        ordered_patches = sorted(
            patch_counts.items(),
            key=lambda item: tuple(int(part) for part in item[0].split("."))
            if item[0] != "unknown" and all(part.isdigit() for part in item[0].split("."))
            else (-1, -1),
            reverse=True,
        )
        history = " · ".join(f"{label}: {count:,}" for label, count in ordered_patches[:10])
        if len(ordered_patches) > 10:
            history += f" · +{len(ordered_patches) - 10} older patches"
        self.patch_status_label.configure(
            text=f"Games by patch: {history or 'no match versions stored yet'}"
        )

    def _append_console(self, text: str) -> None:
        if not text:
            return
        self.console.insert("end", text)
        try:
            line_count = int(str(self.console.index("end-1c")).split(".", 1)[0])
            if line_count > MAX_CONSOLE_LINES:
                delete_through = line_count - MAX_CONSOLE_LINES
                self.console.delete("1.0", f"{delete_through}.0")
        except (ValueError, tk.TclError):
            pass
        self.console.see("end")

    def _apply_ready_image(
        self,
        cache_key: tuple[str, tuple[int, int]],
        pil_image: Image.Image,
        size: tuple[int, int],
    ) -> None:
        image = self.image_cache.get(cache_key)
        if image is None:
            image = ctk.CTkImage(
                light_image=pil_image, dark_image=pil_image, size=size
            )
            self.image_cache[cache_key] = image
            if len(self.image_cache) > 700:
                oldest_key = next(iter(self.image_cache))
                if oldest_key != cache_key:
                    self.image_cache.pop(oldest_key, None)
        waiters = self.image_waiters.pop(cache_key, [])
        self.image_loads_inflight.discard(cache_key)
        for widget in waiters:
            try:
                if widget.winfo_exists():
                    setattr(widget, "_draft_image", image)
                    widget.configure(image=image, text="")
            except (tk.TclError, AttributeError):
                continue

    def _handle_job_done(self, payload: Any) -> None:
        info = payload if isinstance(payload, Mapping) else {"code": int(payload)}
        return_code = int(info.get("code", -1))
        kind = str(info.get("kind", self.current_data_job[0] if self.current_data_job else ""))
        rebuilt = bool(info.get("rebuilt", kind != "scrape"))
        self.current_data_job = None
        for button in self.data_buttons:
            button.configure(state="normal")
        self._refresh_data_status()
        self.job_label.configure(
            text="Complete" if return_code == 0 else f"Failed ({return_code})"
        )
        if return_code == 0 and rebuilt:
            self._reload_engine_async()
        # The automatic watcher may have been waiting on the shared data-job lock.
        self.collector_wake_event.set()

    def _handle_collector_status(self, payload: Mapping[str, Any]) -> None:
        state = str(payload.get("state", "watching"))
        message = str(payload.get("message", state))
        self.collector_status_text = message
        self.collector_busy = bool(payload.get("busy", False))
        if state == "paused":
            # The local watcher is deliberately disabled in Settings; match
            # collection happens on the collector server. Hiding the pill is
            # clearer than a permanently stuck "Paused" notice that reads
            # like a fault. Never clobber an in-flight manual job label.
            if self.current_data_job is not None:
                return
            self.job_label.configure(text="")
            self._refresh_data_status()
            return
        if state == "stopped":
            label = ""  # app shutdown; nothing to show
        else:
            label = state.replace("_", " ").title()
            next_check = payload.get("next_check_seconds")
            if next_check is not None and not self.collector_busy:
                label += f" · next check ~{int(next_check)}s"
        self.job_label.configure(text=f"Watcher: {label}")
        self._refresh_data_status()
        if bool(payload.get("analytics_rebuilt", False)):
            self._reload_engine_async()

    def _handle_ingest_status(self, payload: Mapping[str, Any]) -> None:
        state = str(payload.get("state", "idle"))
        message = str(payload.get("message", state))
        self.ingest_status_text = message
        self.ingest_busy = bool(payload.get("busy", False))
        label = state.replace("_", " ").title()
        if self.ingest_busy:
            self.status_label.configure(text=f"Sync ingest: {label}")
        else:
            self.status_label.configure(text=message)
        self._append_console(message.rstrip() + "\n")
        self._refresh_data_status()
        if bool(payload.get("analytics_rebuilt", False)):
            self._reload_engine_async()

    @staticmethod
    def _should_render_live_picks(
        generation: int,
        newest_generation: int,
        *,
        has_visible_picks: bool,
        session_active: bool,
    ) -> bool:
        """Accept the newest result, or the first usable result while still live."""
        return generation == newest_generation or (session_active and not has_visible_picks)

    def _safe_drain_events(self) -> None:
        """Run one UI event batch and always keep the Tk pump alive.

        Before the v3.0.4 recovery wrapper, one malformed real-world LCU payload
        or rendering exception could terminate the ``after`` callback permanently.
        LCU and analytics
        workers would continue placing results into the queue, but the GUI would
        never consume them—simultaneously causing a frozen live board and an
        infinite “Reloading analytics…” label.
        """
        try:
            self._drain_events()
        except Exception as exc:
            LOGGER.exception("Recovered from an unhandled UI event error.")
            try:
                self.status_label.configure(text=f"Recovered UI event error: {exc}")
            except Exception:
                pass
            try:
                self._append_console(
                    f"Recovered from an internal UI event error: {exc}\n"
                    "The event pump was restarted automatically. See app.log for the traceback.\n"
                )
            except Exception:
                pass
            if not self.shutdown_event.is_set():
                self.after(40, self._safe_drain_events)

    def _drain_events(self) -> None:
        console_chunks: list[str] = []
        processed = 0
        while processed < MAX_UI_EVENTS_PER_TICK:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            processed += 1
            if kind == "console":
                console_chunks.append(str(payload))
                continue
            if kind == "lcu_status":
                self.status_label.configure(text=str(payload))
            elif kind == "lcu_session":
                self._handle_lcu_session(payload)
            elif kind == "analysis":
                generation, picks, bans, active_role, target, insights = payload
                if generation == self.analysis_generations.get(target, -1):
                    self._render_manual_analysis(picks, bans, active_role, insights)
                    self._render_insights(insights)
            elif kind == "live_analysis_picks":
                generation, picks, active_role, insights = payload
                # Match v2's proven behaviour: only the latest generation may
                # update the live cards. Every meaningful draft change submits a
                # fresh evaluation, so the newest result cannot be stranded behind
                # a broken single-in-flight state flag.
                if (
                    generation == self.analysis_generations["live"]
                    and self.champ_select_active.is_set()
                ):
                    self.last_live_render_generation = generation
                    self._render_analysis(picks, None, active_role)
                    self._render_insights(insights)
                    pick_count = sum(len(values) for values in picks.values())
                    if pick_count:
                        self.status_label.configure(
                            text=f"Live recommendations updated · {pick_count} shown"
                        )
                    else:
                        self.status_label.configure(
                            text="No eligible live recommendations · reload/rebuild analytics"
                        )
                        console_chunks.append(
                            "Live evaluation returned no eligible candidates. "
                            "Use Live Draft > Reload analytics or Model & Features > Rebuild analytics + model.\n"
                        )
            elif kind in {"live_bans_ready", "live_analysis_complete"}:
                # ``live_analysis_complete`` remains accepted for compatibility
                # with any already-running pre-hotfix worker during an in-place
                # source upgrade. New work arrives as ``live_bans_ready``.
                if kind == "live_bans_ready":
                    generation, bans, active_role, ban_state_key, elapsed_ms = payload
                else:
                    generation, bans, active_role, ban_state_key = payload
                    elapsed_ms = 0.0
                current_ban_key = (
                    self.snapshot.ban_key
                    if self.snapshot is not EMPTY_SNAPSHOT
                    else None
                )
                if (
                    self.champ_select_active.is_set()
                    and not self.ban_phase_over
                    and generation == self.ban_generation
                    and ban_state_key == current_ban_key
                ):
                    self._finish_ban_watchdog(generation)
                    self._render_analysis(None, bans, active_role)
                    ban_count = sum(len(values) for values in bans.values())
                    LOGGER.info(
                        "Rendered %d live ban recommendations in %.1fms.",
                        ban_count, elapsed_ms,
                    )
                else:
                    LOGGER.debug(
                        "Discarded obsolete ban result generation=%s current=%s key_match=%s.",
                        generation, self.ban_generation, ban_state_key == current_ban_key,
                    )
            elif kind == "live_bans_failed":
                generation, ban_state_key, message = payload
                current_ban_key = (
                    self.snapshot.ban_key
                    if self.snapshot is not EMPTY_SNAPSHOT
                    else None
                )
                if (
                    self.champ_select_active.is_set()
                    and not self.ban_phase_over
                    and generation == self.ban_generation
                    and ban_state_key == current_ban_key
                ):
                    self._finish_ban_watchdog(generation)
                    for role in ROLES:
                        frame = self.role_ban_frames[role]
                        self._clear_frame(frame)
                        ctk.CTkLabel(
                            frame,
                            text="Ban recommendations unavailable · see Data Watcher",
                            text_color="gray60",
                            wraplength=230,
                            justify="left",
                        ).pack(anchor="w", padx=5, pady=2)
                    console_chunks.append(str(message) + "\n")
            elif kind == "analysis_failed":
                target, generation, message = payload
                if generation == self.analysis_generations.get(target, -1):
                    self.status_label.configure(text=str(message))
                    console_chunks.append(str(message) + "\n")
            elif kind == "image_ready":
                cache_key, pil_image, size = payload
                self._apply_ready_image(cache_key, pil_image, size)
            elif kind == "image_failed":
                self.image_loads_inflight.discard(payload)
                self.image_waiters.pop(payload, None)
            elif kind == "api_key_valid":
                platform = str(payload.get("platform", ""))
                name = str(payload.get("name", platform))
                self.api_key_validation_state = "valid"
                self._update_api_key_status()
                self.profile_message.configure(
                    text=f"Riot API key accepted for {platform} ({name})."
                )
                console_chunks.append(
                    f"Riot API key accepted for {platform} ({name}).\n"
                )
                self._refresh_data_status()
                self.collector_wake_event.set()
            elif kind == "api_key_invalid":
                self.api_key_validation_state = "invalid"
                self._update_api_key_status()
                self.profile_message.configure(text=str(payload))
                console_chunks.append(str(payload) + "\n")
                self._refresh_data_status()
            elif kind == "api_key_error":
                self.api_key_validation_state = "validation error"
                self._update_api_key_status()
                message = f"Could not validate key: {payload}"
                self.profile_message.configure(text=message)
                console_chunks.append(message + "\n")
                self._refresh_data_status()
            elif kind == "collector_status":
                if isinstance(payload, Mapping):
                    self._handle_collector_status(payload)
            elif kind == "ingest_status":
                if isinstance(payload, Mapping):
                    self._handle_ingest_status(payload)
            elif kind == "job_done":
                self._handle_job_done(payload)
            elif kind == "gpu_status":
                if isinstance(payload, Mapping) and payload.get("available"):
                    message = (
                        f"CUDA available · {payload.get('device')} · PyTorch {payload.get('torch')} · "
                        f"CUDA runtime {payload.get('cuda')} · {payload.get('count')} device(s)\n"
                        f"Probe Python: {payload.get('python', self._project_python() or sys.executable)}"
                    )
                else:
                    detail = (
                        payload.get("error", "PyTorch installed, but CUDA is unavailable")
                        if isinstance(payload, Mapping) else str(payload)
                    )
                    probe_python = payload.get("python", self._project_python() or sys.executable) if isinstance(payload, Mapping) else (self._project_python() or sys.executable)
                    message = (
                        f"GPU training unavailable: {detail}\n"
                        f"Probe Python: {probe_python}\n"
                        "Run setup_gpu_ml.bat, then click Check GPU again."
                    )
                self.model_status_label.configure(text=message)
                console_chunks.append(message + "\n")
            elif kind == "engine_reloaded":
                if getattr(self, "engine_reload_watchdog_after", None):
                    try:
                        self.after_cancel(self.engine_reload_watchdog_after)
                    except Exception:
                        pass
                    self.engine_reload_watchdog_after = None
                # The worker built a complete replacement without touching the
                # engine currently used by live evaluations. Swapping the object
                # reference is atomic and cannot deadlock the Tk event loop.
                if isinstance(payload, DraftEngine):
                    self.engine = payload
                self.engine_reload_running = False
                self.profile = load_profile()
                self._refresh_pool_header_labels()
                self._update_continuous_controls()
                self.status_label.configure(text="Analytics loaded")
                self._refresh_model_status()
                self._refresh_data_status()
                if self.snapshot is EMPTY_SNAPSHOT:
                    self._show_empty_state()
                else:
                    self.last_draft_key = None
                    self.last_submitted_ban_key = None
                    self._evaluate_current_snapshot()
                if self.engine_reload_pending:
                    self.engine_reload_pending = False
                    self.after_idle(self._reload_engine_async)
            elif kind == "engine_reload_failed":
                if getattr(self, "engine_reload_watchdog_after", None):
                    try:
                        self.after_cancel(self.engine_reload_watchdog_after)
                    except Exception:
                        pass
                    self.engine_reload_watchdog_after = None
                self.engine_reload_running = False
                self.status_label.configure(text=str(payload))
                console_chunks.append(str(payload) + "\n")
                if self.engine_reload_pending:
                    self.engine_reload_pending = False
                    self.after_idle(self._reload_engine_async)
            elif kind == "error":
                self.status_label.configure(text=str(payload))
                console_chunks.append(str(payload) + "\n")

        if console_chunks:
            self._append_console("".join(console_chunks))
        delay = 1 if not self.events.empty() else int(
            self.profile["ui"].get("poll_interval_ms", 40)
        )
        if not self.shutdown_event.is_set():
            self.after(delay, self._safe_drain_events)

    def _on_close(self) -> None:
        self.shutdown_event.set()
        self.collector_wake_event.set()
        try:
            logging.getLogger().removeHandler(self._console_log_handler)
        except Exception:
            pass
        if self.ban_watchdog_after:
            try:
                self.after_cancel(self.ban_watchdog_after)
            except Exception:
                pass
            self.ban_watchdog_after = None
        for executor in (
            getattr(self, "analysis_worker", None),
            getattr(self, "ban_worker", None),
            getattr(self, "reload_worker", None),
            self.worker,
        ):
            if executor is None:
                continue
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
        self.destroy()


def _packaged_self_test() -> int:
    """Minimal post-build test used by build_exe.bat.

    Importing this module has already proven that CustomTkinter is bundled. The
    remaining checks verify that core packaged modules can be located without
    opening the GUI. A marker file is written beside the executable for easy
    diagnosis on Windows.
    """
    marker = EXECUTABLE_DIR / "build_self_test.txt"
    try:
        import customtkinter  # noqa: F401
        import lcu_driver  # noqa: F401
        import numpy  # noqa: F401
        import ml_runtime  # noqa: F401
        import patch_utils  # noqa: F401
        import role_inference  # noqa: F401
        import ingest  # noqa: F401

        ChampionCatalog.load(allow_download=False)
        load_profile()
        shared_marker = EXECUTABLE_DIR / "shared_project_root.txt"
        if shared_marker.exists() and PROJECT_ROOT == EXECUTABLE_DIR:
            raise RuntimeError("shared_project_root.txt exists but the EXE did not resolve the shared project root")
        marker.write_text(
            f"League Draft Lab v{APP_VERSION} packaged self-test passed.\n"
            f"Executable directory: {EXECUTABLE_DIR}\n"
            f"Shared project root: {PROJECT_ROOT}\n"
            f"API key file: {ENV_PATH}\n",
            encoding="utf-8",
        )
        return 0
    except Exception as exc:
        marker.write_text(f"League Draft Lab v{APP_VERSION} packaged self-test failed: {exc!r}\n", encoding="utf-8")
        return 1


def main() -> None:
    if "--self-test" in sys.argv:
        raise SystemExit(_packaged_self_test())
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [
        logging.FileHandler(PROJECT_ROOT / "app.log", encoding="utf-8")
    ]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers)
    app = DraftApp()
    app.mainloop()


if __name__ == "__main__":
    main()
