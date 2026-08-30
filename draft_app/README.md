# League Draft Lab v3.2 — Windows draft app

The Windows-side application for the main PC (League + GPU). It keeps the v3
live LCU drafting, analytics, ML training, pools, and Data Watcher tabs, and
now consumes data pushed from the Ubuntu collector.

## v3.2 changes

- Champion pool editor opens near-instantly: portraits decode from the local disk cache on open, a launch-time prefetch warms missing icons, and icons are shared across widget sizes.
- Fixed duplicated champion entries/icons caused by CommunityDragon's "Jade" variant rows (ids 60001+); stale portrait files are pruned automatically.
- Comfort/Pocket pickers are now chosen from that role's General pool (with a "fill the General pool first" prompt), and picking one champion into both lists warns that Comfort takes precedence.
- Data Watcher tab cleaned up: legacy local-scrape/watcher buttons removed now that collection lives on the collector server.
- Collector: catalog dedupe ported server-side, and an optional `scraper.matches_window_days` recency cutoff (`startTime`) was added for match discovery.

## v3.1.1 changes

- Ban advice now only shows during the live ban phase and is cleared once bans complete (no more lingering or rapid churn through picks).
- Champion Pools gained a clickable, alphabetically-sorted champion-icon picker per role.

## What changed vs v3

- **No Riot scraping by default.** Data collection moved to the Ubuntu server.
  The PC's `background_collector.enabled` defaults to `false` (you can re-enable
  it on Settings for a PC that also scrapes Riot directly).
- **Auto server-data ingest.** The app polls `sync_inbox/` (default
  `draft_app/sync_inbox/`) and, when new `*.sync.zip` bundles appear, it:
  1. merges new matches into `data/draft_data.sqlite3`;
  2. applies the current-patch static/champion snapshots carried in the bundle;
  3. moves the bundle to `sync_inbox_processed/`;
  4. rebuilds analytics + retrains the neural model (CUDA) and reloads the engine.
- New tabs/controls: **Settings → "Auto-ingest server data + rebuild"** toggle
  (default on), **Data Watcher → "Open sync inbox"** button.
- Icon: `lol_draft_icon_option_2.ico` is bundled into the EXE.

## Setup (Windows)

```bat
setup_windows.bat     REM venv + deps + champion/static data
setup_gpu_ml.bat      REM optional: CUDA PyTorch for neural training
launch_app.bat        REM run from source
```

`config.env.example` documents the (now optional) local Riot key. The PC uses
the key only if you re-enable local scraping.

## Build the EXE (with icon)

```bat
build_exe.bat
```

Output:
```text
draft_app\dist\LeagueDraftLab\LeagueDraftLab.exe
```

`build_exe.bat` packages `lol_draft_icon_option_2.ico` as the window/taskbar
icon via the PyInstaller spec (`LeagueDraftLab.spec`). As in v3, the EXE shares
the project's `config_profile.json`, `config.env`, `data`, and `.venv`, and does
not bundle PyTorch (it delegates GPU training to the project `.venv`).

## Sync inbox

- Default inbox: `draft_app/sync_inbox/`
- Processed bundles: `draft_app/sync_inbox_processed/`
- Poll interval: `sync.poll_interval_seconds` (default 300 s) in
  `config_profile.json`.
- If you point the collector's `sync.outbox_dir` at a tailscale-shared folder,
  set the PC `sync.inbox_dir` to that same path.
- **Running everything on one Windows PC?** The collector can write its
  bundles straight into this inbox (`collector_app\single_device_sync.py
  --link`, or answer Y during `setup_windows.bat`). See
  [`collector_app/README.md`](../collector_app/README.md) → *"Setup — Windows"*.
- Do not run the collector and this app's built-in background watcher
  (Settings) at the same time — they share one Riot key and its rate limits.

## Data Watcher

Shows total downloaded games, current-patch count + share, patch distribution,
database size, ML examples, plus a **Server sync ingest** status, the current
sync inbox path, and build/training state.

## Manual rebuilds

- **Model & Features → Rebuild analytics + model** forces a full analytics +
  neural rebuild and engine reload.
- **Data Watcher → Rebuild analytics only** rebuilds without retraining.
- **Refresh static data** redownloads champion/static data directly (normally
  snapshots arrive via ingest).

## Tabs (unchanged from v3)

Live Draft, Manual Lab, Draft Insights, Champion Pools, Model & Features,
Data Watcher, Settings.
