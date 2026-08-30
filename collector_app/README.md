# League Draft Lab v3.1 — data collector (Ubuntu server or Windows PC)

Headless collector. **Collection only** — no analytics, no GPU, no ML
training. It stores raw ranked matches, auto-refreshes static data, and writes
incremental `*.sync.zip` bundles that the Windows draft app ingests.

Run it on either platform:

| Platform | Use case | Auto-start mechanism |
|----------|----------|----------------------|
| **Ubuntu** home server | Separate always-on box next to your gaming PC | systemd user service |
| **Windows** (same PC as the draft app) | No second machine — everything on one device | Task Scheduler task at sign-in |

## Layout

```text
collector_app/
  collector_daemon.py              long-lived watcher + static auto-refresh
  sync_export.py                   incremental delta bundle exporter
  scraper.py                       Riot Match-V5 collector (shared with the PC)
  data/                            draft_data.sqlite3, champion_map.json, static_data.json
  outbox/                          *.sync.zip bundles awaiting transfer
  systemd/lol-draft-collector.service
  single_device_sync.py            point the outbox at a local draft app inbox
  collector_autostart.ps1          Task Scheduler install/remove/start/stop/status
  setup_windows.bat                one-click Windows setup
  start_/status_/stop_...bat       manual Windows launchers
  install_/remove_autostart...bat  Task Scheduler wrappers
```

## Riot API key

- The collector reads its key from `config.env` next to the code
  (`RIOT_API_KEY=RGAPI-...`). That file is git-ignored — never commit or share it.
- Development keys expire (~24h); replace the key in `config.env` and restart
  the collector (`systemctl --user restart lol-draft-collector` on Ubuntu, or
  `stop_collector_windows.bat` → `start_collector_windows.bat` on Windows).
- `--status` and normal collection require a valid key.

---

## Setup — Ubuntu server

```bash
cd collector_app
./setup_ubuntu.sh                 # venv + deps + static data + systemd service
echo 'RIOT_API_KEY=RGAPI-...' > config.env
systemctl --user restart lol-draft-collector
```

`setup_ubuntu.sh`:

- creates `.venv` and installs `requirements.txt` (only `requests` + `python-dotenv`);
- refreshes champion map + static data (current patch);
- generates `config_profile.json` (calls `load_profile`);
- installs + starts a systemd user service `lol-draft-collector`.

### systemd + the 06:00–02:00 power window

The server powers on at 06:00 AEST and off at 02:00. A systemd **user** service
is started at login/boot:

```bash
systemctl --user enable --now lol-draft-collector
loginctl enable-linger youruser   # keep the user service alive without login
```

The unit is at `systemd/lol-draft-collector.service`. The setup script injects
the real absolute path. Tune `RestartSec` and `Environment` as needed.

---

## Setup — Windows (single device, collector + draft app on one PC)

Requires Python 3.11+ from [python.org](https://www.python.org/downloads/)
(tick "Add python.exe to PATH" — the scripts use the `py` launcher).

1. **Set up** — double-click `setup_windows.bat` inside `collector_app/`. It:
   - creates `.venv` and installs `requirements.txt`;
   - refreshes champion + static data for the current patch;
   - creates `config.env` (opens it in Notepad — paste your Riot key) and
     `config_profile.json`;
   - asks whether to **link** the collector's output straight into the draft
     app's `draft_app/sync_inbox/` (answer **Y** for the single-device setup);
   - asks whether to install the **auto-start** task (answer **Y**).
2. **Auto-collect on startup** — if you answered Y, the collector now starts
   hidden at every Windows sign-in via Task Scheduler (task name
   `LeagueDraftLabCollector`, runs `pythonw.exe collector_daemon.py`, restarts
   on failure, no execution time limit). You can manage it later:
   - `install_autostart_windows.bat` — install/replace the task and start it now
   - `remove_autostart_windows.bat` — remove the task and stop the daemon
   - `stop_collector_windows.bat` — stop without uninstalling
   - `status_collector_windows.bat` — DB counts + task/daemon/log status

   *Prefer a manual method instead?* Create a shortcut to
   `.venv\Scripts\pythonw.exe` with argument `collector_daemon.py` and
   "Start in" set to the `collector_app` folder, then drop the shortcut into
   `Win+R` → `shell:startup`.
3. **Point it at the draft app** (if you skipped the link in setup):
   ```bat
   .venv\Scripts\python.exe single_device_sync.py --link
   ```
   This records `draft_app\sync_inbox` as the collector's `sync.outbox_dir`,
   so every `*.sync.zip` lands directly where the draft app ingests it — no
   copying between machines. `--show` prints the current target, `--unlink`
   reverts to the local `outbox/` folder.
4. **Run manually** — `start_collector_windows.bat` opens a visible console
   (handy while tuning `config_profile.json`).

> **One collector at a time.** The draft app also has a built-in background
> watcher (Settings → *background collector*; off by default). Run *either*
> the standalone collector or the draft app's watcher — never both, or they
> compete for the same Riot key's rate limits.

### Where things live on Windows

| What | Where |
|------|-------|
| Database | `collector_app\data\draft_data.sqlite3` |
| Log file | `collector_app\scraper.log` |
| Bundles | `draft_app\sync_inbox\*.sync.zip` (linked) or `collector_app\outbox\` |
| Scheduled task | Task Scheduler → `LeagueDraftLabCollector` |

---

## Running modes

```text
python collector_daemon.py              # long-lived watcher (service/task)
python collector_daemon.py --status     # DB counts + outbox summary
python collector_daemon.py --once       # single scrape batch, then exit
python collector_daemon.py --export     # export-only (write a new bundle)
python collector_daemon.py --refresh-static --verbose   # force static refresh
```

## Auto static-data refresh

The "rarely checked" nature is handled: `_static_refresh_loop` runs on a
daemon thread and force-refreshes the champion map + Data Dragon static data
every 6 hours, and once at startup. This keeps the current patch current even
though nobody visits the server.

## How many matches per day to expect

Collection yield is bounded by the size of the tracked ladder sample, **not**
by the request budget (the limiter allows ~45 requests/minute; discovery needs
far fewer). Defaults track `players_per_run` (80) Emerald players across
divisions I–IV, re-polled in slices of 20 every 45–300 s, listing each player's
15 most recent ranked games on every visit.

That means a realistic ceiling of roughly **250–600 *new* ranked games per day**
for an 80-player roster — about each tracked player's actual play rate. To
scale up, raise these keys in `config_profile.json`:

| Key | Default | Effect |
|-----|---------|--------|
| `scraper.players_per_run` | 80 | Ladder-roster size; the main volume lever. 300–500 quadruples daily yield. |
| `background_collector.players_per_poll` | 20 | How many of those players each cycle visits (keep ≤ 25 for dev-key windows). |
| `scraper.matches_window_days` | 0 (off) | When set (e.g. 30), Match-V5 listings only return games from the last N days (`startTime`), so stale history stops being re-scanned after roster churn. |

Old-patch games are intentionally collected too (the analytics use patch
weighting with a 3-day half-life), so a previous-patch backfill appears once
when a patch rotates or when players join the roster — that is one-time cost,
not a leak. Repeated skipping happens only for remakes / role-incomplete games,
which are too few to matter.

## Transferring data to the PC (two-machine setup only)

On a single device with the link enabled you can skip this section entirely.

The collector writes bundles to `outbox/`. You push them to the Windows PC's
`draft_app/sync_inbox/`. The PC app polls `sync.enable_pc_ingest` on Settings
and auto-processes them. Example server cron (editor's choice; adapt to your
setup):

```bash
# Every 10 minutes, push any new bundles over tailscale to the PC inbox.
*/10 * * * * rsync -a --ignore-existing \
    ~/documents/projects/lol_draft_tool/lol_draft_tool_v3.1/collector_app/outbox/*.sync.zip \
    youruser@your-pc-hostname:~/documents/projects/lol_draft_tool/lol_draft_tool_v3.1/draft_app/sync_inbox/
```

## Why no analytics on the collector

GPU compute stays on the main PC. `run_background_match_watcher` is called
with a no-op `analytics_rebuild_callback`, so the shared `scraper.py` never
imports `analytics_builder` / `numpy` / `pandas` in the collector. The
collector stays tiny and CPU-cheap on both platforms.
