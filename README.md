# League Draft Lab v3.2

A League of Legends draft assistant: live LCU drafting, analytics, and ML
training on your PC, powered by a steady stream of ranked matches collected
from the Riot API.

Two apps, one repo:

| App | Platform | Role |
|-----|----------|------|
| `draft_app/` | **Windows** (main PC with LoL + GPU) | Live LCU drafting, analytics, ML training, and **ingesting** collected data |
| `collector_app/` | **Ubuntu server** *or* the **same Windows PC** | Headless Riot data collection only; no analytics, no ML, no GPU |

Pick the deployment that matches your hardware:

- **Single device (Windows only)** — the collector runs hidden in the
  background on your gaming PC and writes bundles straight into the draft
  app's inbox. Start here: [`collector_app/README.md`](collector_app/README.md)
  → *"Setup — Windows"*.
- **Two devices (Ubuntu server + Windows PC)** — a home server collects 24/7
  and pushes bundles to your PC. Start here: [`collector_app/README.md`](collector_app/README.md)
  → *"Setup — Ubuntu server"*.

The v3 engine, UI, on-database schema, and model are kept near-identical; the
apps only differ in who collects and who crunches.

## How data flows

```
collector_app (Ubuntu server or Windows background task)
  Riot API -> data/draft_data.sqlite3
           -> *.sync.zip   (delta bundles: only new matches + static snapshots)
                       │  single device: written directly into the inbox
                       │  two devices:  rsync/scp over tailscale
                       ▼
draft_app (Windows PC)
  sync_inbox/*.sync.zip
        └─ auto-ingest (poll every ~5 min by default)
             ├─ merge into data/draft_data.sqlite3
             ├─ apply static/champion snapshots (keeps current patch)
             ├─ move bundle to sync_inbox_processed/
             └─ rebuild analytics + train neural model (GPU) on the PC
```

- The **collector only collects** — it never trains or computes analytics,
  saving CPU/battery wherever it runs.
- The **PC does the heavy work** (analytics + CUDA training) whenever new data
  lands.
- The PC still shows number of games, current-patch % and patch distribution.

## Quick start — single device (all on one Windows PC)

```bat
cd draft_app
setup_windows.bat                 REM draft app: venv + deps
setup_gpu_ml.bat                  REM optional: CUDA PyTorch for neural training
launch_app.bat                    REM run the draft app from source

cd ..\collector_app
setup_windows.bat                 REM collector: venv + key + inbox link + auto-start
```

`setup_windows.bat` in `collector_app` asks two questions along the way —
answer **Y** to link the collector's output into `draft_app\sync_inbox` and
**Y** to install the auto-start task (runs hidden at every sign-in, restarts
on failure). See [`collector_app/README.md`](collector_app/README.md) for the
full walkthrough, including manual alternatives.

## Quick start — two devices (Ubuntu server + Windows PC)

### 1. Ubuntu server — collector

```bash
cd collector_app
./setup_ubuntu.sh                 # venv + deps + static data + systemd service
echo 'RIOT_API_KEY=RGAPI-...' > config.env
systemctl --user restart lol-draft-collector
```

### 2. Windows PC — draft app

```bat
cd draft_app
setup_windows.bat                 REM venv + deps
setup_gpu_ml.bat                  REM optional: CUDA PyTorch for neural training
launch_app.bat                    REM run from source
```

To build the EXE (bundles the `lol_draft_icon_option_2.ico` icon):

```bat
build_exe.bat
```

Output: `draft_app\dist\LeagueDraftLab\LeagueDraftLab.exe`

See [`draft_app/README.md`](draft_app/README.md).

### 3. Connect the two

1. Set the PC's sync inbox (default `draft_app/sync_inbox/`) either as the
   tailscale destination or point the server `sync.outbox_dir` at a shared mount.
2. Keep bundles flowing into `sync_inbox/` (cron/rsync on the server, or the
   collector's `--export` + your transfer script).
3. Open the draft app — it auto-polls, ingests, and rebuilds. "Open sync inbox"
   is available on the **Data Watcher** tab, and "Auto-ingest server data +
   rebuild" on **Settings** toggles the behaviour.

## Security & your Riot API key

- Your key lives in **`config.env`** next to each app's code. Both
  `.gitignore` files exclude it (and any `*.env` variant), so it cannot be
  committed by accident. `config.env.example` is the tracked template.
- Never paste a real key into issues, screenshots, or `config.env.example`.
- Development keys expire (~24h); the apps prompt/retry until a fresh key is
  saved. Get one at the [Riot Developer Portal](https://developer.riotgames.com/).
- Everything else in `data/`, `outbox/`, `sync_inbox*/` (databases, bundles,
  model files, logs) is generated at runtime and likewise git-ignored.
- Before your first push, a quick audit helps: `git grep -n "RGAPI-"` should
  only ever match the placeholder in `config.env.example`.

## Model & data notes

- The PC's `config_profile.json` ships with the Riot **background watcher
  disabled** (`background_collector.enabled=false`) because collection lives
  in the collector app. Re-enable it on Settings only if you want the draft
  app itself to scrape Riot directly — and then pause the standalone
  collector, or the two will compete for the same key's rate limits.
- The PC's `sync.enable_pc_ingest=true` by default, so opening the app is enough
  to notice and process newly transferred data.
- Static data / current patch auto-refresh happens in the collector (never
  touched) and its snapshots ride inside every bundle, so the PC stays current.

## License

Released under the [MIT License](LICENSE).
