# Draft Lab v4

A draft assistant that compares **draft lines**, not just champions. It searches legal continuations,
plays the other team as an uncertain opponent, and returns a short list of defensible options, each with
its outlook, worst-branch outlook, likely reply, credible punish, pivot, and game-plan fit.

V3 (`../draft_app`) stays untouched as the baseline. V4 only **reads** the collector's SQLite database.

## Run it

```bash
./setup.sh            # Linux: .venv + CUDA torch (cu128) + deps     (Windows: setup_windows.bat)
./run.sh              # http://127.0.0.1:8765                        (Windows: run_windows.bat)
./run.sh --host 0.0.0.0   # share with coaches on a trusted LAN (no login: do not expose to the internet)
.venv/bin/python -m pytest  # 15 tests: legality fuzzing, determinism, latency, training pipeline
```

With no trained data, the tool runs in **seed mode**: roles and traits come from `data/champion_seed.txt`,
win estimates are flat (about 50%), and the UI says so. Rankings then rest on roles, plan fit and comfort.

## Modes

- **Manual** (scrims, testing, coaching): click champions to fill the lit slot in draft order, or click any
  slot/tick to edit it out of order. Lock roles per pick, set "No ban", undo (Ctrl+Z), swap sides, export or
  import a draft as JSON, keep player pools, and choose whose side to advise. **Fearless** has a
  "Next game" button that locks out every champion picked in earlier games of the series.
- **Live client**: polls the League client (LCU lockfile; set `LOL_LOCKFILE` if it isn't in the default
  location) and re-analyses after every **lock**. Hovers are ignored. Live mode is read-only and is not on
  the first release's critical path. Check Riot's policies before distributing it.

On your own turn you get **recommendations**. On the opponent's turn you get a **forecast** of their
likely actions and what each does to your outlook. Once the draft is complete you get a **final read**.

## How it maps to the proposal

| Proposal component | Where | Status |
|---|---|---|
| Rules, state, legality (pro, ranked, fearless) | `draft4/rules.py` | Done. Illegal actions are masked regardless of model scores |
| **Draft Policy** (behaviour model) + **Engram** | `draft4/nets.py: PolicyNet, EngramMemory` · `train policy` | Training-ready. Engram is a gated hashed-pair memory behind a swap-in interface |
| Smoothed prior policy (baseline, fallback) | `draft4/models.py: PriorPolicy` | Done |
| **Composition Value Model** | `draft4/nets.py: ValueNet` (antisymmetric; `none`/`fm`/`mlp` interactions) · `train value` | Training-ready |
| Additive value baseline | `draft4/models.py: AdditiveValue` (pair weight fitted on calibration) | Done |
| Game-plan scorer (coach-editable) | `draft4/gameplan.py` (6 plans: requirements + failure modes) | Done. Reported apart from win %, never added to it |
| Draft Planner (decision-maker) | `draft4/planner.py` | Done. Paired rollouts, successive halving, punish/pivot search, ban deltas vs. a typical ban |
| Game-Plan Fit Model (third model) | — | Deferred, as the proposal says |
| LLM | — | Not used. Explanations are generated from evaluated branches |

The planner's decision rule follows §4. It filters illegal options and hard constraints first, then
ranks by average outcome. Options within paired-rollout noise (or within 1 pt) are ordered by plan fit,
then by downside. Advice is deterministic for identical state and model versions. Every result carries
the state fingerprint and model versions, so the UI drops stale responses.

## Training (when data is ready)

GPU: verified on an RTX 5060 Ti with torch 2.11 + cu128. Training scripts use CUDA automatically
(`--device cpu` to force CPU). Inference runs on CPU, which the proposal requires for release.

```bash
# 1. Freeze a snapshot (read-only on the DB). Ranked solo queue = 420.
.venv/bin/python -m train.train snapshot --db ../collector_app/data/draft_data.sqlite3
#    -> data/snapshots/snap-*.npz + manifest (matches per patch, ban coverage) and latest.npz

# 2. Baseline priors (fit on the train window; pair weight picked on the calibration window)
.venv/bin/python -m train.train priors
# 3. Composition value: compares to side-only, additive priors, and a main-effects net
.venv/bin/python -m train.train value --interactions fm
# 4. Draft policy with the Engram ablation (same splits, same masks)
.venv/bin/python -m train.train policy --memory both
# 5. Promote only what beats its baseline on the untouched test window (old artifacts kept for rollback)
.venv/bin/python -m train.train promote
# 6. Production priors from every match, after evaluating:
.venv/bin/python -m train.train priors --all && .venv/bin/python -m train.train promote
```

Then restart the server (or `POST /api/reload`). Candidates and metrics live in `models/candidates/`,
promoted ones in `models/active/`. Missing or broken checkpoints fall back to the baselines with a warning.

**Leakage rules built into the pipeline:** splits are chronological and by match (80/10/10) before any
masking. Priors used as baselines are fit on train only. Temperature and pair weights are tuned on the
calibration window. Test is touched once per run. The policy is trained on *eventual lineup appearances*
(ranked has no pick order), with final bans shown only as post-ban context.

**Try this first with your ~200k games:** the defaults (`d=64`, 2 Engram heads × 4096 rows) total
about 70k parameters without Engram and 600k with it. Compare `policy_none` with `policy_engram` in
`policy.metrics.json`. Keep Engram only if it wins on log loss and top-k for the slices you care about.
The Engram research session can replace `EngramMemory` (keys, n-gram order, placement) without touching
the engine.

**Smoke-test without real data:** `python -m train.synthetic --out data/synthetic.sqlite3` writes a fake
collector-schema DB with planted structure. Never promote models trained on it.

## Pro data (Oracle's Elixir)

Put the yearly CSVs from [Oracle's Elixir](https://oracleselixir.com/tools/downloads) in `data/pro/`, then:

```bash
.venv/bin/python -m train.pro_import --years 2024 2025 2026   # ordered drafts + pro role rates
.venv/bin/python -m train.train pro-policy                     # ordered next pick/ban, warm-started from the ranked policy
.venv/bin/python -m train.train promote
```

The importer rebuilds each draft from OE's per-team `ban1-5` / `pick1-5` order, `firstPick` and side.
Since 2026, first pick is no longer tied to blue side, and the UI has a **First pick** selector for pro
formats. Games that can't be rebuilt exactly are dropped and counted. Splits are by **series**, in time
order. Pro formats use the pro policy for opponent forecasts and pro role rates for role assignment.
Candidates from the solo-queue win-rate shortlist must appear in at least 0.5% of pro games (player-pool
picks are exempt).

## Results so far (19 Nov 2026)

Ranked snapshot: 161,893 Emerald+ solo-queue matches (patches 16.x, mostly 16.13–16.22).
Pro: 29,164 ordered drafts from 2024–2026.
Test sets are the most recent 10% of matches or series and were never used for training or tuning.

| Model | Test result | Baseline | Promoted? |
|---|---|---|---|
| Ranked value, additive priors | log loss 0.6906, accuracy 53.3% | side only: 0.6932 / 49.8% | yes (in use) |
| Ranked value, neural (none / fm / mlp) | 0.6906 / 0.6914 / 0.6913 | additive 0.6906 | no: no real gain |
| Ranked policy (no Engram) | log loss 3.827, top-1 7.9%, top-5 27.9% | prior 3.857 / 6.7% / 26.4% | yes |
| Ranked policy + Engram | 3.829 / 8.0% / 27.9% | — | no: no gain, 9× params |
| Pro policy, warm-started | log loss 3.951; picks top-1 17.3%, top-5 51.2%; bans top-5 17.6% | recent frequency 4.097; picks 9.7% / 42.3%; bans 11.5% | yes |
| Pro policy + Engram (2 seeds) | −0.017 and +0.005 vs no Engram | seed noise ≈ 0.03 | no: within noise |

What this means: in solo queue, the draft explains only a small part of who wins, so treat win
estimates as tie-breakers, not odds. The policies predict what people pick much better than popularity
does, which is what makes the opponent simulation credible. Engram stays plugged in, ready for the
research session, but it isn't earning its place yet.

## Data and maintenance

- `data/ddragon_champion.json`: champion list and portraits. Refresh with `python -m draft4.catalog`,
  which also lists new champions missing from the seed.
- `data/champion_seed.txt`: **coach-editable** roles and traits (engage, peel, poke, …). The plan scorer
  reads the traits, and the priors replace the roles once data exists.
- Plan requirements and failure modes are plain Python in `draft4/gameplan.py`, short enough for a coach
  to review line by line.

## Not done yet (deliberately)

- Pro-specific win estimates: pro outcomes (~3k games/year per major region) are too thin for the value model, so they stay ranked-based and labelled.
- Fearless locks from earlier games are not yet masked when training the pro policy (series aren't tagged as fearless in OE).
- Coach-reviewed benchmark positions (§6) and the Game-Plan Fit Model (§3.3).
- Ban **sets** are evaluated marginally (our later bans are sampled), not jointly.
- Series-level fearless strategy. Fearless legality is done.
