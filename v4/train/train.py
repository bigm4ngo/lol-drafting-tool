"""V4 training pipeline. Every step reads a frozen snapshot and uses a chronological split.

    python -m train.train snapshot --db ../collector_app/data/draft_data.sqlite3
    python -m train.train priors   --snapshot data/snapshots/latest.npz
    python -m train.train value    --snapshot data/snapshots/latest.npz [--interactions fm|none|mlp]
    python -m train.train policy   --snapshot data/snapshots/latest.npz --memory engram|none|both
    python -m train.train promote  [--force]

Candidates land in models/candidates/ with a metrics JSON; `promote` copies a candidate to
models/active/ only when it beats its baseline on the untouched test window.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from draft4.catalog import load_catalog
from draft4.dataset import Matches, load_matches, time_split
from draft4.nets import OFFSET, PolicyNet, ValueNet, count_params
from draft4.priors import Priors

ROOT = Path(__file__).resolve().parent.parent
CAND, ACTIVE = ROOT / "models" / "candidates", ROOT / "models" / "active"


def device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def metrics_binary(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    return {"log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), "brier": float(((p - y) ** 2).mean()),
            "ece": float(ece), "accuracy": float(((p > 0.5) == y).mean()), "n": int(len(y))}


def write_ckpt(name: str, model, snapshot: Matches, metrics: dict, extra: dict | None = None) -> Path:
    CAND.mkdir(parents=True, exist_ok=True)
    version = f"{name}-{time.strftime('%Y%m%d-%H%M')}"
    path = CAND / f"{name}.pt"
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "config": model.config,
                "champ_ids": [int(x) for x in snapshot.champ_ids], "version": version, **(extra or {})}, path)
    (CAND / f"{name}.metrics.json").write_text(json.dumps({"version": version, **metrics}, indent=2))
    print(json.dumps({"saved": str(path), "version": version, **metrics}, indent=2))
    return path


# ------------------------------------------------------------------ snapshot / priors
def cmd_snapshot(a) -> None:
    cat = load_catalog()
    m = load_matches(a.db, cat, queue_ids=tuple(a.queue))
    out = a.out or ROOT / "data" / "snapshots" / f"snap-{time.strftime('%Y%m%d-%H%M')}.npz"
    digest = m.save(out)
    latest = out.parent / "latest.npz"
    shutil.copyfile(out, latest)
    manifest = {"snapshot": out.name, "hash": digest, "matches": len(m), "catalog": cat.version,
                "patches": dict(Counter(m.patch.tolist()).most_common()),
                "bans_recorded": float((m.bans >= 0).mean()) if len(m) else 0.0}
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def load_split(a):
    m = Matches.load(a.snapshot)
    if len(m) < 100:
        raise SystemExit(f"snapshot has only {len(m)} matches; need more data")
    return m, *time_split(m)


def cmd_priors(a) -> None:
    cat = load_catalog()
    m, train, calib, test = load_split(a)
    from draft4.models import AdditiveValue
    p = Priors.build(train, cat)
    calib_loss = {}
    for w in (0.0, 0.1, 0.25, 0.5, 1.0):  # pair-term weight chosen on the calibration window only
        p.pair_weight = w
        calib_loss[w] = metrics_binary(AdditiveValue(p).win_prob(calib.blue, calib.red), calib.blue_win.astype(float))["log_loss"]
    p.pair_weight = min(calib_loss, key=calib_loss.get)
    metrics = {"additive_test": metrics_binary(AdditiveValue(p).win_prob(test.blue, test.red), test.blue_win.astype(float)),
               "side_only_test": metrics_binary(np.full(len(test), train.blue_win.mean()), test.blue_win.astype(float)),
               "pair_weight": p.pair_weight, "calib_log_loss_by_weight": calib_loss, "fit_on": "all" if a.all else "train"}
    if a.all:  # production priors: same weight, every match
        weight, p = p.pair_weight, Priors.build(m, cat)
        p.pair_weight = weight
    CAND.mkdir(parents=True, exist_ok=True)
    p.save(CAND / "priors.npz")
    (CAND / "priors.metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


# ------------------------------------------------------------------ value model
def cmd_value(a) -> None:
    dev = device(a.device)
    m, train, calib, test = load_split(a)
    cat = load_catalog()
    C = len(m.champ_ids)
    tens = lambda x: torch.as_tensor(x + OFFSET, device=dev)
    data = {k: (tens(s.blue), tens(s.red), torch.as_tensor(s.blue_win, dtype=torch.float32, device=dev))
            for k, s in (("train", train), ("calib", calib), ("test", test))}

    def fit(interactions: str) -> tuple[ValueNet, float]:
        torch.manual_seed(a.seed)
        net = ValueNet(C, d=a.d, interactions=interactions).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        b, r, y = data["train"]
        best, best_state, bad = 1e9, None, 0
        for epoch in range(a.epochs):
            net.train()
            perm = torch.randperm(len(y), device=dev)
            for i in range(0, len(y), a.batch):
                idx = perm[i:i + a.batch]
                loss = F.binary_cross_entropy_with_logits(net(b[idx], r[idx]), y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                cb, cr, cy = data["calib"]
                val = F.binary_cross_entropy_with_logits(net(cb, cr), cy).item()
            if val < best - 1e-5:
                best, best_state, bad = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= a.patience:
                    break
        net.load_state_dict(best_state)
        # temperature scaling on the calibration window
        with torch.no_grad():
            cb, cr, cy = data["calib"]
            logits = net(cb, cr)
        temps = torch.linspace(0.5, 3.0, 51, device=dev)
        losses = [F.binary_cross_entropy_with_logits(logits / t, cy).item() for t in temps]
        return net, float(temps[int(np.argmin(losses))])

    def evaluate(net, temp) -> dict:
        with torch.no_grad():
            tb, tr, ty = data["test"]
            p = torch.sigmoid(net(tb, tr) / temp).cpu().numpy()
        return metrics_binary(p, ty.cpu().numpy())

    y_test = test.blue_win.astype(float)
    side_only = metrics_binary(np.full(len(test), train.blue_win.mean()), y_test)
    from draft4.models import AdditiveValue
    additive_priors = metrics_binary(AdditiveValue(Priors.build(train, cat)).win_prob(test.blue, test.red), y_test)
    t0 = time.time()
    net, temp = fit(a.interactions)
    main = evaluate(net, temp)
    baselines = {"side_only": side_only, "additive_priors": additive_priors}
    if a.interactions != "none":
        base_net, base_t = fit("none")
        baselines["main_effects_net"] = evaluate(base_net, base_t)
    best_base = min(v["log_loss"] for v in baselines.values())
    metrics = {"test": main, "baselines": baselines, "beats_baseline": main["log_loss"] < best_base,
               "temperature": temp, "params": count_params(net), "device": str(dev), "seconds": round(time.time() - t0, 1),
               "split": {"train": len(train), "calib": len(calib), "test": len(test)}}
    write_ckpt("value", net, m, metrics, {"temperature": temp})


# ------------------------------------------------------------------ policy model
def mask_batch(blue, red, bans, gen: torch.Generator):
    """Masked-lineup samples: hide the target slot and a random subset of the rest (ranked formulation)."""
    B, dev = blue.shape[0], blue.device
    flip = torch.rand(B, generator=gen, device=dev) < 0.5
    ally = torch.where(flip[:, None], red, blue)
    enemy = torch.where(flip[:, None], blue, red)
    role = torch.randint(0, 5, (B,), generator=gen, device=dev)
    target = ally[torch.arange(B, device=dev), role] - OFFSET
    keep_ally = torch.rand(B, 5, generator=gen, device=dev) < torch.rand(B, 1, generator=gen, device=dev)
    keep_ally[torch.arange(B, device=dev), role] = False
    keep_enemy = torch.rand(B, 5, generator=gen, device=dev) < torch.rand(B, 1, generator=gen, device=dev)
    show_bans = torch.rand(B, 1, generator=gen, device=dev) < 0.5  # final ranked bans as post-ban context only
    return (ally * keep_ally, enemy * keep_enemy, bans * show_bans, role, target)


def visible_mask(ally, enemy, bans, C: int) -> torch.Tensor:
    vis = torch.zeros(ally.shape[0], C + OFFSET, dtype=torch.bool, device=ally.device)
    vis.scatter_(1, torch.cat([ally, enemy, bans], 1), True)
    return vis[:, OFFSET:]


def cmd_policy(a) -> None:
    dev = device(a.device)
    m, train, calib, test = load_split(a)
    cat = load_catalog()
    C = len(m.champ_ids)

    def tensors(s: Matches):
        bans = np.where(s.bans >= 0, s.bans + OFFSET, 0).reshape(len(s), 10)
        return (torch.as_tensor(s.blue + OFFSET, device=dev), torch.as_tensor(s.red + OFFSET, device=dev),
                torch.as_tensor(bans, device=dev))

    tr, ca, te = tensors(train), tensors(calib), tensors(test)

    def fixed_eval_set(t, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        n = min(len(t[0]), a.eval_n)
        return mask_batch(t[0][:n], t[1][:n], t[2][:n], g)

    calib_set, test_set = fixed_eval_set(ca, 11), fixed_eval_set(te, 12)

    def score(logits_fn, sample) -> dict:
        ally, enemy, bans, role, target = sample
        losses, top1, top5 = [], [], []
        for i in range(0, len(role), 4096):
            sl = slice(i, i + 4096)
            logits = logits_fn(ally[sl], enemy[sl], bans[sl], role[sl])
            logits = logits.masked_fill(visible_mask(ally[sl], enemy[sl], bans[sl], C), -1e9)
            losses.append(F.cross_entropy(logits, target[sl], reduction="sum").item())
            rank = logits.topk(5, 1).indices
            top1.append((rank[:, 0] == target[sl]).sum().item()); top5.append((rank == target[sl, None]).any(1).sum().item())
        n = len(role)
        return {"log_loss": sum(losses) / n, "top1": sum(top1) / n, "top5": sum(top5) / n, "n": n}

    # smoothed-prior baseline, fitted on train only, vectorised on device
    pri = Priors.build(train, cat)
    log_pick = torch.as_tensor(np.log(pri.pick_rate.T), dtype=torch.float32, device=dev)
    syn = torch.as_tensor(pri.syn, dtype=torch.float32, device=dev)
    ctr = torch.as_tensor(pri.ctr, dtype=torch.float32, device=dev)
    pad_rows = lambda M: torch.cat([torch.zeros(OFFSET, M.shape[1], device=dev), M], 0)
    syn_p, ctrT_p = pad_rows(syn), pad_rows(ctr.T.contiguous())

    def prior_logits(ally, enemy, bans, role):
        return log_pick[role] + syn_p[ally].sum(1) + ctrT_p[enemy].sum(1)

    results = {"prior_baseline": score(prior_logits, test_set)}
    memories = ["none", "engram"] if a.memory == "both" else [a.memory]
    trained = {}
    for mem in memories:
        torch.manual_seed(a.seed)
        net = PolicyNet(C, d=a.d, memory=mem, memory_heads=a.memory_heads, memory_table=a.memory_table).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        best, best_state, bad, t0 = 1e9, None, 0, time.time()
        for epoch in range(a.epochs):
            net.train()
            perm = torch.randperm(len(tr[0]), device=dev)
            for _rep in range(a.masks_per_match):
                for i in range(0, len(perm), a.batch):
                    idx = perm[i:i + a.batch]
                    ally, enemy, bans, role, target = mask_batch(tr[0][idx], tr[1][idx], tr[2][idx], gen)
                    logits = net(ally, enemy, bans, role).masked_fill(visible_mask(ally, enemy, bans, C), -1e9)
                    loss = F.cross_entropy(logits, target)
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                val = score(net, calib_set)["log_loss"]
            print(f"[{mem}] epoch {epoch + 1}: calib log loss {val:.4f}")
            if val < best - 1e-4:
                best, best_state, bad = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= a.patience:
                    break
        net.load_state_dict(best_state)
        with torch.no_grad():
            results[f"policy_{mem}"] = {**score(net, test_set), "params": count_params(net), "seconds": round(time.time() - t0, 1)}
        trained[mem] = net
    winner = min(trained, key=lambda k: results[f"policy_{k}"]["log_loss"])
    metrics = {"test": results[f"policy_{winner}"], "memory": winner, "all": results,
               "beats_baseline": results[f"policy_{winner}"]["log_loss"] < results["prior_baseline"]["log_loss"],
               "engram_ablation": ({"delta_log_loss": results["policy_engram"]["log_loss"] - results["policy_none"]["log_loss"]}
                                   if len(trained) == 2 else None),
               "device": str(dev), "formulation": "ranked masked-lineup (eventual appearances, not pick order)"}
    write_ckpt("policy", trained[winner], m, metrics)


# ------------------------------------------------------------------ pro (ordered) policy
def pro_samples(d, dev):
    """One sample per recorded action: everything visible before it -> the actual next pick/ban.
    shortcut: earlier-pick roles are treated as known (usually evident in pro); fearless series locks
    from earlier games are not masked yet, upgrade once fearless series are tagged reliably."""
    from draft4.rules import SKIPPED
    cols = {k: [] for k in ("ally", "enemy", "bans", "role", "step", "target", "game")}
    for i in range(len(d)):
        team, bans = np.zeros((2, 5), dtype=np.int64), []
        for t, (side, kind, c, role) in enumerate(d.actions(i)):
            if c != SKIPPED:
                cols["ally"].append(team[side].copy()); cols["enemy"].append(team[1 - side].copy())
                cols["bans"].append((bans + [0] * 10)[:10]); cols["role"].append(5 if kind == "ban" else role)
                cols["step"].append(t); cols["target"].append(c); cols["game"].append(i)
            if kind == "ban":
                if c != SKIPPED:
                    bans.append(c + OFFSET)
            else:
                team[side, role] = c + OFFSET
    return {k: torch.as_tensor(np.array(v), device=dev) for k, v in cols.items()}


def cmd_pro_policy(a) -> None:
    from train.pro_import import ProDrafts, series_time_split
    dev = device(a.device)
    d = ProDrafts.load(a.snapshot)
    if a.since_year:
        d = d.subset(d.time >= int(np.datetime64(f"{a.since_year}-01-01", "ms").astype(np.int64)))
    train_d, calib_d, test_d = series_time_split(d)
    C = len(d.champ_ids)
    tr, ca, te = (pro_samples(x, dev) for x in (train_d, calib_d, test_d))

    def score(logits_fn, S) -> dict:
        out = {}
        for name, sel in (("all", None), ("ban", S["role"] == 5), ("pick", S["role"] < 5)):
            idx = torch.arange(len(S["target"]), device=dev) if sel is None else torch.where(sel)[0]
            loss = top1 = top5 = 0.0
            for i in range(0, len(idx), 8192):
                j = idx[i:i + 8192]
                lg = logits_fn(S["ally"][j], S["enemy"][j], S["bans"][j], S["role"][j], S["step"][j])
                lg = lg.masked_fill(visible_mask(S["ally"][j], S["enemy"][j], S["bans"][j], C), -1e9)
                loss += F.cross_entropy(lg, S["target"][j], reduction="sum").item()
                rank = lg.topk(5, 1).indices
                top1 += (rank[:, 0] == S["target"][j]).sum().item(); top5 += (rank == S["target"][j, None]).any(1).sum().item()
            n = max(len(idx), 1)
            out[name] = {"log_loss": loss / n, "top1": top1 / n, "top5": top5 / n, "n": len(idx)}
        return out

    def freq_baseline(S, last: int | None):
        keep = S["game"] >= (S["game"].max() - last) if last else torch.ones_like(S["game"], dtype=torch.bool)
        counts = torch.full((6, C), 0.5, device=dev)
        counts.index_put_((S["role"][keep], S["target"][keep]), torch.ones(int(keep.sum()), device=dev), accumulate=True)
        logp = counts.log()
        return lambda ally, enemy, bans, role, step: logp[role]

    results = {"freq_all_train": score(freq_baseline(tr, None), te), "freq_recent_train": score(freq_baseline(tr, a.recent), te)}
    trained = {}
    for init in ([False, True] if a.init_ranked == "both" else [a.init_ranked == "yes"]):
        torch.manual_seed(a.seed)
        net = PolicyNet(C, d=a.d, memory=a.memory, ordered=True).to(dev)
        tag = f"pro_{a.memory}" + ("_init" if init else "")
        if init:
            ranked = torch.load(ACTIVE / "policy.pt", map_location=dev, weights_only=True)
            if ranked["champ_ids"] != [int(x) for x in d.champ_ids]:
                raise SystemExit("ranked policy uses a different champion list; rebuild one of the snapshots")
            own = net.state_dict()
            own.update({k: v for k, v in ranked["state_dict"].items() if k in own and own[k].shape == v.shape})
            net.load_state_dict(own)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        fn = lambda ally, enemy, bans, role, step: net(ally, enemy, bans, role, step)
        best, best_state, bad = 1e9, None, 0
        for epoch in range(a.epochs):
            net.train()
            perm = torch.randperm(len(tr["target"]), device=dev)
            for i in range(0, len(perm), a.batch):
                j = perm[i:i + a.batch]
                lg = net(tr["ally"][j], tr["enemy"][j], tr["bans"][j], tr["role"][j], tr["step"][j])
                lg = lg.masked_fill(visible_mask(tr["ally"][j], tr["enemy"][j], tr["bans"][j], C), -1e9)
                loss = F.cross_entropy(lg, tr["target"][j])
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                val = score(fn, ca)["all"]["log_loss"]
            print(f"[{tag}] epoch {epoch + 1}: calib log loss {val:.4f}")
            if val < best - 1e-4:
                best, best_state, bad = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= a.patience:
                    break
        net.load_state_dict(best_state)
        with torch.no_grad():
            results[tag] = score(fn, te)
        trained[tag] = net
    winner = min(trained, key=lambda k: results[k]["all"]["log_loss"])
    best_base = min(results[k]["all"]["log_loss"] for k in ("freq_all_train", "freq_recent_train"))
    metrics = {"test": results[winner], "winner": winner, "all": results,
               "beats_baseline": results[winner]["all"]["log_loss"] < best_base, "device": str(dev),
               "split": {"train_games": len(train_d), "calib_games": len(calib_d), "test_games": len(test_d)},
               "formulation": "ordered next action (pick or ban) from verified pro prefixes"}
    write_ckpt("policy_pro", trained[winner], d, metrics)


# ------------------------------------------------------------------ promote
def cmd_promote(a) -> None:
    ACTIVE.mkdir(parents=True, exist_ok=True)
    for name, ext in (("priors", "npz"), ("pro_priors", "npz"), ("value", "pt"), ("policy", "pt"), ("policy_pro", "pt")):
        src = CAND / f"{name}.{ext}"
        if not src.exists():
            continue
        met_path = CAND / f"{name}.metrics.json"
        met = json.loads(met_path.read_text()) if met_path.exists() else {}
        if name not in ("priors", "pro_priors") and not met.get("beats_baseline") and not a.force:
            print(f"skip {name}: does not beat its baseline (use --force to override)")
            continue
        dst = ACTIVE / src.name
        if dst.exists():  # keep the previous artifact for rollback
            shutil.copyfile(dst, ACTIVE / f"{name}.previous.{ext}")
        shutil.copyfile(src, dst)
        if met_path.exists():
            shutil.copyfile(met_path, ACTIVE / met_path.name)
        print(f"promoted {name} -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot"); s.add_argument("--db", type=Path, required=True); s.add_argument("--out", type=Path)
    s.add_argument("--queue", type=int, nargs="+", default=[420])
    snap = ROOT / "data" / "snapshots" / "latest.npz"
    p = sub.add_parser("priors"); p.add_argument("--snapshot", type=Path, default=snap); p.add_argument("--all", action="store_true")
    for name in ("value", "policy"):
        x = sub.add_parser(name)
        x.add_argument("--snapshot", type=Path, default=snap)
        x.add_argument("--device", default="auto")
        x.add_argument("--epochs", type=int, default=30)
        x.add_argument("--patience", type=int, default=3)
        x.add_argument("--batch", type=int, default=2048)
        x.add_argument("--lr", type=float, default=3e-3)
        x.add_argument("--wd", type=float, default=1e-4)
        x.add_argument("--seed", type=int, default=0)
    vp = sub.choices["value"]; vp.add_argument("--d", type=int, default=16); vp.add_argument("--interactions", default="fm", choices=["none", "fm", "mlp"])
    pp = sub.choices["policy"]; pp.add_argument("--d", type=int, default=64)
    pp.add_argument("--memory", default="both", choices=["none", "engram", "both"])
    pp.add_argument("--memory-heads", type=int, default=2); pp.add_argument("--memory-table", type=int, default=4096)
    pp.add_argument("--masks-per-match", type=int, default=2); pp.add_argument("--eval-n", type=int, default=50000)
    pp2 = sub.add_parser("pro-policy")
    pp2.add_argument("--snapshot", type=Path, default=ROOT / "data" / "snapshots" / "pro_latest.npz")
    for flag, kw in (("--device", {"default": "auto"}), ("--epochs", {"type": int, "default": 60}), ("--patience", {"type": int, "default": 5}),
                     ("--batch", {"type": int, "default": 1024}), ("--lr", {"type": float, "default": 2e-3}),
                     ("--wd", {"type": float, "default": 1e-3}), ("--seed", {"type": int, "default": 0}), ("--d", {"type": int, "default": 64}),
                     ("--memory", {"default": "none", "choices": ["none", "engram"]}),
                     ("--init-ranked", {"default": "both", "choices": ["no", "yes", "both"]}),
                     ("--recent", {"type": int, "default": 1500, "help": "games in the recent-frequency baseline"}),
                     ("--since-year", {"type": int, "default": None})):
        pp2.add_argument(flag, **kw)
    pr = sub.add_parser("promote"); pr.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"snapshot": cmd_snapshot, "priors": cmd_priors, "value": cmd_value, "policy": cmd_policy, "pro-policy": cmd_pro_policy,
     "promote": cmd_promote}[a.cmd](a)


if __name__ == "__main__":
    main()
