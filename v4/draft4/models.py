"""Engine-side model wrappers. Baselines always work; trained checkpoints load when promoted."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .catalog import Catalog
from .priors import Priors

LOG = logging.getLogger("draft4.models")
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
NEG = -1e9


def _token_map(ckpt_ids: list[int], catalog: Catalog) -> np.ndarray:
    """catalog idx -> checkpoint token (2.. known, 1 = unknown to the checkpoint)."""
    pos = {cid: i for i, cid in enumerate(ckpt_ids)}
    return np.array([pos[c.id] + 2 if c.id in pos else 1 for c in catalog.champions], dtype=np.int64)


class PriorPolicy:
    """Smoothed popularity policy with synergy/counter tilt. The fallback and the baseline."""
    version = "prior"

    def __init__(self, priors: Priors) -> None:
        self.p = priors
        self.log_pick = np.log(priors.pick_rate.T)  # (5, C)
        self.log_ban = np.log(priors.ban_rate + 1e-4)

    def role_logits(self, ally: np.ndarray, enemy: np.ndarray, bans: np.ndarray, step: int | None = None) -> np.ndarray:
        """ally/enemy: (5,) role-ordered idx, -1 empty. Returns (5, C) logits per target role.
        `step` is the position in an ordered (pro) draft; unordered policies ignore it."""
        a, e = ally[ally >= 0], enemy[enemy >= 0]
        tilt = self.p.syn[a].sum(0) + self.p.ctr[:, e].sum(1) if len(a) or len(e) else 0.0
        return self.log_pick + tilt

    def ban_logits(self, our_team: np.ndarray, their_team: np.ndarray, bans=None, step: int | None = None) -> np.ndarray:
        """Bans by `their` side: popular bans tilted toward champions strong into `our` lineup."""
        o = our_team[our_team >= 0]
        threat = -self.p.ctr[o].sum(0) if len(o) else 0.0  # ctr[o, x] < 0 means x does well into o
        return self.log_ban + threat + 0.5 * self.p.win_logit.max(1)


class NeuralPolicy(PriorPolicy):
    def __init__(self, priors: Priors, path: Path, catalog: Catalog) -> None:
        import torch
        from .nets import PolicyNet
        super().__init__(priors)
        torch.set_num_threads(1)  # tiny per-step forwards: thread start-up costs more than the maths
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.net = PolicyNet(**ckpt["config"]).eval()
        self.net.load_state_dict(ckpt["state_dict"])
        self.tok = _token_map(ckpt["champ_ids"], catalog)
        self.out_rows = np.where(self.tok >= 2)[0]  # catalog idx known to the model
        self.out_cols = self.tok[self.out_rows] - 2
        self.version = ckpt.get("version", path.stem)
        self.ordered = bool(ckpt["config"].get("ordered"))
        self.torch = torch

    def _net(self, ally, enemy, bans, roles: list[int], step: int | None) -> np.ndarray:
        t = self.torch
        n = len(roles)
        with t.no_grad():
            args = [t.tensor([self._tokens(x, k)] * n) for x, k in ((ally, 5), (enemy, 5), (bans, 10))]
            step_t = t.full((n,), 6 if step is None else int(step)) if self.ordered else None  # 6 = first pick slot
            return self.net(*args, t.tensor(roles), step_t).numpy()

    def _tokens(self, arr: np.ndarray, n: int) -> list[int]:
        arr = np.asarray(arr)
        out = [int(self.tok[c]) if c >= 0 else 0 for c in arr[:n]]
        return out + [0] * (n - len(out))

    def role_logits(self, ally, enemy, bans, step=None) -> np.ndarray:
        logits = self._net(ally, enemy, bans, list(range(5)), step)
        out = super().role_logits(ally, enemy, bans) - 8.0  # unseen champions: far below known ones
        out[:, self.out_rows] = logits[:, self.out_cols]
        return out

    def ban_logits(self, our_team, their_team, bans=None, step=None) -> np.ndarray:
        if not self.ordered:  # ranked policy has no ban head
            return super().ban_logits(our_team, their_team)
        logits = self._net(their_team, our_team, [] if bans is None else bans, [5], step)[0]  # banner's view
        out = super().ban_logits(our_team, their_team) - 8.0
        out[self.out_rows] = logits[self.out_cols]
        return out


class AdditiveValue:
    """Side prior + champion-role main effects + shrunk synergy/counter residuals."""
    version = "additive"

    def __init__(self, priors: Priors) -> None:
        self.p = priors
        self.iu, self.ju = np.triu_indices(5, 1)

    def logits(self, blue: np.ndarray, red: np.ndarray) -> np.ndarray:
        """blue/red: (B,5) role-ordered idx. Returns logit P(blue wins)."""
        p, r = self.p, np.arange(5)
        main = p.win_logit[blue, r].sum(1) - p.win_logit[red, r].sum(1)
        syn = p.syn[blue[:, self.iu], blue[:, self.ju]].sum(1) - p.syn[red[:, self.iu], red[:, self.ju]].sum(1)
        ctr = p.ctr[blue[:, :, None], red[:, None, :]].sum((1, 2))
        return p.side_logit + main + p.pair_weight * (syn + ctr)

    def win_prob(self, blue: np.ndarray, red: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logits(blue, red)))

    def coverage(self, blue: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Fraction of the 10 champion-role slots with >= 200 observed games."""
        r = np.arange(5)
        g = np.concatenate([self.p.games[blue, r], self.p.games[red, r]], 1)
        return (g >= 200).mean(1)


class NeuralValue(AdditiveValue):
    def __init__(self, priors: Priors, path: Path, catalog: Catalog) -> None:
        import torch
        from .nets import ValueNet
        super().__init__(priors)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.net = ValueNet(**ckpt["config"]).eval()
        self.net.load_state_dict(ckpt["state_dict"])
        self.temperature = float(ckpt.get("temperature", 1.0))
        self.tok = _token_map(ckpt["champ_ids"], catalog)
        self.version = ckpt.get("version", path.stem)
        self.torch = torch

    def logits(self, blue, red) -> np.ndarray:
        t = self.torch
        with t.no_grad():
            out = self.net(t.from_numpy(self.tok[blue]), t.from_numpy(self.tok[red])).numpy()
        return out / self.temperature


def load_models(catalog: Catalog, models_dir: Path = MODELS_DIR):
    """Return Engine keyword arguments. Missing/broken checkpoints fall back to baselines; pro_policy and
    pro_role_rate are None when nothing pro is promoted (pro drafts then use the ranked evidence)."""
    status: dict = {"warnings": []}
    pri_path = models_dir / "active" / "priors.npz"
    priors = Priors.load(pri_path, catalog) if pri_path.exists() else Priors.from_seed(catalog)
    status["priors"] = {k: v for k, v in priors.meta.items() if k != "champ_ids"}
    policy, value, pro_policy = PriorPolicy(priors), AdditiveValue(priors), None
    for name, cls in (("policy", NeuralPolicy), ("value", NeuralValue), ("policy_pro", NeuralPolicy)):
        path = models_dir / "active" / f"{name}.pt"
        if not path.exists():
            continue
        try:
            obj = cls(priors, path, catalog)
            if name == "policy":
                policy = obj
            elif name == "value":
                value = obj
            else:
                pro_policy = obj
        except Exception as exc:  # keep the tool usable with baselines
            status["warnings"].append(f"{name} checkpoint not loaded: {exc}")
            LOG.warning("failed to load %s: %s", path, exc)
    status["policy"], status["value"] = policy.version, value.version
    status["pro_policy"] = pro_policy.version if pro_policy else None
    pro_role_rate = pro_games = None
    rr_path = models_dir / "active" / "pro_priors.npz"
    if rr_path.exists():
        with np.load(rr_path) as d:
            if d["champ_ids"].tolist() == [c.id for c in catalog.champions]:
                pro_role_rate, pro_games = d["role_rate"], d["games"].sum(1)
            else:
                status["warnings"].append("pro role rates use an old champion list; re-run train.pro_import and promote")
    if priors.meta.get("source") == "seed":
        status["warnings"].append("No match data yet: forecasts use seed roles only and win estimates are flat (~50%).")
    return {"priors": priors, "policy": policy, "value": value, "status": status,
            "pro_policy": pro_policy, "pro_role_rate": pro_role_rate, "pro_games": pro_games}
