"""Torch networks for the two core models. Imported lazily: the engine runs without torch.

Token convention: 0 = empty/masked slot, 1 = champion unknown to this checkpoint, 2.. = champions.
"""

from __future__ import annotations

import math

import torch
from torch import nn

PAD, UNK, OFFSET = 0, 1, 2


class EngramMemory(nn.Module):
    """Engram-style conditional memory (arXiv:2601.07372), draft-sized starting point.

    Keys are hashed champion n-grams (here: pairs of role-tagged context slots). Each head hashes
    a pair into its own embedding table; the retrieved memory is gated by the current context
    representation so a recurring pattern informs, but does not dictate, the forecast.
    The Engram research session can swap keys/tokenization/placement behind this same interface.
    """

    def __init__(self, d: int, n_slots: int, heads: int = 2, table_size: int = 4096, order: int = 2) -> None:
        super().__init__()
        if order != 2:
            raise NotImplementedError("only pair n-grams for now")  # shortcut: pairs only, add triples if ablation shows value
        self.table_size = table_size
        self.tables = nn.ModuleList(nn.Embedding(table_size, d) for _ in range(heads))
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.02)
        g = torch.Generator().manual_seed(1234)
        self.register_buffer("mult", torch.randint(1, 2**20, (heads, 2), generator=g) * 2 + 1, persistent=True)
        i, j = torch.triu_indices(n_slots, n_slots, 1)
        self.register_buffer("pi", i, persistent=False)
        self.register_buffer("pj", j, persistent=False)
        self.query = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, keys: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # keys: (B, S) role-tagged slot keys (0 = empty); h: (B, d) context
        a, b = keys[:, self.pi], keys[:, self.pj]
        valid = ((a > 0) & (b > 0)).float()
        mem = 0
        for k, table in enumerate(self.tables):
            idx = ((a * self.mult[k, 0]) ^ (b * self.mult[k, 1])) % self.table_size
            mem = mem + table(idx)
        gate = torch.sigmoid((mem * self.query(h)[:, None, :]).sum(-1) / math.sqrt(h.shape[-1]))
        pooled = (mem * (gate * valid)[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        return self.out(pooled)


class PolicyNet(nn.Module):
    """Role-aware set encoder: P(champion | target role, visible allies, enemies, bans)."""

    def __init__(self, n_champions: int, d: int = 64, memory: str = "none", memory_heads: int = 2,
                 memory_table: int = 4096, ordered: bool = False) -> None:
        super().__init__()
        self.config = {"n_champions": n_champions, "d": d, "memory": memory,
                       "memory_heads": memory_heads, "memory_table": memory_table, "ordered": ordered}
        vocab = n_champions + OFFSET
        self.champ = nn.Embedding(vocab, d, padding_idx=PAD)
        self.slot = nn.Embedding(5 + 5 + 1, d)  # ally role r, enemy role r, ban
        # ordered (pro) drafts: target 5 = ban, plus the position in the 20-step sequence
        self.target = nn.Embedding(6 if ordered else 5, d)
        self.step = nn.Embedding(20, d) if ordered else None
        self.enc = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.mix = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.memory = EngramMemory(d, 20, memory_heads, memory_table) if memory == "engram" else None
        self.bias = nn.Parameter(torch.zeros(n_champions))
        slot_ids = torch.cat([torch.arange(5), 5 + torch.arange(5), torch.full((10,), 10)])
        self.register_buffer("slot_ids", slot_ids, persistent=False)

    def forward(self, ally: torch.Tensor, enemy: torch.Tensor, bans: torch.Tensor, role: torch.Tensor,
                step: torch.Tensor | None = None) -> torch.Tensor:
        tokens = torch.cat([ally, enemy, bans], 1)  # (B,20)
        present = (tokens > PAD).float()[..., None]
        x = self.enc(self.champ(tokens) + self.slot(self.slot_ids)[None]) * present
        pool = lambda s: x[:, s].sum(1) / present[:, s].sum(1).clamp(min=1)
        target = self.target(role) + (self.step(step) if self.step is not None else 0)
        h = self.mix(torch.cat([pool(slice(0, 5)), pool(slice(5, 10)), pool(slice(10, 20)), target], -1))
        if self.memory is not None:
            keys = torch.where(tokens > PAD, tokens * 16 + self.slot_ids[None] + 1, torch.zeros_like(tokens))
            h = h + self.memory(keys, h)
        return h @ self.champ.weight[OFFSET:].T + self.bias  # (B, C)


class ValueNet(nn.Module):
    """Antisymmetric lineup value: logit P(blue wins) = side + g(blue, red) - g(red, blue)."""

    def __init__(self, n_champions: int, d: int = 16, interactions: str = "fm") -> None:
        super().__init__()
        if interactions not in ("none", "fm", "mlp"):
            raise ValueError(interactions)
        self.config = {"n_champions": n_champions, "d": d, "interactions": interactions}
        vocab = n_champions + OFFSET
        self.main = nn.Embedding(vocab * 5, 1)
        nn.init.zeros_(self.main.weight)
        self.side = nn.Parameter(torch.zeros(()))
        self.interactions = interactions
        if interactions != "none":
            self.role = nn.Embedding(5, d)
            self.u, self.v, self.w = (nn.Embedding(vocab, d) for _ in range(3))
            for e in (self.u, self.v, self.w):
                nn.init.normal_(e.weight, std=0.01)
        if interactions == "mlp":
            self.mlp = nn.Sequential(nn.Linear(4 * d, 4 * d), nn.GELU(), nn.Linear(4 * d, 1))
        self.register_buffer("roles", torch.arange(5), persistent=False)

    def _g(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = self.main(a * 5 + self.roles).squeeze(-1).sum(1)
        if self.interactions == "none":
            return out
        r = self.role(self.roles)[None]
        ua, va, wb = self.u(a) + r, self.v(a) + r, self.w(b) + r
        s = ua.sum(1)
        out = out + 0.5 * ((s * s).sum(-1) - (ua * ua).sum((1, 2)))  # FM pairwise synergy
        out = out + torch.einsum("bid,bjd->b", va, wb)  # counters
        if self.interactions == "mlp":
            out = out + self.mlp(torch.cat([ua.mean(1), va.mean(1), wb.mean(1), (ua.mean(1) * wb.mean(1))], -1)).squeeze(-1)
        return out

    def forward(self, blue: torch.Tensor, red: torch.Tensor) -> torch.Tensor:
        return self.side + self._g(blue, red) - self._g(red, blue)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
