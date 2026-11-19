"""End-to-end smoke test: synthetic DB -> snapshot -> priors/value/policy -> promote -> engine loads them."""

import sys

import pytest

torch = pytest.importorskip("torch")

from draft4.catalog import load_catalog
from draft4.models import NeuralPolicy, NeuralValue, load_models
from draft4.planner import Engine, Prefs
from draft4.rules import DraftState
from train import train
from train.synthetic import generate


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["train", *map(str, args)])
    train.main()


def test_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "CAND", tmp_path / "models" / "candidates")
    monkeypatch.setattr(train, "ACTIVE", tmp_path / "models" / "active")
    db, snap = tmp_path / "syn.sqlite3", tmp_path / "snap.npz"
    generate(db, 3000)
    run(monkeypatch, "snapshot", "--db", db, "--out", snap)
    run(monkeypatch, "priors", "--snapshot", snap)
    run(monkeypatch, "value", "--snapshot", snap, "--epochs", 2, "--device", "cpu")
    run(monkeypatch, "policy", "--snapshot", snap, "--epochs", 1, "--device", "cpu", "--memory", "both", "--eval-n", 2000)
    run(monkeypatch, "promote", "--force")
    cat = load_catalog()
    models = load_models(cat, tmp_path / "models")
    assert isinstance(models["policy"], NeuralPolicy) and isinstance(models["value"], NeuralValue), models["status"]
    res = Engine(cat, **models).analyze(DraftState("pro"), Prefs(budget_ms=300))
    assert res["kind"] == "recommend" and res["lines"]
