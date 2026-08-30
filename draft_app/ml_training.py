"""GPU-capable champion-embedding ensemble training for League Draft Lab v3.

This module is imported lazily by ``analytics_builder``.  If PyTorch is not
installed, statistical analytics still build and the UI explains how to enable
GPU training.  Successful training exports a compact NumPy runtime package.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ml_features import MODEL_FEATURE_NAMES, ROLES, aggregate_team, model_feature_vector
from patch_utils import patch_label, recency_weight
from runtime_paths import PROJECT_ROOT

LOGGER = logging.getLogger("ml_training")
MODEL_PATH = PROJECT_ROOT / "data" / "draft_model_v3.npz"
META_PATH = PROJECT_ROOT / "data" / "draft_model_v3.json"


@dataclass(frozen=True, slots=True)
class NeuralTrainingReport:
    trained: bool
    backend: str
    device: str
    examples: int
    matches: int
    validation_accuracy: float
    validation_brier: float
    reason: str = ""


def torch_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "cuda": False, "device": "not installed", "error": str(exc)}
    cuda = bool(torch.cuda.is_available())
    return {
        "available": True,
        "cuda": cuda,
        "device": torch.cuda.get_device_name(0) if cuda else "CPU",
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda or ""),
    }


def _complete_matches(participants: pd.DataFrame) -> set[str]:
    complete = participants.groupby(["match_id", "team_id"])["role"].nunique().reset_index(name="roles")
    per_match = complete[complete["roles"] == 5].groupby("match_id").size()
    return set(str(x) for x in per_match[per_match == 2].index)


def _feature_lookup(frame: pd.DataFrame) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row.role), int(row.champion_id)): row._asdict()
        for row in frame.itertuples(index=False)
    }


def _build_dataset(
    participants: pd.DataFrame,
    matches: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    current_patch: str,
    half_life_patches: float,
    minimum_patch_weight: float,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    valid_ids = _complete_matches(participants)
    if not valid_ids:
        return [], {}
    match_lookup = matches.set_index("match_id").to_dict("index")
    features = _feature_lookup(feature_frame)
    champion_ids = sorted(int(x) for x in participants["champion_id"].unique() if int(x) > 0)
    champion_to_index = {champion_id: index + 1 for index, champion_id in enumerate(champion_ids)}
    records: list[dict[str, Any]] = []

    for match_id, match in participants[participants["match_id"].isin(valid_ids)].groupby("match_id"):
        teams = sorted(int(x) for x in match["team_id"].unique())
        if len(teams) != 2:
            continue
        raw_meta = match_lookup.get(match_id, {})
        weight = recency_weight(
            raw_meta.get("game_version", ""), current_patch,
            half_life_patches=half_life_patches,
            minimum_weight=minimum_patch_weight,
        )
        for ally_team, enemy_team in (teams, teams[::-1]):
            allies = match[match["team_id"] == ally_team]
            enemies = match[match["team_id"] == enemy_team]
            ally_by_role = {str(row.role): int(row.champion_id) for row in allies.itertuples(index=False)}
            enemy_by_role = {str(row.role): int(row.champion_id) for row in enemies.itertuples(index=False)}
            if set(ally_by_role) != set(ROLES) or set(enemy_by_role) != set(ROLES):
                continue
            ally_metrics = aggregate_team(
                ally_by_role.values(), {champ: role for role, champ in ally_by_role.items()}, features
            )
            enemy_metrics = aggregate_team(
                enemy_by_role.values(), {champ: role for role, champ in enemy_by_role.items()}, features
            )
            feature_map = model_feature_vector(ally_metrics, enemy_metrics)
            records.append({
                "match_id": str(match_id),
                "game_creation": int(raw_meta.get("game_creation", 0) or 0),
                "ally_tokens": [champion_to_index[ally_by_role[role]] for role in ROLES],
                "enemy_tokens": [champion_to_index[enemy_by_role[role]] for role in ROLES],
                "features": [float(feature_map[name]) for name in MODEL_FEATURE_NAMES],
                "label": int(bool(allies["win"].iloc[0])),
                "weight": float(weight),
            })
    return records, champion_to_index


def _temperature(prob_logits: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.linspace(0.55, 2.2, 67)
    best = 1.0
    best_loss = float("inf")
    for value in candidates:
        logits = prob_logits / value
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        loss = -np.mean(
            labels * np.log(np.clip(probabilities, 1e-7, 1.0))
            + (1 - labels) * np.log(np.clip(1 - probabilities, 1e-7, 1.0))
        )
        if loss < best_loss:
            best_loss = float(loss)
            best = float(value)
    return best


def train_neural_ensemble(
    participants: pd.DataFrame,
    matches: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    current_patch: str,
    settings: Mapping[str, Any],
    model_path: Path = MODEL_PATH,
    meta_path: Path = META_PATH,
) -> NeuralTrainingReport:
    status = torch_status()
    if not status["available"]:
        reason = "PyTorch is not installed. Run setup_gpu_ml.bat to enable the v3 neural model."
        LOGGER.warning(reason)
        return NeuralTrainingReport(False, "unavailable", "none", 0, 0, 0.0, 0.25, reason)

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    half_life = float(settings.get("patch_half_life", 3.0))
    minimum_weight = float(settings.get("minimum_patch_weight", 0.12))
    records, champion_to_index = _build_dataset(
        participants, matches, feature_frame,
        current_patch=current_patch,
        half_life_patches=half_life,
        minimum_patch_weight=minimum_weight,
    )
    unique_matches = len({record["match_id"] for record in records})
    minimum_matches = max(100, int(settings.get("minimum_training_matches", 250)))
    if unique_matches < minimum_matches:
        reason = f"Only {unique_matches} complete matches; neural training requires {minimum_matches}."
        LOGGER.warning(reason)
        return NeuralTrainingReport(False, "pytorch", str(status["device"]), len(records), unique_matches, 0.0, 0.25, reason)

    # Sort by creation time and validate on the newest games, preventing future
    # patches from leaking into the older training split.
    records.sort(key=lambda item: (item["game_creation"], item["match_id"], item["label"]))
    split = max(2, int(len(records) * 0.82))
    train_records = records[:split]
    validation_records = records[split:]
    if len(validation_records) < 20:
        validation_records = records[-20:]
        train_records = records[:-20]

    def arrays(source: Sequence[dict[str, Any]]) -> tuple[np.ndarray, ...]:
        return (
            np.asarray([row["ally_tokens"] for row in source], dtype=np.int64),
            np.asarray([row["enemy_tokens"] for row in source], dtype=np.int64),
            np.asarray([row["features"] for row in source], dtype=np.float32),
            np.asarray([row["label"] for row in source], dtype=np.float32),
            np.asarray([row["weight"] for row in source], dtype=np.float32),
        )

    train_ally, train_enemy, train_features, train_labels, train_weights = arrays(train_records)
    val_ally, val_enemy, val_features, val_labels, val_weights = arrays(validation_records)
    feature_mean = train_features.mean(axis=0).astype(np.float32)
    feature_std = train_features.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-4] = 1.0
    train_features = (train_features - feature_mean) / feature_std
    val_features = (val_features - feature_mean) / feature_std

    champion_count = max(champion_to_index.values()) + 1
    embedding_dim = max(8, int(settings.get("embedding_dimension", 24)))
    role_dim = max(3, int(settings.get("role_embedding_dimension", 6)))
    slot_dim = max(16, int(settings.get("slot_hidden_dimension", 32)))
    hidden_dim = max(24, int(settings.get("hidden_dimension", 64)))
    ensemble_size = max(1, min(5, int(settings.get("ensemble_size", 3))))
    epochs = max(10, min(250, int(settings.get("epochs", 80))))
    patience = max(3, min(40, int(settings.get("early_stopping_patience", 10))))
    batch_size = max(32, min(2048, int(settings.get("batch_size", 256))))
    learning_rate = float(settings.get("learning_rate", 0.0025))
    dropout = max(0.0, min(0.5, float(settings.get("dropout", 0.12))))
    gpu_requested = bool(settings.get("use_gpu", True))
    device = torch.device("cuda" if status["cuda"] and gpu_requested else "cpu")
    use_amp = device.type == "cuda" and bool(settings.get("mixed_precision", True))
    if gpu_requested and device.type != "cuda":
        # Avoid an accidental multi-hour CPU rebuild when the user expected CUDA.
        LOGGER.warning(
            "CUDA was requested but is unavailable. Falling back to a reduced CPU "
            "training run (one ensemble member, at most 25 epochs)."
        )
        ensemble_size = 1
        epochs = min(epochs, 25)
        patience = min(patience, 6)
        batch_size = max(batch_size, 256)

    class DraftNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.champion_embedding = nn.Embedding(champion_count, embedding_dim, padding_idx=0)
            self.role_embedding = nn.Embedding(len(ROLES), role_dim)
            self.slot = nn.Linear(embedding_dim + role_dim, slot_dim)
            input_dim = slot_dim * 2 + len(MODEL_FEATURE_NAMES)
            self.head1 = nn.Linear(input_dim, hidden_dim)
            self.head2 = nn.Linear(hidden_dim, max(16, hidden_dim // 2))
            self.output = nn.Linear(max(16, hidden_dim // 2), 1)
            self.dropout = nn.Dropout(dropout)

        def encode(self, tokens: torch.Tensor) -> torch.Tensor:
            batch = tokens.shape[0]
            role_ids = torch.arange(len(ROLES), device=tokens.device).unsqueeze(0).expand(batch, -1)
            slots = torch.cat([self.champion_embedding(tokens), self.role_embedding(role_ids)], dim=-1)
            hidden = torch.nn.functional.gelu(self.slot(slots))
            mask = (tokens != 0).float().unsqueeze(-1)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        def forward(self, allies: torch.Tensor, enemies: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
            ally = self.encode(allies)
            enemy = self.encode(enemies)
            combined = torch.cat([ally - enemy, torch.abs(ally - enemy), features], dim=1)
            hidden = self.dropout(torch.nn.functional.gelu(self.head1(combined)))
            hidden = self.dropout(torch.nn.functional.gelu(self.head2(hidden)))
            return self.output(hidden).squeeze(1)

    train_dataset = TensorDataset(
        torch.from_numpy(train_ally), torch.from_numpy(train_enemy),
        torch.from_numpy(train_features), torch.from_numpy(train_labels),
        torch.from_numpy(train_weights),
    )
    val_tensors = (
        torch.from_numpy(val_ally).to(device), torch.from_numpy(val_enemy).to(device),
        torch.from_numpy(val_features).to(device), torch.from_numpy(val_labels).to(device),
        torch.from_numpy(val_weights).to(device),
    )
    exported: dict[str, np.ndarray] = {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }
    validation_logits: list[np.ndarray] = []
    LOGGER.info(
        "Training v3 neural ensemble: %d matches / %d examples / %d model(s) / "
        "%d max epochs on %s (AMP=%s).",
        unique_matches, len(records), ensemble_size, epochs, device, use_amp,
    )

    for model_index in range(ensemble_size):
        seed = 4100 + model_index * 97
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = DraftNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
        best_state: dict[str, torch.Tensor] | None = None
        best_loss = float("inf")
        stale = 0

        for epoch in range(epochs):
            model.train()
            for ally_batch, enemy_batch, feature_batch, label_batch, weight_batch in loader:
                ally_batch = ally_batch.to(device, non_blocking=True)
                enemy_batch = enemy_batch.to(device, non_blocking=True)
                feature_batch = feature_batch.to(device, non_blocking=True)
                label_batch = label_batch.to(device, non_blocking=True)
                weight_batch = weight_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(ally_batch, enemy_batch, feature_batch)
                    loss_values = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, label_batch, reduction="none"
                    )
                    loss = (loss_values * weight_batch).sum() / weight_batch.sum().clamp_min(1.0)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
                scaler.step(optimizer)
                scaler.update()

            model.eval()
            with torch.no_grad():
                val_logits = model(val_tensors[0], val_tensors[1], val_tensors[2])
                val_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    val_logits, val_tensors[3], reduction="none"
                )
                val_loss = float((val_losses * val_tensors[4]).sum() / val_tensors[4].sum().clamp_min(1.0))
            if epoch == 0 or (epoch + 1) % 10 == 0:
                LOGGER.info(
                    "Neural member %d/%d epoch %d/%d · validation loss %.4f.",
                    model_index + 1, ensemble_size, epoch + 1, epochs, val_loss,
                )
            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_state is None:
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.to(device).eval()
        with torch.no_grad():
            logits = model(val_tensors[0], val_tensors[1], val_tensors[2]).float().cpu().numpy()
        validation_logits.append(logits)

        state = {name: tensor.detach().cpu().numpy().astype(np.float32) for name, tensor in best_state.items()}
        prefix = f"m{model_index}_"
        exported[prefix + "champion_embedding"] = state["champion_embedding.weight"]
        exported[prefix + "role_embedding"] = state["role_embedding.weight"]
        exported[prefix + "slot_w"] = state["slot.weight"].T
        exported[prefix + "slot_b"] = state["slot.bias"]
        exported[prefix + "head1_w"] = state["head1.weight"].T
        exported[prefix + "head1_b"] = state["head1.bias"]
        exported[prefix + "head2_w"] = state["head2.weight"].T
        exported[prefix + "head2_b"] = state["head2.bias"]
        exported[prefix + "out_w"] = state["output.weight"].reshape(-1)
        exported[prefix + "out_b"] = state["output.bias"].reshape(())
        LOGGER.info("Neural ensemble member %d/%d trained on %s.", model_index + 1, ensemble_size, device)

    mean_logits = np.mean(np.vstack(validation_logits), axis=0)
    temperature = _temperature(mean_logits, val_labels)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(mean_logits / temperature, -30, 30)))
    accuracy = float(np.mean((probabilities >= 0.5) == (val_labels >= 0.5)))
    brier = float(np.mean((probabilities - val_labels) ** 2))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **exported)
    temporary.replace(model_path)
    metadata = {
        "model_version": "3.0.0",
        "backend": "PyTorch embedding ensemble",
        "device": str(status["device"] if device.type == "cuda" else "CPU"),
        "torch_version": str(status.get("torch_version", "")),
        "cuda_version": str(status.get("cuda_version", "")),
        "trained_examples": len(records),
        "trained_matches": unique_matches,
        "validation_examples": len(validation_records),
        "validation_accuracy": accuracy,
        "validation_brier": brier,
        "temperature": temperature,
        "current_patch": patch_label(current_patch),
        "built_at": datetime.now(UTC).isoformat(),
        "embedding_dimension": embedding_dim,
        "ensemble_size": ensemble_size,
        "feature_names": list(MODEL_FEATURE_NAMES),
        "champion_to_index": {str(key): value for key, value in champion_to_index.items()},
        "patch_half_life": half_life,
        "minimum_patch_weight": minimum_weight,
    }
    temporary_meta = meta_path.with_suffix(".tmp.json")
    temporary_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_meta.replace(meta_path)
    LOGGER.info(
        "V3 neural model complete: %d matches, %.1f%% validation accuracy, Brier %.4f.",
        unique_matches, accuracy * 100.0, brier,
    )
    return NeuralTrainingReport(
        True, "pytorch", str(metadata["device"]), len(records), unique_matches, accuracy, brier
    )
