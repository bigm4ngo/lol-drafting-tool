"""Portable NumPy inference runtime for the v3 neural draft ensemble.

Training uses PyTorch when available, but the exported model is deliberately a
small ``.npz`` package.  Live Champion Select and the packaged EXE therefore do
not need to import or bundle PyTorch.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from runtime_paths import PROJECT_ROOT

LOGGER = logging.getLogger("ml_runtime")
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "draft_model_v3.npz"
DEFAULT_META_PATH = PROJECT_ROOT / "data" / "draft_model_v3.json"
ROLES = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")
ROLE_INDEX = {role: index for index, role in enumerate(ROLES)}


@dataclass(frozen=True, slots=True)
class NeuralPrediction:
    probability: float
    confidence: float
    ensemble_std: float
    model_count: int
    available: bool


@dataclass(frozen=True, slots=True)
class ModelStatus:
    available: bool
    backend: str
    device: str
    trained_matches: int
    validation_accuracy: float
    validation_brier: float
    current_patch: str
    built_at: str
    embedding_dimension: int


class PortableDraftModel:
    """Loads an exported ensemble and evaluates complete or partial drafts."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        meta_path: Path = DEFAULT_META_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.meta: dict[str, Any] = {}
        self.arrays: dict[str, np.ndarray] = {}
        self.champion_to_index: dict[int, int] = {}
        self.feature_names: tuple[str, ...] = ()
        self.model_count = 0
        self.reload()

    @property
    def available(self) -> bool:
        return bool(self.arrays and self.model_count > 0)

    @property
    def status(self) -> ModelStatus:
        return ModelStatus(
            available=self.available,
            backend=str(self.meta.get("backend", "not trained")),
            device=str(self.meta.get("device", "none")),
            trained_matches=int(self.meta.get("trained_matches", 0) or 0),
            validation_accuracy=float(self.meta.get("validation_accuracy", 0.0) or 0.0),
            validation_brier=float(self.meta.get("validation_brier", 0.25) or 0.25),
            current_patch=str(self.meta.get("current_patch", "unknown")),
            built_at=str(self.meta.get("built_at", "")),
            embedding_dimension=int(self.meta.get("embedding_dimension", 0) or 0),
        )

    def reload(self) -> None:
        self.meta = {}
        self.arrays = {}
        self.champion_to_index = {}
        self.feature_names = ()
        self.model_count = 0
        if not self.model_path.exists() or not self.meta_path.exists():
            return
        try:
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            with np.load(self.model_path, allow_pickle=False) as package:
                self.arrays = {key: package[key] for key in package.files}
            raw_mapping = self.meta.get("champion_to_index", {})
            self.champion_to_index = {int(key): int(value) for key, value in raw_mapping.items()}
            self.feature_names = tuple(str(x) for x in self.meta.get("feature_names", []))
            self.model_count = int(self.meta.get("ensemble_size", 0) or 0)
            required = {"feature_mean", "feature_std"}
            if not required.issubset(self.arrays):
                raise ValueError("Model package is missing feature normalization arrays.")
        except Exception:
            LOGGER.exception("Could not load v3 portable neural model.")
            self.meta = {}
            self.arrays = {}
            self.model_count = 0

    @staticmethod
    def _gelu(value: np.ndarray) -> np.ndarray:
        return 0.5 * value * (
            1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (value + 0.044715 * value**3))
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def _tokens(self, picks: Mapping[str, int] | Sequence[tuple[str, int]]) -> np.ndarray:
        mapping = dict(picks)
        return np.asarray(
            [self.champion_to_index.get(int(mapping.get(role, 0) or 0), 0) for role in ROLES],
            dtype=np.int64,
        )

    def _one_model_probability(
        self,
        model_index: int,
        ally_tokens: np.ndarray,
        enemy_tokens: np.ndarray,
        feature_vector: np.ndarray,
    ) -> float:
        prefix = f"m{model_index}_"
        champion_embedding = self.arrays[prefix + "champion_embedding"]
        role_embedding = self.arrays[prefix + "role_embedding"]
        role_ids = np.arange(len(ROLES), dtype=np.int64)

        def encode(tokens: np.ndarray) -> np.ndarray:
            champion = champion_embedding[tokens]
            roles = role_embedding[role_ids]
            slots = np.concatenate([champion, roles], axis=1)
            hidden = self._gelu(slots @ self.arrays[prefix + "slot_w"] + self.arrays[prefix + "slot_b"])
            present = (tokens != 0).astype(np.float32)[:, None]
            denominator = max(1.0, float(present.sum()))
            return (hidden * present).sum(axis=0) / denominator

        ally = encode(ally_tokens)
        enemy = encode(enemy_tokens)
        normalized = (feature_vector - self.arrays["feature_mean"]) / self.arrays["feature_std"]
        combined = np.concatenate([ally - enemy, np.abs(ally - enemy), normalized], axis=0)
        hidden1 = self._gelu(combined @ self.arrays[prefix + "head1_w"] + self.arrays[prefix + "head1_b"])
        hidden2 = self._gelu(hidden1 @ self.arrays[prefix + "head2_w"] + self.arrays[prefix + "head2_b"])
        logit = float(hidden2 @ self.arrays[prefix + "out_w"] + self.arrays[prefix + "out_b"])
        temperature = max(0.35, float(self.meta.get("temperature", 1.0) or 1.0))
        return self._sigmoid(logit / temperature)

    def predict(
        self,
        allies: Mapping[str, int] | Sequence[tuple[str, int]],
        enemies: Mapping[str, int] | Sequence[tuple[str, int]],
        handcrafted: Mapping[str, float] | Sequence[float],
        *,
        data_coverage: float = 0.5,
        role_confidence: float = 0.5,
    ) -> NeuralPrediction:
        if not self.available:
            return NeuralPrediction(0.5, 0.0, 0.0, 0, False)
        if isinstance(handcrafted, Mapping):
            feature_vector = np.asarray(
                [float(handcrafted.get(name, 0.0)) for name in self.feature_names],
                dtype=np.float32,
            )
        else:
            feature_vector = np.asarray(list(handcrafted), dtype=np.float32)
        if feature_vector.size != len(self.feature_names):
            LOGGER.warning(
                "Neural feature size mismatch: expected %d, received %d.",
                len(self.feature_names), feature_vector.size,
            )
            return NeuralPrediction(0.5, 0.0, 0.0, 0, False)

        ally_tokens = self._tokens(allies)
        enemy_tokens = self._tokens(enemies)
        probabilities = np.asarray(
            [
                self._one_model_probability(index, ally_tokens, enemy_tokens, feature_vector)
                for index in range(self.model_count)
            ],
            dtype=np.float64,
        )
        mean = float(probabilities.mean())
        std = float(probabilities.std())
        agreement = max(0.0, 1.0 - std / 0.16)
        decisiveness = min(1.0, abs(mean - 0.5) * 2.0)
        validation_brier = float(self.meta.get("validation_brier", 0.25) or 0.25)
        reliability = max(0.0, min(1.0, 1.0 - validation_brier / 0.25))
        confidence = 100.0 * (
            0.35 * agreement
            + 0.25 * max(0.0, min(1.0, data_coverage))
            + 0.20 * max(0.0, min(1.0, role_confidence))
            + 0.12 * reliability
            + 0.08 * decisiveness
        )
        return NeuralPrediction(mean, confidence, std, len(probabilities), True)

    def embedding(self, champion_id: int) -> np.ndarray | None:
        if not self.available:
            return None
        index = self.champion_to_index.get(int(champion_id), 0)
        if index <= 0:
            return None
        vectors = [self.arrays[f"m{i}_champion_embedding"][index] for i in range(self.model_count)]
        vector = np.mean(vectors, axis=0).astype(np.float64)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def nearest_champions(
        self,
        champion_id: int,
        candidate_ids: Iterable[int],
        *,
        limit: int = 8,
    ) -> list[tuple[int, float]]:
        target = self.embedding(champion_id)
        if target is None:
            return []
        output: list[tuple[int, float]] = []
        for candidate_id in candidate_ids:
            if int(candidate_id) == int(champion_id):
                continue
            vector = self.embedding(int(candidate_id))
            if vector is None:
                continue
            output.append((int(candidate_id), float(np.dot(target, vector))))
        output.sort(key=lambda item: item[1], reverse=True)
        return output[: max(1, int(limit))]
