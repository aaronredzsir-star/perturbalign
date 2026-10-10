"""Multimodal perturbation unified embedding service.

Maps arbitrary perturbation inputs (gene knockout, overexpression, mutation, small molecules,
combinations, controls) to a unified continuous vector space (default 512-dimensional dense vector).

Modality-adaptive routing:
  - Genetic perturbations (sgTP53, MYC_OE) -> ESM2 protein language model embedding projection
  - Small molecules (Trametinib 10nM) -> ChemBERTa molecular structure embedding + dose dimension encoding
  - Combinations / double knockouts (sgBCL2L1+sgMCL1) -> Component-wise adaptive semantic fusion
  - Controls (DMSO, sgCtrl) -> Baseline zero vector

Note:
  Continuous deep embeddings require valid pretrained weights or explicit model callable instances.
"""
from __future__ import annotations

import abc
import math
from typing import Any, Sequence

from .cache import SQLiteCache
from .gene import get_esm2_embedding
from .compound import get_chemberta_embedding
from .schema import (
    AlignResult,
    MODALITY_CONTROL,
    MODALITY_GENETIC,
    MODALITY_SMALL_MOLECULE,
    MODALITY_COMBO,
)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def _project_vector(src_vec: list[float], target_dim: int) -> list[float]:
    """Deterministically project and normalize source vector to target dimension."""
    if len(src_vec) == target_dim:
        return src_vec
    src_len = len(src_vec)
    out = [0.0] * target_dim
    for i in range(target_dim):
        src_idx = int((i / target_dim) * src_len)
        out[i] = src_vec[src_idx % src_len]
    norm = math.sqrt(sum(x * x for x in out)) or 1.0
    return [round(x / norm, 6) for x in out]


class BaseEmbeddingProvider(abc.ABC):
    """Abstract base provider for modality-specific perturbation embedding models."""

    name: str = "base"

    @abc.abstractmethod
    def embed(
        self,
        target: str,
        model: Any = None,
        cache: SQLiteCache | None = None,
        **kwargs: Any,
    ) -> list[float]:
        """Produce an embedding vector for given target string."""
        raise NotImplementedError


class ESM2EmbeddingProvider(BaseEmbeddingProvider):
    """Default genetic perturbation embedding provider using ESM2 protein language models."""

    name = "esm2"

    def embed(
        self,
        target: str,
        model: Any = None,
        cache: SQLiteCache | None = None,
        **kwargs: Any,
    ) -> list[float]:
        return get_esm2_embedding(target, model=model, cache=cache)


class ChemBERTaEmbeddingProvider(BaseEmbeddingProvider):
    """Default small molecule chemical embedding provider using ChemBERTa transformer models."""

    name = "chemberta"

    def embed(
        self,
        target: str,
        model: Any = None,
        cache: SQLiteCache | None = None,
        **kwargs: Any,
    ) -> list[float]:
        return get_chemberta_embedding(target, model=model, cache=cache)


class EmbeddingRegistry:
    """Pluggable registry for modality-specific embedding providers."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseEmbeddingProvider] = {}
        self.reset()

    def register(self, modality: str, provider: BaseEmbeddingProvider) -> None:
        """Register or replace an embedding provider for a modality (e.g. 'genetic', 'small_molecule')."""
        self._providers[modality.lower()] = provider

    def unregister(self, modality: str) -> bool:
        """Remove a provider for a modality."""
        return bool(self._providers.pop(modality.lower(), None))

    def get(self, modality: str) -> BaseEmbeddingProvider | None:
        """Get the active provider for a modality."""
        return self._providers.get(modality.lower())

    def list_providers(self) -> dict[str, str]:
        """List registered modalities and provider names."""
        return {mod: p.name for mod, p in self._providers.items()}

    def reset(self) -> None:
        """Reset registry to default built-in providers."""
        self._providers.clear()
        self.register(MODALITY_GENETIC, ESM2EmbeddingProvider())
        self.register(MODALITY_SMALL_MOLECULE, ChemBERTaEmbeddingProvider())


default_embedding_registry = EmbeddingRegistry()
register_embedding_provider = default_embedding_registry.register


def embed_perturbation(
    perturbation: str | AlignResult,
    dimension: int = 512,
    include_dose: bool = True,
    esm2_model: Any = None,
    chemberta_model: Any = None,
    cache: SQLiteCache | None = None,
    as_numpy: bool = False,
) -> Any:
    """Map a single perturbation label or AlignResult to a continuous dense vector.

    Parameters:
      perturbation: Raw perturbation string (e.g. "Trametinib 10nM", "sgTP53", "DMSO") or AlignResult.
      dimension: Target vector space dimension (default 512).
      include_dose: Whether to integrate dose concentration/dimension (small molecules only).
      esm2_model: Pretrained ESM2 model or callable.
      chemberta_model: Pretrained ChemBERTa model or callable.
      cache: SQLiteCache instance.
      as_numpy: Whether to return numpy.ndarray if numpy is installed.

    Returns:
      list[float] or np.ndarray
    """
    from .align import align

    if isinstance(perturbation, AlignResult):
        res = perturbation
    else:
        results = align([str(perturbation)], online=False)
        res = results[0]

    # 1. Controls -> Baseline zero vector
    if res.is_control or res.modality == MODALITY_CONTROL:
        vec = [0.0] * dimension
        if as_numpy and _HAS_NUMPY:
            return np.array(vec, dtype=np.float32)
        return vec

    # 2. Genetic perturbations -> Registered embedding provider projection
    if res.modality == MODALITY_GENETIC:
        gene = res.genetic_target or res.parsed_name or res.input
        provider = default_embedding_registry.get(MODALITY_GENETIC)
        if provider is None:
            raise RuntimeError("No embedding provider registered for modality 'genetic'")
        esm_vec = provider.embed(gene, model=esm2_model, cache=cache)
        vec = _project_vector(esm_vec, dimension)
        # Integrate action modality features (KO vs OE vs mutant)
        action = (res.genetic_action or "").lower()
        if "oe" in action:
            vec[0] = round(vec[0] + 0.5, 6)
        elif "ko" in action:
            vec[0] = round(vec[0] - 0.5, 6)
        if as_numpy and _HAS_NUMPY:
            return np.array(vec, dtype=np.float32)
        return vec

    # 3. Combinations / multi-target knockouts -> Adaptive component fusion
    if res.modality == MODALITY_COMBO and res.candidates:
        sub_vecs = []
        for cand in res.candidates:
            cand_name = cand.canonical_name or cand.name
            sub_v = embed_perturbation(
                cand_name,
                dimension=dimension,
                include_dose=include_dose,
                esm2_model=esm2_model,
                chemberta_model=chemberta_model,
                cache=cache,
            )
            sub_vecs.append(sub_v)
        if sub_vecs:
            # Semantic mean fusion and re-normalization
            fused = [0.0] * dimension
            for sv in sub_vecs:
                for i in range(dimension):
                    fused[i] += sv[i]
            n = len(sub_vecs)
            norm = math.sqrt(sum((x / n) ** 2 for x in fused)) or 1.0
            vec = [round((x / n) / norm, 6) for x in fused]
            if as_numpy and _HAS_NUMPY:
                return np.array(vec, dtype=np.float32)
            return vec

    # 4. Small molecules & extensible modalities -> Provider embedding + physical dose modulation
    provider = default_embedding_registry.get(res.modality) or default_embedding_registry.get(MODALITY_SMALL_MOLECULE)
    if provider is None:
        raise RuntimeError(f"No embedding provider registered for modality {res.modality!r}")
    name = res.parsed_name or res.input
    chem_vec = provider.embed(name, model=chemberta_model, cache=cache)
    vec = _project_vector(chem_vec, dimension)

    if include_dose and res.parsed_dose is not None:
        # Encode log10 dose at the final dimension
        try:
            log_dose = math.log10(max(float(res.parsed_dose), 1e-6))
            vec[-1] = round(math.tanh(log_dose * 0.2), 6)
        except Exception:
            pass

    if as_numpy and _HAS_NUMPY:
        return np.array(vec, dtype=np.float32)
    return vec



def embed_batch(
    perturbations: Sequence[str | AlignResult],
    dimension: int = 512,
    include_dose: bool = True,
    esm2_model: Any = None,
    chemberta_model: Any = None,
    cache: SQLiteCache | None = None,
    as_numpy: bool = True,
) -> Any:
    """Batch compute continuous representation matrix for a list of perturbations.

    Returns:
      If as_numpy=True and numpy is installed, returns (N, dimension) 2D ndarray;
      otherwise returns list[list[float]].
    """
    matrix = [
        embed_perturbation(
            p,
            dimension=dimension,
            include_dose=include_dose,
            esm2_model=esm2_model,
            chemberta_model=chemberta_model,
            cache=cache,
            as_numpy=False,
        )
        for p in perturbations
    ]
    if as_numpy and _HAS_NUMPY:
        return np.array(matrix, dtype=np.float32)
    return matrix


__all__ = [
    "embed_perturbation",
    "embed_batch",
    "BaseEmbeddingProvider",
    "ESM2EmbeddingProvider",
    "ChemBERTaEmbeddingProvider",
    "EmbeddingRegistry",
    "default_embedding_registry",
    "register_embedding_provider",
]

