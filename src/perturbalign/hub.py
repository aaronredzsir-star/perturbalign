"""Perturbalign library hub and ecosystem interface.

Elevates Perturbalign into an all-in-one library interface for biological perturbation analysis:
1. One-stop query and feature extraction: query_gene / query_compound / get_profile / get_ppi_network
2. Multimodal continuous vector representations: embed_perturbation / embed_batch
3. Single-cell ecosystem featurization: featurize_anndata (injects annotations, profiles, STRING into obsm / uns)
"""
from __future__ import annotations

import abc
from typing import Any, Sequence


from .align import align, to_dataframe
from .cache import SQLiteCache
from .compound import (
    get_chemberta_embedding,
    get_jump_profile,
    get_l1000_profile,
    get_prism_profile,
    get_pubchem_compound,
    query_compound,
)
from .embedding import embed_batch, embed_perturbation
from .gene import (
    get_depmap_dependency,
    get_esm2_embedding,
    get_string_network,
    lookup_ncbi_gene,
    query_gene,
)
from .schema import (
    AlignResult,
    MODALITY_CONTROL,
    MODALITY_GENETIC,
    MODALITY_SMALL_MOLECULE,
)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 1. Unified high-level facade functions & pluggable profile providers
# --------------------------------------------------------------------------
def get_ppi_network(
    genes: str | list[str],
    species: int = 9606,
    score_threshold: int = 400,
    limit: int = 20,
    cache: SQLiteCache | None = None,
) -> dict[str, Any]:
    """Retrieve STRING protein-protein interaction network topology for a gene or gene list."""
    return get_string_network(genes, species=species, score_threshold=score_threshold, limit=limit, cache=cache)


class BaseProfileProvider(abc.ABC):
    """Abstract base class for perturbation assay profile providers."""

    assay_name: str = "base"

    @abc.abstractmethod
    def get_profile(self, target: str, cache: SQLiteCache | None = None, **kwargs: Any) -> dict[str, Any] | None:
        """Extract bioactivity or phenotypic profile for the target."""
        raise NotImplementedError


class L1000ProfileProvider(BaseProfileProvider):
    """LINCS L1000 transcriptomic expression signature profile provider."""

    assay_name = "l1000"

    def get_profile(self, target: str, cache: SQLiteCache | None = None, **kwargs: Any) -> dict[str, Any] | None:
        return get_l1000_profile(target, cache=cache)


class PRISMProfileProvider(BaseProfileProvider):
    """DepMap PRISM cancer cell line viability profile provider."""

    assay_name = "prism"

    def get_profile(self, target: str, cache: SQLiteCache | None = None, **kwargs: Any) -> dict[str, Any] | None:
        return get_prism_profile(target, cache=cache)


class JUMPProfileProvider(BaseProfileProvider):
    """JUMP-Cell Painting morphological feature profile provider."""

    assay_name = "jump"

    def get_profile(self, target: str, cache: SQLiteCache | None = None, **kwargs: Any) -> dict[str, Any] | None:
        return get_jump_profile(target, cache=cache)


class DepMapProfileProvider(BaseProfileProvider):
    """DepMap genome-wide CRISPR dependency profile provider."""

    assay_name = "depmap"

    def get_profile(self, target: str, cache: SQLiteCache | None = None, **kwargs: Any) -> dict[str, Any] | None:
        return get_depmap_dependency(target, cache=cache)


class ProfileRegistry:
    """Pluggable registry for perturbation assay profile providers."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseProfileProvider] = {}
        self._aliases: dict[str, str] = {}
        self.reset()

    def register(
        self,
        assay: str,
        provider: BaseProfileProvider,
        aliases: Sequence[str] | None = None,
    ) -> None:
        """Register or replace an assay profile provider."""
        key = assay.lower()
        self._providers[key] = provider
        if aliases:
            for alias in aliases:
                self._aliases[alias.lower()] = key

    def unregister(self, assay: str) -> bool:
        """Remove a profile provider."""
        key = assay.lower()
        if key in self._providers:
            del self._providers[key]
            self._aliases = {a: tgt for a, tgt in self._aliases.items() if tgt != key}
            return True
        return False

    def get(self, assay: str) -> BaseProfileProvider | None:
        """Retrieve the active provider for an assay name or alias."""
        key = assay.lower()
        actual_key = self._aliases.get(key, key)
        return self._providers.get(actual_key)

    def list_assays(self) -> list[str]:
        """List registered primary assay names."""
        return sorted(self._providers.keys())

    def reset(self) -> None:
        """Reset registry to default built-in providers."""
        self._providers.clear()
        self._aliases.clear()
        self.register("l1000", L1000ProfileProvider())
        self.register("prism", PRISMProfileProvider())
        self.register("jump", JUMPProfileProvider(), aliases=["jump_cp", "cell_painting"])
        self.register("depmap", DepMapProfileProvider())


default_profile_registry = ProfileRegistry()
register_profile_provider = default_profile_registry.register


def get_profile(
    label_or_result: str | AlignResult,
    assay: str = "l1000",
    cache: SQLiteCache | None = None,
) -> dict[str, Any] | None:
    """Extract bioactivity and phenotypic signature profiles for a perturbation.

    Supported default assay types:
      - 'l1000': LINCS L1000 transcriptomic expression signature
      - 'prism': DepMap PRISM cancer cell line viability profile
      - 'jump': JUMP-Cell Painting morphological profile (aliases: 'jump_cp', 'cell_painting')
      - 'depmap': DepMap genome-wide CRISPR dependency profile
      (Custom assays can be plugged in via register_profile_provider).
    """
    if isinstance(label_or_result, AlignResult):
        res = label_or_result
    else:
        results = align([str(label_or_result)], online=False)
        res = results[0]

    assay_norm = assay.strip().lower()
    provider = default_profile_registry.get(assay_norm)
    if provider is None:
        supported = ", ".join(repr(a) for a in default_profile_registry.list_assays())
        raise ValueError(f"Unsupported assay type: {assay!r}; supported: {supported}")

    if provider.assay_name == "depmap":
        target = res.genetic_target or res.parsed_name or res.input
    else:
        target = res.parsed_name or res.input

    return provider.get_profile(target, cache=cache)



# --------------------------------------------------------------------------
# 2. Single-cell AnnData featurization & graph injection (featurize_anndata)
# --------------------------------------------------------------------------
def featurize_anndata(
    adata: Any,
    drug_col: str = "perturbation",
    embed_dim: int = 512,
    add_embeddings: bool = True,
    add_profiles: bool = True,
    build_ppi_network: bool = True,
    prefix: str = "perturbalign_",
    online: bool = False,
    esm2_model: Any = None,
    chemberta_model: Any = None,
    cache: SQLiteCache | None = None,
    inplace: bool = True,
    **align_kwargs,
) -> Any:
    """Automated single-cell AnnData perturbation featurization and knowledge graph injection engine.

    Executes in a single pass:
    1. Metadata normalization, desalting, spatiotemporal parsing -> writes to adata.obs[f"{prefix}*"]
    2. ESM2 protein representation & ChemBERTa molecular embeddings -> writes to adata.obsm["X_pert_embedding"]
    3. L1000 / PRISM / JUMP / DepMap bioactivity fingerprints -> writes to adata.obsm["X_pert_profile"]
    4. Perturbation gene target STRING interaction network graph -> writes to adata.uns["perturbalign_ppi_network"]
    """
    from .adapter import annotate_anndata

    target_adata = adata if inplace else adata.copy()

    # 1. Base metadata parsing and annotation
    annotate_anndata(target_adata, drug_col=drug_col, prefix=prefix, online=online, inplace=True, **align_kwargs)

    obs = target_adata.obs
    unique_labels = list(obs[drug_col].unique())

    # 2. Vector embedding injection (obsm["X_pert_embedding"])
    if add_embeddings:
        label_to_emb: dict[str, list[float]] = {}
        for lbl in unique_labels:
            try:
                label_to_emb[lbl] = embed_perturbation(
                    lbl,
                    dimension=embed_dim,
                    esm2_model=esm2_model,
                    chemberta_model=chemberta_model,
                    cache=cache,
                    as_numpy=False,
                )
            except RuntimeError:
                label_to_emb[lbl] = [0.0] * embed_dim

        emb_matrix = [label_to_emb.get(val, [0.0] * embed_dim) for val in obs[drug_col]]
        if _HAS_NUMPY:
            target_adata.obsm["X_pert_embedding"] = np.array(emb_matrix, dtype=np.float32)
        else:
            target_adata.obsm["X_pert_embedding"] = emb_matrix

    # 3. Bioactivity phenotypic profile injection (obsm["X_pert_profile"])
    if add_profiles:
        profile_dim = 64
        label_to_prof: dict[str, list[float]] = {}
        for lbl in unique_labels:
            status_val = obs.loc[obs[drug_col] == lbl, f"{prefix}modality"].iloc[0]
            if status_val == MODALITY_GENETIC:
                dep = get_depmap_dependency(lbl, online=online, cache=cache)
                raw_vec = dep.get("dependency_vector") if dep else None
                vec = (raw_vec * 2)[:profile_dim] if raw_vec else [0.0] * profile_dim
            else:
                l1k = get_l1000_profile(lbl, online=online, cache=cache)
                raw_vec = l1k.get("landmark_signature_vector") if l1k else None
                vec = raw_vec[:profile_dim] if raw_vec else [0.0] * profile_dim
            if not vec or len(vec) != profile_dim:
                vec = [0.0] * profile_dim
            label_to_prof[lbl] = vec

        prof_matrix = [label_to_prof.get(val, [0.0] * profile_dim) for val in obs[drug_col]]
        if _HAS_NUMPY:
            target_adata.obsm["X_pert_profile"] = np.array(prof_matrix, dtype=np.float32)
        else:
            target_adata.obsm["X_pert_profile"] = prof_matrix

    # 4. STRING network topology graph injection (uns["perturbalign_ppi_network"])
    if build_ppi_network:
        target_genes = []
        if f"{prefix}genetic_target" in obs.columns:
            for g in obs[f"{prefix}genetic_target"].dropna().unique():
                if str(g).strip():
                    target_genes.append(str(g).strip())
        if not target_genes and f"{prefix}targets" in obs.columns:
            for t_list in obs[f"{prefix}targets"].dropna():
                if isinstance(t_list, list):
                    target_genes.extend(t_list)

        if target_genes:
            ppi_data = get_string_network(list(set(target_genes))[:30], online=online, cache=cache)
            target_adata.uns["perturbalign_ppi_network"] = ppi_data

    return target_adata


__all__ = [
    "get_ppi_network",
    "get_profile",
    "featurize_anndata",
    "BaseProfileProvider",
    "L1000ProfileProvider",
    "PRISMProfileProvider",
    "JUMPProfileProvider",
    "DepMapProfileProvider",
    "ProfileRegistry",
    "default_profile_registry",
    "register_profile_provider",
]

