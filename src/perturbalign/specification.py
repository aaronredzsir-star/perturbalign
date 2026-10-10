"""Universal Perturbation Data Specification (UPDS).

Defines a unified cross-modal input data model for single-cell computational biology,
functional genomics, and AI-driven therapeutics.
Overcomes fragmented modalities, unstandardized combination treatments, lost dose dimensions,
and heterogeneous downstream model ingestion formats.

Core Object Hierarchy:
1. PerturbationAtom: Minimal atomic chemical/genetic/biologic/delivery/environmental agent
2. PerturbationUnit: Full perturbation event applied to a single cell or sample (single, combo, sequential)
3. PerturbationDataset: Batch dataset container (seamless conversion with AnnData, MuData, DataFrame, PyTorch)
4. parse_input: Highly forgiving multi-source parsing facade (supports str, dict, list, Series, AnnData)
"""
from __future__ import annotations

import collections
import dataclasses
from dataclasses import dataclass, field, asdict
import json
import math
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, Union

from .schema import Modality, MODALITY_SMALL_MOLECULE, MODALITY_GENETIC, MODALITY_CONTROL, MODALITY_COMBO
from .dose import standardize_dose, to_pconc


# --------------------------------------------------------------------------
# 1. PerturbationAtom: Atomic Perturbation Unit
# --------------------------------------------------------------------------
@dataclass
class PerturbationAtom:
    """Atomic Perturbation Unit representing a single chemical, genetic, or environmental agent.

    Attributes:
      name: Standardized primary name (e.g., 'Doxorubicin', 'TP53')
      modality: Perturbation modality (small_molecule, genetic, biologic, delivery, environmental, control)
      action: Perturbation action mechanism (e.g., 'KO', 'KD', 'OE', 'CRISPRi', 'inhibitor', 'agonist', 'mutant:T790M')
      identifiers: Global cross-database persistent keys (e.g. {'pubchem_cid': 31703, 'inchikey': '...', 'entrez_id': '7157'})
      dose: Scalar dose value (float)
      dose_unit: Physical unit (e.g., 'nM', 'µM', 'mg/kg', 'MOI', 'Gy')
      pconc: Standardized pharmacological negative log molar concentration (-log10[M])
      duration: Exposure duration scalar value (float)
      duration_unit: Duration time unit (e.g., 'h', 'd', 'min')
      embedding: Continuous pre-trained semantic vector (e.g., ESM-2 or ChemBERTa)
      profile: Multi-dimensional activity fingerprints (L1000/PRISM/JUMP/DepMap/GDSC)
      annotations: Domain knowledge base annotations (Reactome, GO, OpenTargets, DrugBank, STITCH, etc.)
      extra: Custom user-defined key-value attributes
    """
    name: str
    modality: str = MODALITY_SMALL_MOLECULE
    action: Optional[str] = None
    identifiers: dict[str, Any] = field(default_factory=dict)
    dose: Optional[float] = None
    dose_unit: Optional[str] = None
    pconc: Optional[float] = None
    duration: Optional[float] = None
    duration_unit: Optional[str] = None
    embedding: Optional[list[float]] = None
    profile: Optional[dict[str, Any]] = None
    annotations: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Automatically compute logarithmic molar concentration scalar (pconc)
        if self.dose is not None and self.dose_unit is not None and self.pconc is None:
            self.pconc = to_pconc(self.dose, self.dose_unit)

    @property
    def is_control(self) -> bool:
        """Check whether atom represents a negative control or vehicle."""
        return self.modality == MODALITY_CONTROL or self.name.upper() in {
            "CONTROL", "CTRL", "DMSO", "PBS", "SALINE", "UNTREATED", "WATER", "VEHICLE"
        }

    def canonical_string(self) -> str:
        """Generate deterministic atomic fingerprint signature."""
        parts = [self.name]
        if self.action:
            parts.append(f"({self.action})")
        if self.dose is not None and self.dose_unit:
            d_str = f"{self.dose:g}{self.dose_unit}"
            if self.duration is not None and self.duration_unit:
                d_str += f"@{self.duration:g}{self.duration_unit}"
            parts.append(f"[{d_str}]")
        return "".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to standard nested dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PerturbationAtom:
        """Deserialize from dictionary."""
        d = dict(data)
        return cls(
            name=str(d.get("name", "Unknown")),
            modality=str(d.get("modality", MODALITY_SMALL_MOLECULE)),
            action=d.get("action"),
            identifiers=dict(d.get("identifiers") or {}),
            dose=float(d["dose"]) if d.get("dose") is not None else None,
            dose_unit=d.get("dose_unit"),
            pconc=float(d["pconc"]) if d.get("pconc") is not None else None,
            duration=float(d["duration"]) if d.get("duration") is not None else None,
            duration_unit=d.get("duration_unit"),
            embedding=d.get("embedding"),
            profile=d.get("profile"),
            annotations=dict(d.get("annotations") or {}),
            extra=dict(d.get("extra") or {}),
        )


# --------------------------------------------------------------------------
# 2. PerturbationUnit: Full Perturbation Event Complex
# --------------------------------------------------------------------------
@dataclass
class PerturbationUnit:
    """Full perturbation event applied to single cells, cell populations, or assay wells.

    Attributes:
      raw_input: Raw input string or origin label
      atoms: List of constituent perturbation atoms
      is_combination: Whether perturbation is a cocktail or multi-agent combination
      is_sequential: Whether perturbation involves sequential or pulse-chase treatment
      regime_order: Treatment sequence order index list
      canonical_signature: Deterministic deduplicated and sorted perturbation signature
      unified_embedding: Cross-modal continuous representation vector (512d)
      qc_flags: Quality control and audit warnings
    """
    raw_input: str
    atoms: list[PerturbationAtom] = field(default_factory=list)
    is_combination: bool = False
    is_sequential: bool = False
    regime_order: list[int] = field(default_factory=list)
    canonical_signature: str = ""
    unified_embedding: Optional[list[float]] = None
    qc_flags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.canonical_signature and self.atoms:
            # Deterministic sorting to generate canonical signature
            sorted_atoms = sorted(self.atoms, key=lambda a: (a.modality, a.name))
            self.canonical_signature = " + ".join(a.canonical_string() for a in sorted_atoms)
        if len(self.atoms) > 1:
            self.is_combination = True

    @property
    def is_control(self) -> bool:
        """Determined as control if all constituent atoms are controls or empty."""
        if not self.atoms:
            return True
        return all(a.is_control for a in self.atoms)

    @property
    def primary_atom(self) -> Optional[PerturbationAtom]:
        """Get the primary perturbation atom (first or non-control atom)."""
        if not self.atoms:
            return None
        non_ctrl = [a for a in self.atoms if not a.is_control]
        return non_ctrl[0] if non_ctrl else self.atoms[0]

    @property
    def names(self) -> list[str]:
        """List of standardized names of constituent components."""
        return [a.name for a in self.atoms]

    @property
    def modalities(self) -> list[str]:
        """List of modalities of constituent components."""
        return [a.modality for a in self.atoms]

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "raw_input": self.raw_input,
            "canonical_signature": self.canonical_signature,
            "is_control": self.is_control,
            "is_combination": self.is_combination,
            "is_sequential": self.is_sequential,
            "atoms": [a.to_dict() for a in self.atoms],
            "unified_embedding": self.unified_embedding,
            "qc_flags": list(self.qc_flags),
        }

    def to_jsonld(self) -> dict[str, Any]:
        """Export as W3C / BioChemEntity compliant JSON-LD semantic graph format."""
        elements = []
        for a in self.atoms:
            elem = {
                "@type": "BioChemEntity",
                "name": a.name,
                "bioChemEntityCategory": a.modality,
                "identifier": a.identifiers,
            }
            if a.dose is not None:
                elem["dosage"] = {"value": a.dose, "unitText": a.dose_unit, "pconc": a.pconc}
            elements.append(elem)
        return {
            "@context": "https://schema.org",
            "@type": "MedicalConditionEvaluation",
            "name": self.canonical_signature,
            "isControl": self.is_control,
            "interventions": elements,
        }

    def validate(self) -> list[str]:
        """Strict data compliance and sanity validation."""
        warnings = []
        if not self.atoms and not self.is_control:
            warnings.append("EMPTY_ATOMS: Perturbation unit contains no atoms and is not an explicit control.")
        for i, a in enumerate(self.atoms):
            if not a.name or a.name.strip() == "":
                warnings.append(f"ATOM_{i}_BLANK_NAME: Perturbation atom has blank name.")
            if a.dose is not None and a.dose < 0:
                warnings.append(f"ATOM_{i}_NEGATIVE_DOSE: Dose value is negative (<0).")
        return warnings


# --------------------------------------------------------------------------
# 3. PerturbationDataset: Universal Batch Dataset Container
# --------------------------------------------------------------------------
class PerturbationDataset:
    """Cross-modal universal single-cell perturbation dataset container.

    Supports:
    - List-like slicing and iterable indexing
    - Zero-copy bidirectional conversion with Pandas DataFrame
    - Bidirectional feature injection and reading with AnnData (.obs, .obsm, .uns)
    - Exporting PyTorch / NumPy tensors for ML training (embeddings, doses, masks)
    """

    def __init__(
        self,
        units: Sequence[PerturbationUnit],
        dataset_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.units: list[PerturbationUnit] = list(units)
        self.dataset_id: str = dataset_id or "perturb_dataset"
        self.metadata: dict[str, Any] = dict(metadata or {})

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, idx: Union[int, slice]) -> Union[PerturbationUnit, PerturbationDataset]:
        if isinstance(idx, slice):
            return PerturbationDataset(self.units[idx], dataset_id=self.dataset_id, metadata=self.metadata)
        return self.units[idx]

    def __iter__(self) -> Iterator[PerturbationUnit]:
        return iter(self.units)

    @property
    def signatures(self) -> list[str]:
        """List of unique signature strings for all perturbation units."""
        return [u.canonical_signature for u in self.units]

    @property
    def modalities(self) -> list[str]:
        """List of primary modalities across all perturbation units."""
        return [u.primary_atom.modality if u.primary_atom else "unknown" for u in self.units]

    @property
    def is_control_mask(self) -> list[bool]:
        """Boolean control mask across all perturbation units."""
        return [u.is_control for u in self.units]

    def to_dataframe(self) -> Any:
        """Export to standard Pandas DataFrame format."""
        import pandas as pd
        rows = []
        for u in self.units:
            prim = u.primary_atom
            rows.append({
                "perturbation": u.canonical_signature or u.raw_input,
                "raw_input": u.raw_input,
                "is_control": u.is_control,
                "is_combination": u.is_combination,
                "primary_name": prim.name if prim else None,
                "primary_modality": prim.modality if prim else None,
                "primary_action": prim.action if prim else None,
                "primary_dose": prim.dose if prim else None,
                "primary_unit": prim.dose_unit if prim else None,
                "primary_pconc": prim.pconc if prim else None,
                "primary_duration": prim.duration if prim else None,
                "n_components": len(u.atoms),
                "component_names": ";".join(u.names),
            })
        return pd.DataFrame(rows)

    def to_tensors(self, format: str = "numpy") -> dict[str, Any]:
        """Export feature tensors for downstream statistical modeling and machine learning.

        Returns:
          {
            'embeddings': (N, D) continuous feature tensor,
            'doses': (N, 1) continuous logarithmic molar concentration,
            'is_control': (N,) control boolean tensor,
            'n_components': (N,) number of components in combination,
          }
        """
        import numpy as np

        n = len(self.units)
        dim = 512
        for u in self.units:
            if u.unified_embedding:
                dim = len(u.unified_embedding)
                break

        emb_mat = np.zeros((n, dim), dtype=np.float32)
        doses = np.zeros((n, 1), dtype=np.float32)
        is_ctrl = np.zeros((n,), dtype=bool)
        n_comps = np.zeros((n,), dtype=np.int32)

        for i, u in enumerate(self.units):
            if u.unified_embedding:
                emb_mat[i] = np.array(u.unified_embedding, dtype=np.float32)
            prim = u.primary_atom
            if prim and prim.pconc is not None:
                doses[i, 0] = prim.pconc
            is_ctrl[i] = u.is_control
            n_comps[i] = len(u.atoms)

        result = {
            "embeddings": emb_mat,
            "doses": doses,
            "is_control": is_ctrl,
            "n_components": n_comps,
        }

        if format.lower() == "torch":
            try:
                import torch
                return {k: torch.from_numpy(v) for k, v in result.items()}
            except ImportError:
                pass

        return result

    def to_anndata(self, adata: Any = None) -> Any:
        """Inject or generate standardized AnnData object."""
        import numpy as np
        df = self.to_dataframe()
        tensors = self.to_tensors(format="numpy")

        if adata is None:
            import pandas as pd
            class AnnDataStub:
                def __init__(self, obs_df):
                    self.obs = obs_df
                    self.obsm = {}
                    self.uns = {}
            adata = AnnDataStub(df)
        else:
            for col in df.columns:
                adata.obs[f"pert_{col}"] = df[col].values

        adata.obsm["X_pert_embedding"] = tensors["embeddings"]
        adata.obsm["X_pert_dose"] = tensors["doses"]
        adata.uns["pert_specification_version"] = "1.0.0"
        adata.uns["pert_dataset_id"] = self.dataset_id
        return adata

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            "dataset_id": self.dataset_id,
            "metadata": self.metadata,
            "units": [u.to_dict() for u in self.units],
        }, indent=indent)

    @classmethod
    def from_dataframe(cls, df: Any, col: str = "perturbation") -> PerturbationDataset:
        """Construct from Pandas DataFrame column."""
        series = df[col] if col in df.columns else df.iloc[:, 0]
        units = [parse_single_input(val) for val in series]
        return cls(units=units)


# --------------------------------------------------------------------------
# 4. parse_input: Multi-Source Input Parsing Facade
# --------------------------------------------------------------------------
def parse_single_input(raw: Any) -> PerturbationUnit:
    """Normalize any single input (string, dict, AlignResult) into standard PerturbationUnit."""
    if isinstance(raw, PerturbationUnit):
        return raw

    # 1. Dictionary input
    if isinstance(raw, Mapping):
        name = str(raw.get("name") or raw.get("symbol") or raw.get("compound") or "Unknown")
        modality = str(raw.get("modality") or MODALITY_SMALL_MOLECULE)
        action = raw.get("action")
        dose_val = float(raw["dose"]) if raw.get("dose") is not None else None
        dose_unit = raw.get("dose_unit") or raw.get("unit")
        atom = PerturbationAtom(
            name=name,
            modality=modality,
            action=action,
            dose=dose_val,
            dose_unit=dose_unit,
            identifiers=dict(raw.get("identifiers") or {}),
            embedding=raw.get("embedding"),
            profile=raw.get("profile"),
        )
        return PerturbationUnit(raw_input=str(raw), atoms=[atom])

    # 2. String input -> invoke perturbalign alignment
    s = str(raw).strip()
    from .align import align
    from .embedding import embed_perturbation

    align_out = align(s, online=False)
    res = align_out[0] if isinstance(align_out, list) and len(align_out) > 0 else align_out
    atoms: list[PerturbationAtom] = []

    if res.parts:
        for p in res.parts:
            atom_name = p.parsed_name or (p.entity.canonical_name if p.entity else p.input)
            atom_mod = p.modality or MODALITY_SMALL_MOLECULE
            atom_act = getattr(p, "genetic_action", None)
            ident: dict[str, Any] = {}
            if p.entity:
                if getattr(p.entity, "pubchem_cid", None):
                    ident["pubchem_cid"] = p.entity.pubchem_cid
                if getattr(p.entity, "inchikey", None):
                    ident["inchikey"] = p.entity.inchikey
                if getattr(p.entity, "entrez_id", None):
                    ident["entrez_id"] = p.entity.entrez_id
            atoms.append(PerturbationAtom(
                name=atom_name,
                modality=atom_mod,
                action=atom_act,
                dose=p.parsed_dose,
                dose_unit=p.parsed_unit,
                duration=p.parsed_duration,
                duration_unit=p.parsed_duration_unit,
                identifiers=ident,
            ))
    else:
        atom_name = res.parsed_name or (res.entity.canonical_name if res.entity else s)
        atom_mod = res.modality or MODALITY_SMALL_MOLECULE
        atom_act = getattr(res, "genetic_action", None)
        ident = {}
        if res.entity:
            if getattr(res.entity, "pubchem_cid", None):
                ident["pubchem_cid"] = res.entity.pubchem_cid
            if getattr(res.entity, "inchikey", None):
                ident["inchikey"] = res.entity.inchikey
            if getattr(res.entity, "entrez_id", None):
                ident["entrez_id"] = res.entity.entrez_id
        atoms.append(PerturbationAtom(
            name=atom_name,
            modality=atom_mod,
            action=atom_act,
            dose=res.parsed_dose,
            dose_unit=res.parsed_unit,
            duration=res.parsed_duration,
            duration_unit=res.parsed_duration_unit,
            identifiers=ident,
        ))

    try:
        emb = embed_perturbation(s)
    except Exception:
        emb = None

    return PerturbationUnit(
        raw_input=s,
        atoms=atoms,
        is_combination=res.status == "combo" or len(atoms) > 1,
        is_sequential=res.regime_type is not None,
        unified_embedding=emb,
    )


def parse_input(
    raw: Any,
    col: Optional[str] = None,
) -> Union[PerturbationUnit, PerturbationDataset]:
    """Universal multi-source perturbation input parser.

    Accepts:
    - Single string or dictionary -> returns PerturbationUnit
    - List, set, Pandas Series -> returns PerturbationDataset
    - Pandas DataFrame or AnnData -> extracts target column and returns PerturbationDataset
    """
    if isinstance(raw, (str, Mapping)):
        return parse_single_input(raw)

    if hasattr(raw, "columns"):
        col_to_use = col or ("perturbation" if "perturbation" in raw.columns else raw.columns[0])
        return PerturbationDataset.from_dataframe(raw, col=col_to_use)

    if hasattr(raw, "obs"):
        col_to_use = col or ("perturbation" if "perturbation" in raw.obs.columns else raw.obs.columns[0])
        return PerturbationDataset.from_dataframe(raw.obs, col=col_to_use)

    if isinstance(raw, (Sequence, Iterable)):
        units = [parse_single_input(item) for item in raw]
        return PerturbationDataset(units=units)

    return parse_single_input(str(raw))


# --------------------------------------------------------------------------
# 5. PerturbationContext: Biological Context Model
# --------------------------------------------------------------------------
@dataclass
class PerturbationContext:
    """Holistic Single-Cell Perturbation Context Model (The Perturbation Trinity Context).

    Models single-cell experiments across three interconnected pillars:
    1. agent: Perturbation agent (PerturbationUnit - atomic entities, chemical structures, dose, duration, embedding)
    2. host_cell_line: Host cell line background (CellLineEntity - mutations, ploidy, TP53/KRAS status)
    3. microenvironment: Environmental conditions (e.g., oxygen_pct=1.0, temperature_c=37, media='RPMI-1640')
    """
    agent: PerturbationUnit
    host_cell_line: Optional[Any] = None
    microenvironment: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        host_dict = None
        if self.host_cell_line is not None:
            host_dict = self.host_cell_line.to_dict() if hasattr(self.host_cell_line, "to_dict") else dict(self.host_cell_line)
        return {
            "agent": self.agent.to_dict(),
            "host_cell_line": host_dict,
            "microenvironment": self.microenvironment,
            "metadata": self.metadata,
        }

    def summarize_system(self) -> str:
        """Generate a system-level holistic summary string."""
        pert_desc = self.agent.canonical_signature
        host_name = getattr(self.host_cell_line, "name", "General/In Vitro") if self.host_cell_line else "General/In Vitro"
        host_disease = getattr(self.host_cell_line, "cancer_type", "") if self.host_cell_line else ""
        host_desc = f"{host_name} ({host_disease})" if host_disease else host_name
        env_items = [f"{k}={v}" for k, v in self.microenvironment.items()]
        env_desc = f" [{', '.join(env_items)}]" if env_items else ""
        return f"[System Context] {pert_desc} @ {host_desc}{env_desc}"


def build_context(
    perturbation: Union[str, PerturbationUnit, Mapping[str, Any]],
    cell_line: Optional[Any] = None,
    microenvironment: Optional[dict[str, Any]] = None,
    **extra_env: Any,
) -> PerturbationContext:
    """Convenience builder for PerturbationContext."""
    if isinstance(perturbation, PerturbationUnit):
        unit = perturbation
    else:
        unit = parse_single_input(perturbation)

    host = None
    if cell_line is not None:
        if hasattr(cell_line, "to_dict"):
            host = cell_line
        elif isinstance(cell_line, str):
            from .cell_line import query_cell_line, CellLineEntity
            info = query_cell_line(cell_line)
            if info:
                host = CellLineEntity(**info)

    env = dict(microenvironment or {})
    env.update(extra_env)

    return PerturbationContext(
        agent=unit,
        host_cell_line=host,
        microenvironment=env,
    )
