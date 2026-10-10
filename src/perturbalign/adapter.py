"""Module 4: Single-Cell Ecosystem Adapter。

annotate_anndata(adata, drug_col=...) cleanly writes alignment results back:
- obs: one column per field (status / drug_name / canonical_smiles / inchikey / dose / is_control / ...)
- uns: audit log (version / summary statistics / unmatched / ambiguous records)

Zero hard dependencies: only requires the object to possess .obs (DataFrame-like) and .uns (dict);
both real anndata.AnnData and lightweight test stubs are directly supported.

Data defaults to online API queries (align online=True) + local cache, independent of user input shape.
"""
from __future__ import annotations

import abc
import re
from typing import Any

from . import __version__
from .align import align, to_dataframe
from .preprocess import PreprocessConfig, DEFAULT_CONFIG
from .resources import VOCABULARY_VERSION, load_perturbation_protocols
from .schema import (
    AlignResult,
    GeneticEntity,
    DeliveryVehicleEntity,
    MODALITY_BIOLOGIC,
    MODALITY_CONTROL,
    MODALITY_ENVIRONMENTAL,
    MODALITY_GENETIC,
    MODALITY_SMALL_MOLECULE,
    MODALITY_DELIVERY,
    STATUS_CONTROL,
    STATUS_UNMATCHED,
    STATUS_EXACT,
)

# Output fields written back to obs / table (prefix + field_name)
_OBS_COLUMNS = [
    "status", "is_control", "modality", "regime_type", "drug_name", "canonical_smiles", "inchikey",
    "pubchem_cid", "targets", "moa", "confidence", "source",
    "parsed_dose", "parsed_unit", "parsed_duration", "parsed_duration_unit",
    "genetic_target", "genetic_action", "parsed_name", "n_candidates",
    "combo_names", "combo_smiles", "combo_doses", "combo_units", "combo_targets",
]

_AUDIT_STATUS = ("unmatched", "ambiguous", "control", "combo", "empty")


def annotate_anndata(adata, drug_col: str = "perturbation",
                     prefix: str = "perturbalign_",
                     audit: bool = True,
                     inplace: bool = True,
                     **align_kwargs):
    """Align obs[drug_col] and write results back to adata.

    Args:
      adata: Any object containing .obs (DataFrame-like) and .uns (dict), such as anndata.AnnData or a stub.
      drug_col: Perturbation label column name, default "perturbation".
      prefix: Column prefix for fields written back to obs, default "perturbalign_" (avoids polluting original columns).
      audit: Whether to write audit log to uns["perturbalign"] (version, statistics, unmatched, ambiguous details).
      inplace: If True, modifies and returns adata directly; if False, returns (new obs DataFrame, audit dict).
      **align_kwargs: Forwarded to perturbalign.align (fuzzy/online/enrich_* etc.).

    Returns:
      inplace=True  -> adata (updated in-place)
      inplace=False -> (obs copy, audit_dict) (convenient for previewing before commit)
    """
    obs = adata.obs
    if drug_col not in obs.columns:
        raise KeyError(
            f"adata.obs lacks perturbation column {drug_col!r}; available columns: {list(obs.columns)}"
        )
    mapping = align(obs[drug_col], **align_kwargs)
    df = to_dataframe(mapping)

    out_cols = {}
    for col in _OBS_COLUMNS:
        out_cols[prefix + col] = df[col].tolist()

    audit_dict = None
    if audit:
        audit_dict = _build_audit(mapping, drug_col, prefix, align_kwargs)

    if not inplace:
        new_obs = obs.copy()
        for k, v in out_cols.items():
            new_obs[k] = v
        return new_obs, audit_dict

    for k, v in out_cols.items():
        obs[k] = v
    if audit:
        adata.uns["perturbalign"] = audit_dict
    return adata


def annotate_dataframe(
    df,
    pert_col: str = "perturbation",
    dose_col: str | None = None,
    unit_col: str | None = None,
    time_col: str | None = None,
    time_unit_col: str | None = None,
    prefix: str = "perturbalign_",
    standardize_doses: bool = False,
    target_dose_unit: str | None = None,
    mw_col: str | None = None,
    audit: bool = True,
    inplace: bool = False,
    return_audit: bool = False,
    **align_kwargs,
):
    """Align perturbation columns in tabular data (pandas/polars DataFrame), fusing with existing structured columns.

    Args:
      df: pandas DataFrame (or DataFrame-like object supporting column access).
      pert_col: Column name containing perturbation labels, default 'perturbation'.
      dose_col: Optional column name containing existing numeric doses (takes priority over parsed text values).
      unit_col: Optional column name containing existing dose units.
      time_col: Optional column name containing existing duration values.
      time_unit_col: Optional column name containing existing duration units.
      prefix: Column prefix for output columns, default 'perturbalign_'.
      standardize_doses: Whether to automatically normalize doses/concentrations by physical dimension (calls dose.standardize_dose).
      target_dose_unit: Target unit for normalization (e.g. 'µM', 'µg/ml', 'mg/kg'; None selects dimensional default).
      mw_col: Molecular weight column name (used for automated cross-dimensional conversion between mass and molar concentrations).
      audit: Whether to record audit log (version, distributions, unaligned details, saved to df.attrs['perturbalign']).
      inplace: Whether to modify df in-place and return it (default False, returns a new copy).
      return_audit: Whether to explicitly return a (df, audit_dict) tuple (default False, returns df directly).
      **align_kwargs: Additional parameters forwarded to perturbalign.align().

    Returns:
      DataFrame, or (DataFrame, audit_dict) when return_audit=True.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("annotate_dataframe requires pandas, please run: pip install pandas") from exc

    if pert_col not in df.columns:
        raise KeyError(f"DataFrame lacks perturbation column {pert_col!r}; available columns: {list(df.columns)}")

    mapping = align(df[pert_col], **align_kwargs)

    # Intelligent fusion of structured independent columns
    has_custom_dose = any([dose_col, unit_col, time_col, time_unit_col, standardize_doses])
    if has_custom_dose:
        for i, res in enumerate(mapping):
            if dose_col and dose_col in df.columns:
                v = df[dose_col].iloc[i]
                if pd.notna(v):
                    try:
                        res.parsed_dose = float(v)
                    except (ValueError, TypeError):
                        pass
            if unit_col and unit_col in df.columns:
                v = df[unit_col].iloc[i]
                if pd.notna(v) and str(v).strip():
                    res.parsed_unit = str(v).strip()
            if time_col and time_col in df.columns:
                v = df[time_col].iloc[i]
                if pd.notna(v):
                    try:
                        res.parsed_duration = float(v)
                    except (ValueError, TypeError):
                        pass
            if time_unit_col and time_unit_col in df.columns:
                v = df[time_unit_col].iloc[i]
                if pd.notna(v) and str(v).strip():
                    res.parsed_duration_unit = str(v).strip()
            if standardize_doses and res.parsed_dose is not None and res.parsed_unit is not None:
                from .dose import standardize_dose
                mw_val = None
                if mw_col and mw_col in df.columns:
                    mw_raw = df[mw_col].iloc[i]
                    if pd.notna(mw_raw):
                        try:
                            mw_val = float(mw_raw)
                        except (ValueError, TypeError):
                            mw_val = None
                try:
                    s_dose, s_unit = standardize_dose(
                        res.parsed_dose, res.parsed_unit,
                        target_unit=target_dose_unit, mw=mw_val,
                    )
                    res.parsed_dose = s_dose
                    res.parsed_unit = s_unit
                except ValueError:
                    pass

    df_aligned = to_dataframe(mapping)
    out_df = df if inplace else df.copy()

    for col in _OBS_COLUMNS:
        out_df[prefix + col] = df_aligned[col].values

    audit_dict = None
    if audit:
        audit_dict = _build_audit(mapping, pert_col, prefix, align_kwargs)
        try:
            out_df.attrs["perturbalign"] = audit_dict
        except Exception:
            pass

    if return_audit:
        return out_df, audit_dict
    return out_df


def explode_combos(
    df,
    prefix: str = "perturbalign_",
    status_col: str | None = None,
    names_col: str | None = None,
    smiles_col: str | None = None,
    doses_col: str | None = None,
    units_col: str | None = None,
):
    """Explode combinations in tabular data by component (explodes single combination row into multiple single-agent rows).

    Args:
      df: Input pandas DataFrame (typically output from annotate_dataframe).
      prefix: Alignment column prefix (default 'perturbalign_').
      status_col: Status column name (default f'{prefix}status').
      names_col: Combination drug names column name (default f'{prefix}combo_names').
      smiles_col: Combination SMILES column name (default f'{prefix}combo_smiles').
      doses_col: Combination doses column name (default f'{prefix}combo_doses').
      units_col: Combination units column name (default f'{prefix}combo_units').

    Returns:
      Exploded new DataFrame with added fields:
      - 'combo_parent_index': Original row index
      - 'component_index': Component index (0, 1, ...)
      - 'component_name': Standardized drug name for this component
      - 'component_smiles': Canonical SMILES for this component
      - 'component_dose': Dose value for this component
      - 'component_unit': Dose unit for this component
      - 'is_combo_component': Whether row originates from an exploded combination (True / False)
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("explode_combos requires pandas, please run: pip install pandas") from exc

    s_col = status_col or (prefix + "status")
    n_col = names_col or (prefix + "combo_names")
    sm_col = smiles_col or (prefix + "combo_smiles")
    d_col = doses_col or (prefix + "combo_doses")
    u_col = units_col or (prefix + "combo_units")

    drug_name_col = prefix + "drug_name"
    parsed_name_col = prefix + "parsed_name"
    smiles_single_col = prefix + "canonical_smiles"
    dose_single_col = prefix + "parsed_dose"
    unit_single_col = prefix + "parsed_unit"

    exploded_rows = []

    for idx, row in df.iterrows():
        is_combo = (s_col in row and row[s_col] == "combo" and n_col in row and pd.notna(row[n_col]))
        if is_combo:
            names = [x.strip() for x in str(row[n_col]).split(";")]

            def _parse_dose_elem(val: str | None) -> float | None:
                if val is None:
                    return None
                val = val.strip()
                if not val or val.lower() in ("none", "nan", "null"):
                    return None
                try:
                    return float(val)
                except ValueError:
                    return None

            def _parse_str_elem(val: str | None) -> str | None:
                if val is None:
                    return None
                val = val.strip()
                if not val or val.lower() in ("none", "nan", "null"):
                    return None
                return val

            smiles_list = [_parse_str_elem(x) for x in str(row[sm_col]).split(";")] if (sm_col in row and pd.notna(row[sm_col])) else []
            doses_list = [_parse_dose_elem(x) for x in str(row[d_col]).split(",")] if (d_col in row and pd.notna(row[d_col])) else []
            units_list = [_parse_str_elem(x) for x in str(row[u_col]).split(",")] if (u_col in row and pd.notna(row[u_col])) else []

            for comp_idx, name in enumerate(names):
                new_row = row.to_dict()
                new_row["combo_parent_index"] = idx
                new_row["component_index"] = comp_idx
                new_row["component_name"] = name
                new_row["component_smiles"] = smiles_list[comp_idx] if comp_idx < len(smiles_list) else None
                new_row["component_dose"] = doses_list[comp_idx] if comp_idx < len(doses_list) else None
                new_row["component_unit"] = units_list[comp_idx] if comp_idx < len(units_list) else None
                new_row["is_combo_component"] = True
                exploded_rows.append(new_row)
        else:
            new_row = row.to_dict()
            new_row["combo_parent_index"] = idx
            new_row["component_index"] = 0
            new_row["component_name"] = row.get(drug_name_col) or row.get(parsed_name_col)
            new_row["component_smiles"] = row.get(smiles_single_col)
            new_row["component_dose"] = row.get(dose_single_col)
            new_row["component_unit"] = row.get(unit_single_col)
            new_row["is_combo_component"] = False
            exploded_rows.append(new_row)

    return pd.DataFrame(exploded_rows)


def _build_audit(mapping, drug_col: str, prefix: str, align_kwargs: dict) -> dict:
    stats = {s: 0 for s in _AUDIT_STATUS}
    unmatched: list[str] = []
    ambiguous: list[dict] = []
    for r in mapping:
        if r.status in stats:
            stats[r.status] += 1
        if r.status == "unmatched":
            unmatched.append(r.input)
        elif r.status == "ambiguous":
            ambiguous.append({
                "input": r.input,
                "candidates": [c.entity.canonical_name for c in r.candidates],
            })
    return {
        "version": __version__,
        "drug_col": drug_col,
        "column_prefix": prefix,
        "vocabulary_version": VOCABULARY_VERSION,
        "align_kwargs": {k: v for k, v in align_kwargs.items()},
        "n_total": len(mapping),
        "stats": stats,
        "unmatched": unmatched,
        "ambiguous": ambiguous,
    }


def classify_protocol(protocol_key: str, config: PreprocessConfig = DEFAULT_CONFIG) -> dict | None:
    """Query public perturbation experimental protocol metadata standards (LINCS CMap pert_type / scPerturb etc.).

    Returns:
      dict containing {"standard_source", "canonical_modality", "default_action", "description"},
      or None if unindexed.
    """
    k = protocol_key.strip().lower()
    return config.perturbation_protocols.get(k)


class BaseProtocolAdapter(abc.ABC):
    """Abstract base class for public dataset perturbation protocol adapters (SPI)."""

    @property
    @abc.abstractmethod
    def protocol_key(self) -> str:
        """Unique protocol identifier key (e.g. 'lincs', 'scperturb', 'prism')."""
        ...

    @abc.abstractmethod
    def adapt(self, record: Any, config: PreprocessConfig = DEFAULT_CONFIG, **kwargs) -> AlignResult:
        """Convert raw record into standardized AlignResult."""
        ...


class LINCSAdapter(BaseProtocolAdapter):
    """LINCS L1000 / Broad CMap protocol adapter."""

    @property
    def protocol_key(self) -> str:
        return "lincs"

    def adapt(self, record: Any, config: PreprocessConfig = DEFAULT_CONFIG, **kwargs) -> AlignResult:
        if isinstance(record, dict):
            pert_id = record.get("pert_id", "")
            pert_type = record.get("pert_type", "")
            pert_iname = record.get("pert_iname")
            pert_dose = record.get("pert_dose")
            pert_dose_unit = record.get("pert_dose_unit")
            pert_time = record.get("pert_time")
            pert_time_unit = record.get("pert_time_unit")
        else:
            pert_id = str(record)
            pert_type = kwargs.pop("pert_type", "")
            pert_iname = kwargs.pop("pert_iname", None)
            pert_dose = kwargs.pop("pert_dose", None)
            pert_dose_unit = kwargs.pop("pert_dose_unit", None)
            pert_time = kwargs.pop("pert_time", None)
            pert_time_unit = kwargs.pop("pert_time_unit", None)

        p_info = classify_protocol(pert_type, config)
        modality = p_info["canonical_modality"] if p_info else MODALITY_SMALL_MOLECULE
        action = p_info.get("default_action") if p_info else None

        query_name = (pert_iname or pert_id).strip()

        # 1) Control group identification (ctl_vehicle, ctl_vector, ctl_untrt)
        if modality == MODALITY_CONTROL:
            return AlignResult(
                input=pert_id,
                status=STATUS_CONTROL,
                is_control=True,
                modality=MODALITY_CONTROL,
                parsed_name=query_name,
                parsed_dose=pert_dose,
                parsed_unit=pert_dose_unit,
                parsed_duration=pert_time,
                parsed_duration_unit=pert_time_unit,
                note=f"LINCS control ({pert_type}: {action})",
            )

        # 2) Genetic perturbation (shRNA: trt_sh / cDNA overexpression: trt_oe / CRISPR: trt_x)
        if modality == MODALITY_GENETIC:
            target = query_name
            act = action or "perturbation"
            return AlignResult(
                input=pert_id,
                status=STATUS_UNMATCHED,
                is_control=False,
                modality=MODALITY_GENETIC,
                parsed_name=query_name,
                parsed_dose=pert_dose,
                parsed_unit=pert_dose_unit,
                parsed_duration=pert_time,
                parsed_duration_unit=pert_time_unit,
                genetic_target=target,
                genetic_action=act,
                note=f"LINCS {pert_type} targeting {target}",
            )

        # 3) Biological ligand (trt_lig)
        if modality == MODALITY_BIOLOGIC:
            res_list = align([query_name], config=config, **kwargs)
            res = res_list[0]
            res.input = pert_id
            if pert_dose is not None:
                res.parsed_dose = pert_dose
                res.parsed_unit = pert_dose_unit
            if pert_time is not None:
                res.parsed_duration = pert_time
                res.parsed_duration_unit = pert_time_unit
            return res

        # 4) Default small molecule compound (trt_cp)
        search_target = pert_iname if pert_iname else pert_id
        res_list = align([search_target], config=config, **kwargs)
        res = res_list[0]
        res.input = pert_id
        if pert_dose is not None:
            res.parsed_dose = pert_dose
            res.parsed_unit = pert_dose_unit
        if pert_time is not None:
            res.parsed_duration = pert_time
            res.parsed_duration_unit = pert_time_unit
        return res


class ScPerturbAdapter(BaseProtocolAdapter):
    """scPerturb single-cell perturbation protocol adapter."""

    @property
    def protocol_key(self) -> str:
        return "scperturb"

    def adapt(self, record: Any, config: PreprocessConfig = DEFAULT_CONFIG, **kwargs) -> AlignResult:
        if isinstance(record, dict):
            row = record
        else:
            row = dict(record)
        perturbation_col = kwargs.pop("perturbation_col", "perturbation")
        perturbation_type_col = kwargs.pop("perturbation_type_col", "perturbation_type")

        raw_pert = str(row.get(perturbation_col, "")).strip()
        raw_type = str(row.get(perturbation_type_col, "")).strip()

        p_info = classify_protocol(raw_type, config) if raw_type else None

        if p_info and p_info["canonical_modality"] == MODALITY_GENETIC:
            from .preprocess import detect_control
            if detect_control(raw_pert, config):
                return AlignResult(
                    input=raw_pert,
                    status=STATUS_CONTROL,
                    is_control=True,
                    modality=MODALITY_CONTROL,
                    parsed_name=raw_pert,
                    note="scPerturb control",
                )
            from .preprocess import detect_genetic_perturbation
            gen = detect_genetic_perturbation(raw_pert, config)
            if gen:
                target, action = gen
            else:
                target, action = raw_pert, p_info.get("default_action", "perturbation")
            return AlignResult(
                input=raw_pert,
                status=STATUS_UNMATCHED,
                is_control=False,
                modality=MODALITY_GENETIC,
                parsed_name=raw_pert,
                genetic_target=target,
                genetic_action=action,
                note=f"scPerturb genetic perturbation ({target})",
            )

        return align([raw_pert], config=config, **kwargs)[0]


class PRISMAdapter(BaseProtocolAdapter):
    """DepMap PRISM compound identifier protocol adapter."""

    @property
    def protocol_key(self) -> str:
        return "prism"

    def adapt(self, record: Any, config: PreprocessConfig = DEFAULT_CONFIG, **kwargs) -> AlignResult:
        composite_or_id = str(record)
        dose = kwargs.pop("dose", None)
        screen_id = kwargs.pop("screen_id", None)
        default_unit = kwargs.pop("default_unit", "µM")

        input_str = composite_or_id.strip()
        parsed_comp = input_str
        parsed_dose = dose
        parsed_screen = screen_id

        if "::" in input_str:
            parts = [p.strip() for p in input_str.split("::")]
            if len(parts) >= 1 and parts[0]:
                parsed_comp = parts[0]
            if len(parts) >= 2 and parsed_dose is None:
                try:
                    parsed_dose = float(parts[1])
                except ValueError:
                    if parsed_screen is None:
                        parsed_screen = parts[1]
            if len(parts) >= 3 and parsed_screen is None:
                parsed_screen = parts[2]

        from .preprocess import detect_control
        if detect_control(parsed_comp, config):
            note_str = "PRISM control"
            if parsed_screen:
                note_str += f" ({parsed_screen})"
            return AlignResult(
                input=input_str,
                status=STATUS_CONTROL,
                is_control=True,
                modality=MODALITY_CONTROL,
                parsed_name=parsed_comp,
                parsed_dose=parsed_dose,
                parsed_unit=default_unit if parsed_dose is not None else None,
                note=note_str,
            )

        res_list = align([parsed_comp], config=config, **kwargs)
        res = res_list[0]
        res.input = input_str
        if parsed_dose is not None:
            res.parsed_dose = parsed_dose
            res.parsed_unit = default_unit
        if parsed_screen:
            p_info = classify_protocol(parsed_screen, config)
            if p_info:
                note_prefix = f"PRISM screen: {parsed_screen} ({p_info['description']})"
            else:
                note_prefix = f"PRISM screen: {parsed_screen}"
            res.note = f"{note_prefix}; {res.note}" if res.note else note_prefix
        return res


class JUMPCPAdapter(BaseProtocolAdapter):
    """JUMP-Cell Painting image phenotype high-throughput screening protocol adapter."""

    @property
    def protocol_key(self) -> str:
        return "jump_cp"

    def adapt(self, record: Any, config: PreprocessConfig = DEFAULT_CONFIG, **kwargs) -> AlignResult:
        if isinstance(record, dict):
            row = record
        else:
            row = dict(record)

        jcp_id = str(row.get("Metadata_JCP2022") or row.get("JCP2022") or "").strip()
        inchikey = str(row.get("Metadata_InChIKey") or row.get("InChIKey") or "").strip()
        pert_type = str(row.get("Metadata_pert_type") or row.get("pert_type") or "compound").strip().lower()
        broad_sample = str(row.get("Metadata_broad_sample") or row.get("broad_sample") or "").strip()
        pert_name = str(row.get("Metadata_pert_name") or row.get("pert_name") or "").strip()
        gene_name = str(row.get("Metadata_gene_name") or row.get("gene_name") or row.get("Metadata_target") or "").strip()
        dose_val = row.get("Metadata_dose") or row.get("Metadata_mmoles_per_liter") or kwargs.get("dose")
        unit_val = kwargs.get("default_unit", "µM")

        # 1) Control and vehicle treatments
        if pert_type in ("negcon", "control", "jump_cp_negcon", "poscon", "jump_cp_poscon"):
            is_pos = "poscon" in pert_type
            return AlignResult(
                input=jcp_id or broad_sample or pert_type,
                status=STATUS_CONTROL,
                is_control=True,
                modality=MODALITY_CONTROL,
                parsed_name="JUMP-CP positive control" if is_pos else (pert_name or "DMSO vehicle"),
                note=f"JUMP-CP {'positive' if is_pos else 'negative'} control",
            )

        # 2) CRISPR / ORF genetic screening
        if pert_type in ("crispr", "jump_cp_crispr", "orf", "jump_cp_orf"):
            act = "KO" if "crispr" in pert_type else "OE"
            target = gene_name or pert_name or jcp_id
            return AlignResult(
                input=jcp_id or target,
                status=STATUS_UNMATCHED,
                is_control=False,
                modality=MODALITY_GENETIC,
                parsed_name=target,
                genetic_target=target,
                genetic_action=act,
                note=f"JUMP-CP genetic {act} targeting {target}",
            )

        # 3) Small molecule compound screening (InChIKey prioritized -> drug name -> JCP ID)
        query = inchikey if inchikey else (pert_name if pert_name else (broad_sample if broad_sample else jcp_id))
        res_list = align([query], config=config, **kwargs)
        res = res_list[0]
        res.input = jcp_id or query
        if dose_val is not None:
            try:
                res.parsed_dose = float(dose_val)
                res.parsed_unit = unit_val
            except (ValueError, TypeError):
                pass
        note_p = f"JUMP-CP compound ({jcp_id})" if jcp_id else "JUMP-CP compound"
        res.note = f"{note_p}; {res.note}" if res.note else note_p
        return res


class ProtocolRegistry:
    """Registry for dataset protocol adapters supporting pluggable and dynamic extensions."""

    def __init__(self) -> None:
        self._adapters: dict[str, BaseProtocolAdapter] = {}
        self.reset()

    def register_adapter(self, adapter: BaseProtocolAdapter) -> None:
        """Register or update protocol adapter."""
        self._adapters[adapter.protocol_key.lower()] = adapter

    def unregister_adapter(self, protocol_key: str) -> None:
        """Unregister protocol adapter."""
        self._adapters.pop(protocol_key.lower(), None)

    def get_adapter(self, protocol_key: str) -> BaseProtocolAdapter | None:
        """Retrieve specified protocol adapter."""
        return self._adapters.get(protocol_key.lower())

    def list_protocols(self) -> list[str]:
        """Return list of all registered protocol identifier keys."""
        return sorted(self._adapters.keys())

    def adapt_protocol(
        self,
        protocol_key: str,
        record: Any,
        config: PreprocessConfig = DEFAULT_CONFIG,
        **kwargs,
    ) -> AlignResult:
        """Uniformly invoke corresponding protocol adapter to convert record."""
        adapter = self.get_adapter(protocol_key)
        if not adapter:
            raise KeyError(f"Unregistered protocol adapter: {protocol_key!r}, available protocols: {self.list_protocols()}")
        return adapter.adapt(record, config=config, **kwargs)

    def reset(self) -> None:
        """Reset to default adapter suite (lincs, scperturb, prism, jump_cp)."""
        self._adapters = {
            "lincs": LINCSAdapter(),
            "scperturb": ScPerturbAdapter(),
            "prism": PRISMAdapter(),
            "jump_cp": JUMPCPAdapter(),
        }


default_protocol_registry = ProtocolRegistry()


def adapt_protocol(
    protocol_key: str,
    record: Any,
    config: PreprocessConfig = DEFAULT_CONFIG,
    registry: ProtocolRegistry | None = None,
    **kwargs,
) -> AlignResult:
    """Convert raw record to standardized AlignResult using specified protocol adapter."""
    reg = registry or default_protocol_registry
    return reg.adapt_protocol(protocol_key, record, config=config, **kwargs)


def from_lincs(
    pert_id: str,
    pert_type: str,
    pert_iname: str | None = None,
    pert_dose: float | None = None,
    pert_dose_unit: str | None = None,
    pert_time: float | None = None,
    pert_time_unit: str | None = None,
    config: PreprocessConfig = DEFAULT_CONFIG,
    **align_kwargs,
) -> AlignResult:
    """Parse LINCS L1000 / Broad CMap standard metadata record into standardized AlignResult."""
    return default_protocol_registry.adapt_protocol(
        "lincs",
        {
            "pert_id": pert_id,
            "pert_type": pert_type,
            "pert_iname": pert_iname,
            "pert_dose": pert_dose,
            "pert_dose_unit": pert_dose_unit,
            "pert_time": pert_time,
            "pert_time_unit": pert_time_unit,
        },
        config=config,
        **align_kwargs,
    )


def from_scperturb(
    row: dict,
    perturbation_col: str = "perturbation",
    perturbation_type_col: str = "perturbation_type",
    config: PreprocessConfig = DEFAULT_CONFIG,
    **align_kwargs,
) -> AlignResult:
    """Parse single observation record from scPerturb single-cell perturbation dataset."""
    return default_protocol_registry.adapt_protocol(
        "scperturb",
        row,
        perturbation_col=perturbation_col,
        perturbation_type_col=perturbation_type_col,
        config=config,
        **align_kwargs,
    )


def from_prism(
    composite_or_id: str,
    dose: float | None = None,
    screen_id: str | None = None,
    default_unit: str = "µM",
    config: PreprocessConfig = DEFAULT_CONFIG,
    **align_kwargs,
) -> AlignResult:
    """Parse DepMap PRISM compound perturbation identifier (e.g. 'BRD-A00077618-236-07-6::2.5::HTS')."""
    return default_protocol_registry.adapt_protocol(
        "prism",
        composite_or_id,
        dose=dose,
        screen_id=screen_id,
        default_unit=default_unit,
        config=config,
        **align_kwargs,
    )


def from_jump_cp(
    record: dict | Any,
    config: PreprocessConfig = DEFAULT_CONFIG,
    **align_kwargs,
) -> AlignResult:
    """Parse JUMP-Cell Painting image phenotype screening record (supports JCP2022 / InChIKey / pert_type)."""
    return default_protocol_registry.adapt_protocol(
        "jump_cp",
        record,
        config=config,
        **align_kwargs,
    )


def annotate_mudata(
    mdata: Any,
    modal_cols: dict[str, str] | None = None,
    prefix: str = "perturbalign_",
    audit: bool = True,
    inplace: bool = True,
    **align_kwargs,
):
    """Align multi-omics / multi-modal MuData object (or object possessing .mod attribute) per modality and write back.

    Args:
      mdata: mudata.MuData or object possessing .mod modal dictionary and .uns dictionary.
      modal_cols: {modal_name: perturbation_col}, e.g. {'rna': 'perturbation', 'prot': 'antibody_target'}.
        If None, automatically inspects common perturbation column names in each modality obs.
      prefix: Field prefix written back to obs, default 'perturbalign_'.
      audit: Whether to record audit log to mdata.uns['perturbalign'].
      inplace: Whether to modify mdata in-place.
      **align_kwargs: Forwarded to perturbalign.align.

    Returns:
      Aligned mdata object.
    """
    if not hasattr(mdata, "mod") or not isinstance(getattr(mdata, "mod"), dict):
        raise TypeError("mdata object must have a .mod dictionary attribute (conforming to mudata.MuData specification)")

    mod_dict = mdata.mod if inplace else {k: v.copy() for k, v in mdata.mod.items()}
    candidate_cols = ["perturbation", "drug", "gene", "condition", "treatment"]

    multimodal_audits = {}

    for mod_name, mod_obj in mod_dict.items():
        if not hasattr(mod_obj, "obs"):
            continue
        p_col = None
        if modal_cols and mod_name in modal_cols:
            p_col = modal_cols[mod_name]
        else:
            for c in candidate_cols:
                if c in mod_obj.obs.columns:
                    p_col = c
                    break
        if not p_col:
            continue

        annotate_anndata(
            mod_obj,
            drug_col=p_col,
            prefix=prefix,
            audit=audit,
            inplace=True,
            **align_kwargs,
        )
        if audit and hasattr(mod_obj, "uns") and "perturbalign" in mod_obj.uns:
            multimodal_audits[mod_name] = mod_obj.uns["perturbalign"]

    if audit and hasattr(mdata, "uns"):
        mdata.uns["perturbalign"] = {
            "version": __version__,
            "modalities": multimodal_audits,
        }

    return mdata


def lookup_sgrna(
    sequence: str,
    config: PreprocessConfig = DEFAULT_CONFIG,
) -> GeneticEntity | None:
    """Reverse-lookup target gene and reference library from 19-21bp CRISPR sgRNA protospacer sequence.

    Args:
      sequence: 19-21bp single-stranded sgRNA sequence (e.g. 'CCCGGACGATATTGAACAAT').
      config: PreprocessConfig instance, containing human Brunello/GeCKO reference tables.

    Returns:
      GeneticEntity(gene_symbol=..., guide_id=..., modality='genetic') on match, else None.
    """
    seq_clean = sequence.strip().upper().replace("U", "T")
    if not (18 <= len(seq_clean) <= 23) or not re.match(r"^[ACGT]+$", seq_clean):
        return None

    ref_map = config.sgrna_reference
    hit = ref_map.get(seq_clean)
    if hit:
        target = hit["gene_symbol"]
        is_control = target.lower() in ("nt_control", "non-targeting", "nt", "scramble") or target.lower() in config.genetic_control_targets
        act = "control" if is_control else "KO"
        return GeneticEntity(
            input_string=sequence,
            canonical_name=target,
            gene_symbol=target,
            guide_id=seq_clean,
            action=act,
            confidence=1.0,
            match_path=["sgrna:sequence_lookup"],
            source=f"sgrna_reference:{hit.get('source', 'unknown')}",
            extra_metadata=hit,
        )
    return None


def to_frictionless(results: list[AlignResult] | AlignResult) -> dict:
    """Export alignment results to Open Perturbation Exchange Format (OPEF) adhering to Frictionless Data Package standards."""
    res_list = [results] if isinstance(results, AlignResult) else list(results)
    fields = [
        {"name": "input", "type": "string", "description": "Raw perturbation input label"},
        {"name": "status", "type": "string", "description": "PerturbAlign alignment status"},
        {"name": "is_control", "type": "boolean", "description": "Whether perturbation is a control/vehicle"},
        {"name": "modality", "type": "string", "description": "Canonical perturbation modality"},
        {"name": "regime_type", "type": "string", "description": "single, simultaneous, sequential, or pulse_chase"},
        {"name": "canonical_name", "type": "string", "description": "Standardized canonical entity name"},
        {"name": "canonical_smiles", "type": "string", "description": "Standardized canonical SMILES"},
        {"name": "inchikey", "type": "string", "description": "Standard 27-char InChIKey"},
        {"name": "pubchem_cid", "type": "integer", "description": "PubChem Compound ID"},
        {"name": "parsed_dose", "type": "number", "description": "Extracted numerical dose"},
        {"name": "parsed_unit", "type": "string", "description": "Standardized dose unit"},
        {"name": "parsed_duration", "type": "number", "description": "Extracted numerical duration"},
        {"name": "parsed_duration_unit", "type": "string", "description": "Standardized duration unit"},
        {"name": "targets", "type": "array", "description": "Biological target genes"},
    ]
    data = []
    for r in res_list:
        ent = r.entity
        data.append({
            "input": r.input,
            "status": str(r.status),
            "is_control": r.is_control,
            "modality": str(r.modality),
            "regime_type": r.regime_type,
            "canonical_name": ent.canonical_name if ent else (r.parsed_name or r.input),
            "canonical_smiles": getattr(ent, "canonical_smiles", None),
            "inchikey": getattr(ent, "inchikey", None),
            "pubchem_cid": getattr(ent, "pubchem_cid", None),
            "parsed_dose": r.parsed_dose,
            "parsed_unit": r.parsed_unit,
            "parsed_duration": r.parsed_duration,
            "parsed_duration_unit": r.parsed_duration_unit,
            "targets": getattr(ent, "targets", []) if ent else ([r.genetic_target] if r.genetic_target else []),
        })
    return {
        "name": "open_perturbation_exchange_dataset",
        "profile": "tabular-data-resource",
        "perturbalign_version": __version__,
        "schema": {"fields": fields},
        "data": data,
    }


def to_jsonld(results: list[AlignResult] | AlignResult) -> dict:
    """Export alignment results to semantic web JSON-LD format adhering to BioSchemas / Schema.org specifications."""
    res_list = [results] if isinstance(results, AlignResult) else list(results)
    graph = []
    for i, r in enumerate(res_list):
        ent = r.entity
        c_name = ent.canonical_name if ent else (r.parsed_name or r.input)
        node = {
            "@id": f"urn:perturbalign:perturbation:{i+1}",
            "name": c_name,
            "alternateName": r.input,
            "perturbalign:status": str(r.status),
            "perturbalign:isControl": r.is_control,
            "perturbalign:regimeType": r.regime_type,
        }
        if r.parsed_dose is not None:
            node["perturbalign:dose"] = {
                "@type": "QuantitativeValue",
                "value": r.parsed_dose,
                "unitText": r.parsed_unit,
            }
        if r.parsed_duration is not None:
            node["perturbalign:duration"] = {
                "@type": "QuantitativeValue",
                "value": r.parsed_duration,
                "unitText": r.parsed_duration_unit,
            }

        if r.modality == MODALITY_SMALL_MOLECULE:
            node["@type"] = ["ChemicalSubstance", "bioschemas:ChemicalSubstance"]
            if getattr(ent, "canonical_smiles", None):
                node["smiles"] = ent.canonical_smiles
            if getattr(ent, "inchikey", None):
                node["inChIKey"] = ent.inchikey
            if getattr(ent, "pubchem_cid", None):
                node["identifier"] = f"https://pubchem.ncbi.nlm.nih.gov/compound/{ent.pubchem_cid}"
        elif r.modality == MODALITY_GENETIC:
            node["@type"] = ["Gene", "bioschemas:Gene"]
            if r.genetic_target:
                node["targetGene"] = r.genetic_target
            if r.genetic_action:
                node["geneticAction"] = r.genetic_action
        elif r.modality == MODALITY_DELIVERY:
            node["@type"] = ["MedicalDevice", "perturbalign:DeliveryVehicle"]
            if getattr(ent, "carrier_type", None):
                node["perturbalign:carrierType"] = ent.carrier_type
            if getattr(ent, "cargo", None):
                node["perturbalign:cargo"] = ent.cargo
        else:
            node["@type"] = "BioChemEntity"

        graph.append(node)

    return {
        "@context": {
            "@vocab": "https://schema.org/",
            "bioschemas": "https://bioschemas.org/",
            "perturbalign": "https://github.com/perturbalign/ontology#",
        },
        "@graph": graph,
    }
