"""Cell line and host genotype context engine.

Integrates Cellosaurus (CVCL canonical identifiers) and DepMap/CCLE cancer model metadata
to annotate genetic background, driver mutations, lineage, and ploidy of host cell lines.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class CellLineEntity:
    name: str
    cellosaurus_id: str
    depmap_id: str
    lineage: str
    primary_disease: str
    cancer_type: str
    driver_mutations: list[str] = field(default_factory=list)
    tp53_status: str = "WT"
    kras_status: str = "WT"
    msi_status: str = "MSS"
    ploidy: float = 2.0
    growth_properties: str = "Adherent"
    tissue_origin: str = "Unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CELL_LINE_DATABASE: dict[str, CellLineEntity] = {
    "A549": CellLineEntity(
        name="A549",
        cellosaurus_id="CVCL_0023",
        depmap_id="ACH-000681",
        lineage="Lung",
        primary_disease="Non-Small Cell Lung Carcinoma",
        cancer_type="Lung Adenocarcinoma",
        driver_mutations=["KRAS_G12S", "STK11_Q37*", "KEAP1_G333C"],
        tp53_status="WT",
        kras_status="Mutant (G12S)",
        msi_status="MSS",
        ploidy=2.8,
        growth_properties="Adherent",
        tissue_origin="Lung",
    ),
    "MCF7": CellLineEntity(
        name="MCF7",
        cellosaurus_id="CVCL_0031",
        depmap_id="ACH-000019",
        lineage="Breast",
        primary_disease="Breast Carcinoma",
        cancer_type="Invasive Ductal Carcinoma (ER+/PR+)",
        driver_mutations=["PIK3CA_E545K", "GATA3_Splice"],
        tp53_status="WT",
        kras_status="WT",
        msi_status="MSS",
        ploidy=3.0,
        growth_properties="Adherent",
        tissue_origin="Breast",
    ),
    "K562": CellLineEntity(
        name="K562",
        cellosaurus_id="CVCL_0004",
        depmap_id="ACH-000551",
        lineage="Blood",
        primary_disease="Chronic Myelogenous Leukemia (CML)",
        cancer_type="Chronic Myelogenous Leukemia (Blast Phase)",
        driver_mutations=["BCR-ABL1_Fusion", "TP53_Frameshift"],
        tp53_status="Null/Mutant",
        kras_status="WT",
        msi_status="MSS",
        ploidy=3.0,
        growth_properties="Suspension",
        tissue_origin="Bone Marrow",
    ),
    "JURKAT": CellLineEntity(
        name="Jurkat",
        cellosaurus_id="CVCL_0065",
        depmap_id="ACH-000096",
        lineage="Lymphoid",
        primary_disease="T-Cell Acute Lymphoblastic Leukemia",
        cancer_type="T-ALL",
        driver_mutations=["PTEN_Mutant", "TP53_R196*"],
        tp53_status="Mutant",
        kras_status="WT",
        msi_status="MSS",
        ploidy=2.0,
        growth_properties="Suspension",
        tissue_origin="Peripheral Blood",
    ),
    "HELA": CellLineEntity(
        name="HeLa",
        cellosaurus_id="CVCL_0030",
        depmap_id="ACH-001060",
        lineage="Cervix",
        primary_disease="Cervical Adenocarcinoma",
        cancer_type="Cervical Carcinoma (HPV18+)",
        driver_mutations=["HPV18_E6_E7_Integration", "PIK3CA_Mutant"],
        tp53_status="Degraded (HPV18 E6 mediated)",
        kras_status="WT",
        msi_status="MSS",
        ploidy=3.3,
        growth_properties="Adherent",
        tissue_origin="Cervix",
    ),
    "HEPG2": CellLineEntity(
        name="HepG2",
        cellosaurus_id="CVCL_0027",
        depmap_id="ACH-000527",
        lineage="Liver",
        primary_disease="Hepatoblastoma",
        cancer_type="Hepatoblastoma",
        driver_mutations=["CTNNB1_S45del", "NRAS_Q61K"],
        tp53_status="WT",
        kras_status="WT",
        msi_status="MSS",
        ploidy=2.0,
        growth_properties="Adherent",
        tissue_origin="Liver",
    ),
    "PC9": CellLineEntity(
        name="PC9",
        cellosaurus_id="CVCL_B260",
        depmap_id="ACH-000624",
        lineage="Lung",
        primary_disease="Non-Small Cell Lung Carcinoma",
        cancer_type="Lung Adenocarcinoma",
        driver_mutations=["EGFR_E746_A750del", "TP53_R248Q"],
        tp53_status="Mutant",
        kras_status="WT",
        msi_status="MSS",
        ploidy=2.6,
        growth_properties="Adherent",
        tissue_origin="Lung",
    ),
    "HCT116": CellLineEntity(
        name="HCT116",
        cellosaurus_id="CVCL_0291",
        depmap_id="ACH-000971",
        lineage="Bowel",
        primary_disease="Colorectal Carcinoma",
        cancer_type="Colorectal Carcinoma",
        driver_mutations=["KRAS_G13D", "PIK3CA_H1047R", "MLH1_Deficient"],
        tp53_status="WT",
        kras_status="Mutant (G13D)",
        msi_status="MSI-H",
        ploidy=2.0,
        growth_properties="Adherent",
        tissue_origin="Colon",
    ),
    "RPE1": CellLineEntity(
        name="RPE1",
        cellosaurus_id="CVCL_4388",
        depmap_id="Non-Cancer",
        lineage="Eye",
        primary_disease="Normal / Immortalized",
        cancer_type="Non-tumorigenic hTERT RPE-1",
        driver_mutations=["hTERT_Stable_Transfection"],
        tp53_status="WT",
        kras_status="WT",
        msi_status="MSS",
        ploidy=2.0,
        growth_properties="Adherent",
        tissue_origin="Retina",
    ),
    "U2OS": CellLineEntity(
        name="U2OS",
        cellosaurus_id="CVCL_0042",
        depmap_id="ACH-000364",
        lineage="Bone",
        primary_disease="Osteosarcoma",
        cancer_type="Osteosarcoma",
        driver_mutations=["CDKN2A_Homozygous_Deletion"],
        tp53_status="WT",
        kras_status="WT",
        msi_status="MSS",
        ploidy=3.0,
        growth_properties="Adherent",
        tissue_origin="Bone",
    ),
}


def normalize_cell_line_name(name: str) -> str:
    """Normalize cell line name: strip hyphens/spaces and convert to uppercase."""
    cleaned = re.sub(r"[\s\-_/]+", "", name.strip().upper())
    return cleaned


def query_cell_line(name: str, online: bool = False) -> Optional[dict[str, Any]]:
    """Query cell line genetic background, driver mutations, lineage, and cancer model attributes.

    Parameters
    ----------
    name : str
        Cell line name (e.g. 'A549', 'A-549', 'MCF7', 'K562', 'PC9', 'HCT116').
    online : bool
        Whether to attempt online retrieval via the Cellosaurus API.

    Returns
    -------
    dict or None
        Cell line metadata dictionary, or None if not found.
    """
    if not name or not isinstance(name, str):
        return None

    norm = normalize_cell_line_name(name)
    if norm in _CELL_LINE_DATABASE:
        return _CELL_LINE_DATABASE[norm].to_dict()

    # Online fallback (via api.cellosaurus.org)
    if online:
        try:
            import urllib.parse
            from .async_engine import http_get_json

            url = f"https://api.cellosaurus.org/search/cell-line?q={urllib.parse.quote(name)}&fields=id,ac,ox,ca&format=json"
            data = http_get_json(url, timeout=5.0)
            records = data.get("cell-line-list", [])
            if records:
                first = records[0]
                return {
                    "name": first.get("id", name),
                    "cellosaurus_id": first.get("ac", "Unknown"),
                    "depmap_id": "Online_Retrieved",
                    "lineage": "Online_Annotation",
                    "primary_disease": first.get("ca", "Unknown"),
                    "cancer_type": first.get("ca", "Unknown"),
                    "driver_mutations": [],
                    "tp53_status": "Unknown",
                    "kras_status": "Unknown",
                    "msi_status": "Unknown",
                    "ploidy": 2.0,
                    "growth_properties": "Unknown",
                    "tissue_origin": "Unknown",
                }
        except Exception:
            pass

    return None


_DEFAULT_CELL_LINE_KEYS = set(_CELL_LINE_DATABASE.keys())


def register_cell_line(cell_line: CellLineEntity | dict[str, Any]) -> None:
    """Register or customize a host cell line, organoid, or primary model."""
    if isinstance(cell_line, dict):
        ent = CellLineEntity(
            name=cell_line["name"],
            cellosaurus_id=cell_line.get("cellosaurus_id", "Custom"),
            depmap_id=cell_line.get("depmap_id", "Custom"),
            lineage=cell_line.get("lineage", "Custom"),
            primary_disease=cell_line.get("primary_disease", "Custom"),
            cancer_type=cell_line.get("cancer_type", "Custom"),
            driver_mutations=cell_line.get("driver_mutations", []),
            tp53_status=cell_line.get("tp53_status", "WT"),
            kras_status=cell_line.get("kras_status", "WT"),
            msi_status=cell_line.get("msi_status", "MSS"),
            ploidy=cell_line.get("ploidy", 2.0),
            growth_properties=cell_line.get("growth_properties", "Unknown"),
            tissue_origin=cell_line.get("tissue_origin", "Unknown"),
        )
    else:
        ent = cell_line
    norm = normalize_cell_line_name(ent.name)
    _CELL_LINE_DATABASE[norm] = ent


def reset_cell_line_registry() -> None:
    """Reset registered cell lines to default built-in cell lines."""
    for k in list(_CELL_LINE_DATABASE.keys()):
        if k not in _DEFAULT_CELL_LINE_KEYS:
            del _CELL_LINE_DATABASE[k]


__all__ = [
    "CellLineEntity",
    "normalize_cell_line_name",
    "query_cell_line",
    "register_cell_line",
    "reset_cell_line_registry",
]

