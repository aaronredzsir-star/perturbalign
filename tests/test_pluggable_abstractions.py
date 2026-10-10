import pytest
from typing import Any

import perturbalign
from perturbalign.schema import (
    STATUS_EXACT,
    MODALITY_SMALL_MOLECULE,
    MODALITY_GENETIC,
)
from perturbalign import (
    BaseEmbeddingProvider,
    BaseProfileProvider,
    BaseResolver,
    AlignmentContext,
    AlignResult,
    CellLineEntity,
    default_embedding_registry,
    register_embedding_provider,
    default_profile_registry,
    register_profile_provider,
    get_default_pipeline,
    reset_default_pipeline,
    align,
    embed_perturbation,
    get_profile,
    query_cell_line,
    register_cell_line,
    register_reference_signature,
    predict_reversal_drugs,
    evaluate_target_safety,
    register_safety_target,
    http_get_json,
    http_post_json,
    http_get_text,
)


def test_embedding_registry_pluggability():
    # 1. Verify default providers
    providers = default_embedding_registry.list_providers()
    assert "genetic" in providers
    assert "small_molecule" in providers
    assert providers["genetic"] == "esm2"
    assert providers["small_molecule"] == "chemberta"

    # 2. Register custom small molecule embedder
    class MockUniMolProvider(BaseEmbeddingProvider):
        name = "unimol_mock"

        def embed(self, target: str, model: Any = None, cache: Any = None, **kwargs: Any) -> list[float]:
            # Returns fixed deterministic vector with 0.77 at first element
            return [0.77] * 64

    try:
        register_embedding_provider(MODALITY_SMALL_MOLECULE, MockUniMolProvider())
        assert default_embedding_registry.list_providers()["small_molecule"] == "unimol_mock"

        vec = embed_perturbation("Trametinib 10nM", dimension=64, include_dose=False, as_numpy=False)
        assert len(vec) == 64
        # Deterministically projected and normalized
        assert round(vec[0], 2) > 0.1
    finally:
        default_embedding_registry.reset()
        assert default_embedding_registry.list_providers()["small_molecule"] == "chemberta"


def test_profile_registry_pluggability():
    # 1. Verify default assays
    assays = default_profile_registry.list_assays()
    assert "l1000" in assays
    assert "prism" in assays
    assert "jump" in assays
    assert "depmap" in assays

    # 2. Register custom assay provider
    class MockProteomicsProvider(BaseProfileProvider):
        assay_name = "proteomics"

        def get_profile(self, target: str, cache: Any = None, **kwargs: Any) -> dict[str, Any] | None:
            return {
                "assay": "proteomics",
                "target": target,
                "protein_hits": ["AKT1", "MTOR"],
                "p_value": 0.001,
            }

    try:
        register_profile_provider("proteomics", MockProteomicsProvider(), aliases=["dia_proteomics"])
        assert "proteomics" in default_profile_registry.list_assays()

        # Query via primary name
        res1 = get_profile("Trametinib", assay="proteomics")
        assert res1 is not None
        assert res1["assay"] == "proteomics"
        assert res1["protein_hits"] == ["AKT1", "MTOR"]

        # Query via alias
        res2 = get_profile("Trametinib", assay="dia_proteomics")
        assert res2 is not None
        assert res2["assay"] == "proteomics"
    finally:
        default_profile_registry.reset()
        assert "proteomics" not in default_profile_registry.list_assays()


def test_pipeline_pluggability():
    # 1. Access active pipeline
    pipeline = get_default_pipeline()
    orig_len = len(pipeline)
    assert orig_len >= 10

    # 2. Plug in a proprietary LIMS custom resolver at the top
    class ProprietaryLimsResolver(BaseResolver):
        name = "ProprietaryLimsResolver"

        def resolve(self, ctx: AlignmentContext) -> AlignResult | None:
            if ctx.base == "LIMS-TEST-999":
                return AlignResult(
                    input=str(ctx.raw_input),
                    status=STATUS_EXACT,
                    parsed_name="Compound_LIMS_999",
                    modality=MODALITY_SMALL_MOLECULE,
                    note="Resolved via Proprietary LIMS",
                )
            return None

    try:
        pipeline.add_resolver(ProprietaryLimsResolver(), priority=0)
        assert len(pipeline) == orig_len + 1

        results = align(["LIMS-TEST-999"], online=False)
        assert len(results) == 1
        assert results[0].status == STATUS_EXACT
        assert results[0].parsed_name == "Compound_LIMS_999"
        assert results[0].note == "Resolved via Proprietary LIMS"
    finally:
        reset_default_pipeline()
        assert len(get_default_pipeline()) == orig_len


def test_cell_line_pluggability():
    custom_cell = CellLineEntity(
        name="PATIENT_DERIVED_ORGANOID_X",
        cellosaurus_id="CVCL_CUSTOM_01",
        depmap_id="CUSTOM_ORG_01",
        lineage="Colorectal",
        primary_disease="Colorectal Carcinoma",
        cancer_type="Organoid Model",
        driver_mutations=["APC_trunc", "KRAS_G12D"],
        tp53_status="Mutant",
        kras_status="Mutant (G12D)",
        msi_status="MSI-H",
        ploidy=2.0,
        growth_properties="3D Organoid",
        tissue_origin="Colon",
    )
    register_cell_line(custom_cell)

    retrieved = query_cell_line("PATIENT_DERIVED_ORGANOID_X", online=False)
    assert retrieved is not None
    assert retrieved["cellosaurus_id"] == "CVCL_CUSTOM_01"
    assert "KRAS_G12D" in retrieved["driver_mutations"]
    assert retrieved["growth_properties"] == "3D Organoid"


def test_connectivity_and_safety_pluggability():
    # 1. Plug in custom reference signature
    custom_sig = {
        "drug": "CustomTestDrug",
        "target": "TEST1",
        "moa": "Custom Inhibitor",
        "up_regulated": ["GENE_A", "GENE_B"],
        "down_regulated": ["GENE_C", "GENE_D"],
    }
    register_reference_signature("CustomTestDrug", custom_sig)

    hits = predict_reversal_drugs(up_genes=["GENE_C"], down_genes=["GENE_A"])
    assert any(h["drug"] == "CustomTestDrug" for h in hits)

    # 2. Test passing custom dictionary directly
    custom_dict = {
        "DirectDrug": {
            "target": "DIRECT_TGT",
            "moa": "Direct MoA",
            "up_regulated": ["UP1"],
            "down_regulated": ["DOWN1"],
        }
    }
    hits_direct = predict_reversal_drugs(
        up_genes=["DOWN1"],
        down_genes=["UP1"],
        reference_signatures=custom_dict,
    )
    assert len(hits_direct) == 1
    assert hits_direct[0]["drug"] == "DirectDrug"
    assert hits_direct[0]["tau_score"] < 0

    # 3. Plug in custom safety counter-screen target
    register_safety_target("CUSTOM_TOX_TARGET", {
        "alias": "ToxKinase",
        "organ_system": "Hepatic",
        "risk_type": "Drug-Induced Liver Injury (DILI)",
        "severity": "CRITICAL",
        "monitoring": "ALT/AST, Bilirubin",
    })
    safety_info = evaluate_target_safety("CUSTOM_TOX_TARGET")
    assert safety_info["safetyscreen44_member"] is True
    assert safety_info["severity"] == "CRITICAL"
    assert "Liver Injury" in safety_info["clinical_risk"]


def test_unified_http_helpers_export():
    assert callable(http_get_json)
    assert callable(http_post_json)
    assert callable(http_get_text)


def test_vocabulary_registry_pluggability():
    from perturbalign import (
        default_vocab_registry,
        register_vocabulary,
        get_vocabulary,
        list_vocabularies,
        reset_vocabulary_registry,
        resolve_investigational_code,
        PreprocessConfig,
    )

    # 1. Check default vocabs exist
    vocabs = list_vocabularies()
    assert "control_tokens" in vocabs
    assert "investigational_codes" in vocabs
    assert "vendor_prefixes" in vocabs
    assert len(vocabs) >= 22

    # 2. Register / override investigational codes in-memory
    custom_codes = {
        "CUSTOM-999": {
            "code": "CUSTOM-999",
            "canonical_name": "Antigravib",
            "targets": ["BRAF", "KRAS"],
            "class_name": "Kinase Inhibitor",
            "developer": "InHouse",
            "smiles": "CC1=CC=CC=C1",
        }
    }
    try:
        register_vocabulary("investigational_codes", custom_codes)
        # Directly retrieved from registry
        retrieved = get_vocabulary("investigational_codes")
        assert "custom-999" in retrieved or "CUSTOM-999" in retrieved

        # Propagates dynamically to resolve_investigational_code
        inv = resolve_investigational_code("CUSTOM 999")
        assert inv is not None
        assert inv["canonical_name"] == "Antigravib"
        assert "BRAF" in inv["targets"]

        # 3. Register custom vendor prefix
        register_vocabulary("vendor_prefixes", ["MYVENDOR-", "CHEM-"])
        cfg = PreprocessConfig()
        assert "MYVENDOR-" in cfg.vendor_prefixes

        # 4. Register custom domain table
        register_vocabulary("custom_clinical_phases", {"phase_4": "Post-marketing surveillance"})
        assert get_vocabulary("custom_clinical_phases")["phase_4"] == "Post-marketing surveillance"
        assert "custom_clinical_phases" in list_vocabularies()
    finally:
        reset_vocabulary_registry()
        assert resolve_investigational_code("CUSTOM-999") is None
        assert "custom_clinical_phases" not in list_vocabularies()


def test_epigenetics_pluggability():
    from perturbalign import (
        CisRegulatoryElement,
        resolve_ccres,
        register_ccre,
        reset_epigenetics_registry,
    )

    try:
        # Custom cCRE
        custom_el = CisRegulatoryElement(
            accession="CUSTOM_CCRE_001",
            ccre_type="PLS",
            chromosome="chr1",
            start=1000,
            end=1500,
            target_gene="CUSTOM_GENE",
            tf_motifs=["TF_ALPHA"],
            histone_marks=["H3K4me3"],
        )
        register_ccre("CUSTOM_GENE", custom_el)
        ccres = resolve_ccres("CUSTOM_GENE")
        assert len(ccres) == 1
        assert ccres[0].accession == "CUSTOM_CCRE_001"
    finally:
        reset_epigenetics_registry()
        assert resolve_ccres("CUSTOM_GENE") == []


def test_cascade_and_safety_pluggability():
    from perturbalign import (
        infer_causal_cascade,
        register_target_cascade,
        register_dorothea_regulon,
        reset_cascade_registry,
        evaluate_target_safety,
        register_safety_compound,
        reset_safety_registry,
    )

    try:
        # 1. Plug in custom signaling cascade & regulon
        register_target_cascade("CUSTOM_KINASE", ["CUSTOM_TF"], "Novel Oncogenic Axis")
        register_dorothea_regulon("CUSTOM_TF", ["GENE_A", "GENE_B", "GENE_C"])

        cascade = infer_causal_cascade(["CUSTOM_KINASE"])
        assert "Novel Oncogenic Axis" in cascade["pathway_names"]
        assert "CUSTOM_TF" in cascade["downstream_tfs"]
        assert "GENE_A" in cascade["response_regulon"]
        assert "GENE_B" in cascade["response_regulon"]

        # 2. Plug in custom compound safety liability
        register_safety_compound("CUSTOM_DRUG_X", {
            "safety_tier": "Critical Risk / Experimental",
            "black_box_warning": "Fatal cardiotoxicity and severe bone marrow suppression",
            "primary_organ_risks": ["Cardiovascular", "Hematologic"],
            "critical_off_targets": ["KCNH2", "TOP2A"],
        })

        safety = evaluate_target_safety("CUSTOM_DRUG_X")
        assert safety["entity_type"] == "Compound"
        assert safety["safetyscreen44_flagged"] is True
        assert "Fatal cardiotoxicity" in safety["black_box_warning"]
        assert "Cardiovascular" in safety["primary_organ_risks"]
    finally:
        reset_cascade_registry()
        reset_safety_registry()

