"""     all      Test    (Candidate Databases Expansion Test Suite).

   10  all    data  ：
【gene   】
1. Open Targets Platform (get_opentargets_info):    、    、 all  
2. JASPAR / UniBind (get_jaspar_motifs):              
3. Human Protein Atlas (get_hpa_expression):      、tissue    
4. ClinVar & COSMIC (get_clinvar_variants):       、    mutation 5. ENCODE cCREs (get_encode_ccres):    /         
【small molecule   】
6. GDSC (get_gdsc_sensitivity):            mutation 7. STITCH (get_stitch_interactions): compound -   interaction network  target 8. BindingDB (get_bindingdb_affinity):        (Kd/Ki/IC50)     
9. DrugBank (get_drugbank_annotations):    MoA、CYP    、   
10. openFDA (get_fda_adverse_events):    not           
【      】
11. query_gene & query_compound 11-in-1 all    
12. schema.py        Verify """
import pytest
import perturbalign


def test_opentargets_info():
    tp53_info = perturbalign.get_opentargets_info("TP53", online=False)
    assert tp53_info is not None
    assert tp53_info["symbol"] == "TP53"
    assert "tractability" in tp53_info
    assert "top_diseases" in tp53_info
    assert len(tp53_info["top_diseases"]) > 0

    egfr_info = perturbalign.get_opentargets_info("EGFR", online=False)
    assert egfr_info["is_druggable"] is True
    assert "small_molecule" in egfr_info["tractability"]


def test_jaspar_motifs():
    # Verify TF
    tp53_tf = perturbalign.get_jaspar_motifs("TP53", online=False)
    assert tp53_tf["is_transcription_factor"] is True
    assert tp53_tf["matrix_id"] == "MA0106.3"
    assert tp53_tf["consensus"] == "RRRCWWGYYY"

    myc_tf = perturbalign.get_jaspar_motifs("MYC", online=False)
    assert myc_tf["is_transcription_factor"] is True
    assert myc_tf["consensus"] == "CACGTG"

    # Verify TF
    egfr_tf = perturbalign.get_jaspar_motifs("EGFR", online=False)
    assert egfr_tf["is_transcription_factor"] is False
    assert egfr_tf["matrix_id"] is None


def test_hpa_expression():
    tp53_hpa = perturbalign.get_hpa_expression("TP53", online=False)
    assert tp53_hpa["symbol"] == "TP53"
    assert "Nucleoplasm" in tp53_hpa["subcellular_location"]
    assert "top_tissues" in tp53_hpa
    assert len(tp53_hpa["top_tissues"]) > 0

    egfr_hpa = perturbalign.get_hpa_expression("EGFR", online=False)
    assert any("membrane" in loc.lower() for loc in egfr_hpa["subcellular_location"])


def test_clinvar_variants():
    tp53_clin = perturbalign.get_clinvar_variants("TP53", online=False)
    assert tp53_clin["symbol"] == "TP53"
    assert tp53_clin["pathogenic_count"] > 1000
    assert any(m["mutation"] == "R175H" for m in tp53_clin["top_pathogenic_mutations"])

    kras_clin = perturbalign.get_clinvar_variants("KRAS", online=False)
    assert any("G12" in m["mutation"] for m in kras_clin["top_pathogenic_mutations"])


def test_encode_ccres():
    tp53_enc = perturbalign.get_encode_ccres("TP53", online=False)
    assert tp53_enc["symbol"] == "TP53"
    assert tp53_enc["promoter_ccres"] >= 1
    assert tp53_enc["total_ccres"] > 10
    assert tp53_enc["regulatory_density"] in {"High", "Very High", "Super-Enhancer Driven"}


def test_gdsc_sensitivity():
    dox_gdsc = perturbalign.get_gdsc_sensitivity("Doxorubicin", online=False)
    assert dox_gdsc["compound"] == "Doxorubicin"
    assert dox_gdsc["mean_ic50_uM"] < 1.0
    assert "Leukemia" in dox_gdsc["top_sensitive_cancers"]
    assert any("TP53" in b for b in dox_gdsc["resistant_biomarkers"])

    tram_gdsc = perturbalign.get_gdsc_sensitivity("Trametinib", online=False)
    assert tram_gdsc["mean_ic50_uM"] < 0.1
    assert any("Melanoma" in c for c in tram_gdsc["top_sensitive_cancers"])


def test_stitch_interactions():
    tram_stitch = perturbalign.get_stitch_interactions("Trametinib", online=False)
    assert tram_stitch["compound"] == "Trametinib"
    assert "MAP2K1" in tram_stitch["direct_targets"]
    assert any(i["target_symbol"] == "MAP2K1" and i["score"] > 900 for i in tram_stitch["interactors"])

    dox_stitch = perturbalign.get_stitch_interactions("Doxorubicin", online=False)
    assert "TOP2A" in dox_stitch["direct_targets"]


def test_bindingdb_affinity():
    tram_bind = perturbalign.get_bindingdb_affinity("Trametinib", online=False)
    assert tram_bind["primary_target"] == "MAP2K1"
    assert tram_bind["selectivity_ratio"] > 1000.0
    assert "Low" in tram_bind["off_target_risk"]

    dox_bind = perturbalign.get_bindingdb_affinity("Doxorubicin", online=False)
    assert dox_bind["primary_target"] == "TOP2A"
    assert "High" in dox_bind["off_target_risk"]


def test_drugbank_annotations():
    dox_db = perturbalign.get_drugbank_annotations("Doxorubicin", online=False)
    assert dox_db["drugbank_id"] == "DB00997"
    assert "topoisomerase ii" in dox_db["moa"].lower()
    assert "CYP3A4" in dox_db["cyp_enzymes"]
    assert any("ABCB1" in t for t in dox_db["transporters"])

    tram_db = perturbalign.get_drugbank_annotations("Trametinib", online=False)
    assert tram_db["drugbank_id"] == "DB08911"
    assert "MAP2K1" in tram_db["targets"]


def test_fda_adverse_events():
    dox_fda = perturbalign.get_fda_adverse_events("Doxorubicin", online=False)
    assert dox_fda["black_box_warning"] is True
    assert dox_fda["cardiotoxicity_warning"] is True
    assert any("Cardio" in e["term"] for e in dox_fda["top_adverse_events"])

    tram_fda = perturbalign.get_fda_adverse_events("Trametinib", online=False)
    assert any("Rash" in e["term"] for e in tram_fda["top_adverse_events"])


def test_query_gene_full_facade():
    res = perturbalign.query_gene("TP53", online=False)
    assert res["gene"] == "TP53"
    # Verify 11 all return
    assert "ncbi" in res
    assert "esm2_embedding" in res
    assert "string_network" in res
    assert "depmap_dependency" in res
    assert "reactome_pathways" in res
    assert "gene_ontology" in res
    assert "opentargets" in res
    assert "jaspar" in res
    assert "hpa" in res
    assert "clinvar" in res
    assert "encode" in res
    assert res["esm2_embedding"] is None or len(res["esm2_embedding"]) == 320


def test_query_compound_full_facade():
    res = perturbalign.query_compound("Doxorubicin", online=False)
    assert res["compound"] == "Doxorubicin"
    # Verify 11 all return
    assert "pubchem" in res
    assert "chemberta_embedding" in res
    assert "l1000_profile" in res
    assert "prism_profile" in res
    assert "jump_profile" in res
    assert "chembl_clinical" in res
    assert "gdsc_sensitivity" in res
    assert "stitch_interactions" in res
    assert "bindingdb_affinity" in res
    assert "drugbank_annotations" in res
    assert "fda_adverse_events" in res
    assert res["chemberta_embedding"] is None or len(res["chemberta_embedding"]) == 384


def test_schema_entities():
    d = perturbalign.DrugEntity(
        input_string="Doxorubicin 10nM",
        canonical_name="Doxorubicin",
        gdsc_sensitivity={"mean_ic50": 0.12},
        stitch_interactions=[{"target": "TOP2A"}],
        bindingdb_affinity={"primary": "TOP2A"},
        drugbank_annotations={"id": "DB00997"},
        fda_adverse_events={"black_box": True},
    )
    assert d.modality == perturbalign.MODALITY_SMALL_MOLECULE
    assert d.gdsc_sensitivity["mean_ic50"] == 0.12
    assert d.drugbank_annotations["id"] == "DB00997"

    g = perturbalign.GeneticEntity(
        input_string="TP53 KO",
        gene_symbol="TP53",
        action="KO",
        opentargets={"is_druggable": False},
        jaspar={"is_tf": True},
        hpa={"location": "Nucleus"},
        clinvar={"pathogenic": 1420},
        encode={"density": "High"},
    )
    assert g.modality == perturbalign.MODALITY_GENETIC
    assert g.canonical_name == "TP53"
    assert g.opentargets["is_druggable"] is False
    assert g.clinvar["pathogenic"] == 1420
