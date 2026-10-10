"""Tests for Extended Library Capabilities: Cell Line, Connectivity, Safety, and Metrics.

Test Verify ：
1. Cell Line       (Cellosaurus / DepMap   mutation      )
2. Connectivity        (L1000 / CMap Tau            )
3. Safety           (SafetyScreen44 / hERG / BSEP /     )
4. Metrics perturbation          (scPerturb / Pertpy Energy Distance)
5. MCP Server        (10  all    Agent    to  Verify )
"""
from __future__ import annotations

import json
import numpy as np
import pytest

import perturbalign
from perturbalign import (
    query_cell_line,
    predict_reversal_drugs,
    compute_connectivity_score,
    evaluate_target_safety,
    PerturbAlignMCPServer,
)


def test_cell_line_context():
    # 1. A549
    a549 = query_cell_line("A-549")
    assert a549 is not None
    assert a549["cellosaurus_id"] == "CVCL_0023"
    assert a549["depmap_id"] == "ACH-000681"
    assert "KRAS_G12S" in a549["driver_mutations"]
    assert a549["tp53_status"] == "WT"
    assert a549["ploidy"] == 2.8

    # 2. MCF-7
    mcf7 = query_cell_line("MCF7")
    assert mcf7 is not None
    assert mcf7["cellosaurus_id"] == "CVCL_0031"
    assert "PIK3CA_E545K" in mcf7["driver_mutations"]

    # 3. K562
    k562 = query_cell_line("k562")
    assert k562 is not None
    assert k562["growth_properties"] == "Suspension"
    assert "BCR-ABL1_Fusion" in k562["driver_mutations"]


def test_signature_reversal_and_connectivity():
    # MYC / CCND1 exception ，CDKN1A exception
    up_genes = ["MYC", "CCND1"]
    down_genes = ["CDKN1A"]

    # phenotype
    reversal_candidates = predict_reversal_drugs(up_genes, down_genes, top_k=3)
    assert len(reversal_candidates) >= 1

    top_drug = reversal_candidates[0]
    assert top_drug["tau_score"] < 0  # is
    assert top_drug["drug"] in ["Vorinostat (SAHA)", "Trametinib"]
    assert len(top_drug["reversed_up_genes"]) > 0 or len(top_drug["restored_down_genes"]) > 0

    # Verify
    tau = compute_connectivity_score(
        query_up=["MYC"],
        query_down=["CDKN1A"],
        reference_up=["CDKN1A"],
        reference_down=["MYC"],
    )
    assert tau == -100.0  # all


def test_safety_and_secondary_pharmacology():
    # 1. gene (hERG / KCNH2)
    herg_eval = evaluate_target_safety("KCNH2")
    assert herg_eval["safetyscreen44_member"] is True
    assert herg_eval["severity"] == "CRITICAL"
    assert "Torsades de Pointes" in herg_eval["clinical_risk"]

    # 2. target (BSEP / ABCB11)
    bsep_eval = evaluate_target_safety("ABCB11")
    assert bsep_eval["safetyscreen44_member"] is True
    assert bsep_eval["organ_system"] == "Hepatic"

    # 3. (Doxorubicin)
    dox_eval = evaluate_target_safety("Doxorubicin")
    assert dox_eval["entity_type"] == "Compound"
    assert "cardiotoxicity" in dox_eval["black_box_warning"].lower()





def test_mcp_extended_tools():
    server = PerturbAlignMCPServer()

    # 1. Verify tools/list 10
    list_resp = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = list_resp["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    assert "perturbalign_query_cell_line" in tool_names
    assert "perturbalign_predict_reversal_drugs" in tool_names
    assert "perturbalign_evaluate_safety" in tool_names
    assert len(tools) >= 10

    # 2. perturbalign_query_cell_line
    call_cell = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "perturbalign_query_cell_line", "arguments": {"cell_line": "A549"}},
    }
    resp_cell = server.handle_message(call_cell)
    res_data = json.loads(resp_cell["result"]["content"][0]["text"])
    assert res_data["cellosaurus_id"] == "CVCL_0023"
    assert "KRAS_G12S" in res_data["driver_mutations"]

    # 3. perturbalign_predict_reversal_drugs
    call_rev = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "perturbalign_predict_reversal_drugs", "arguments": {"up_genes": ["MYC"], "down_genes": ["CDKN1A"]}},
    }
    resp_rev = server.handle_message(call_rev)
    rev_data = json.loads(resp_rev["result"]["content"][0]["text"])
    assert len(rev_data) > 0
    assert rev_data[0]["tau_score"] < 0

    # 4. perturbalign_evaluate_safety
    call_safe = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "perturbalign_evaluate_safety", "arguments": {"query": "KCNH2"}},
    }
    resp_safe = server.handle_message(call_safe)
    safe_data = json.loads(resp_safe["result"]["content"][0]["text"])
    assert safe_data["severity"] == "CRITICAL"
