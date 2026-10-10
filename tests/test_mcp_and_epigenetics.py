"""MCP protocol   、epigenetic       knowledge graph     Test   。

  ：
1. KGRegistry    ：BaseKGProvider、    、 data        
2. Epigenetics   ：cCREs parse 、        、CRISPRi/a    causal   
3. MCP (Model Context Protocol) Server：initialize, tools/list, tools/call  to    
"""
import json
import pytest

import perturbalign
from perturbalign import (
    BaseKGProvider,
    KGRegistry,
    default_kg_registry,
    register_kg_provider,
    HeterogeneousKnowledgeGraph,
    resolve_ccres,
    predict_epigenetic_impact,
    query_epigenetics,
    PerturbAlignMCPServer,
)


def test_kg_registry_and_custom_provider():
    registry = KGRegistry()

    @registry.register("custom_demo")
    class DemoProvider(BaseKGProvider):
        name = "DemoProvider"
        version = "1.0.0"
        description = "Test custom provider for community extensibility"
        node_types = ("Gene",)
        edge_types = ("custom_activates",)

        def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
            kg.add_node("Gene:DEMO_SRC", "DEMO_SRC", "Gene")
            kg.add_node("Gene:DEMO_TGT", "DEMO_TGT", "Gene")
            kg.add_edge("Gene:DEMO_SRC", "Gene:DEMO_TGT", "custom_activates", weight=0.95, effect_sign=1)
            return 1

    # Verify data
    providers = registry.list_providers()
    assert "custom_demo" in providers
    meta = providers["custom_demo"]
    assert meta["name"] == "DemoProvider"
    assert "custom_activates" in meta["edge_types"]

    # to
    test_kg = HeterogeneousKnowledgeGraph(load_curated=False)
    added = registry.load_provider("custom_demo", test_kg)
    assert added == 1
    paths = test_kg.search_pathways("DEMO_SRC", "DEMO_TGT")
    assert len(paths) == 1
    assert paths[0].overall_effect_sign == 1


def test_epigenetics_module():
    # 1. cCRE parse
    ccres_tp53 = resolve_ccres("TP53")
    assert len(ccres_tp53) >= 2
    types = {c.ccre_type for c in ccres_tp53}
    assert "PLS" in types
    assert "pELS" in types
    assert "H3K4me3" in ccres_tp53[0].histone_marks

    # 2. epigenetic perturbation causal
    krab = predict_epigenetic_impact("dCas9-KRAB:MYC")
    assert krab["effect_sign"] == -1
    assert "closed" in krab["chromatin_state"]

    vpr = predict_epigenetic_impact("dCas9-VPR:CDKN1A")
    assert vpr["effect_sign"] == 1
    assert "open" in vpr["chromatin_state"]

    hdaci = predict_epigenetic_impact("Belinostat 1uM")
    assert hdaci["mechanism"] == "HDAC Inhibitor"
    assert hdaci["effect_sign"] == 1

    # 3.
    epi_myc = query_epigenetics("MYC")
    assert epi_myc["target"] == "MYC"
    assert epi_myc["total_ccres"] >= 3
    assert len(epi_myc["associated_tf_motifs"]) > 0


def test_mcp_server_protocol():
    server = PerturbAlignMCPServer()

    # 1. protocol initialize
    init_req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    init_resp = server.handle_message(init_req)
    assert init_resp["result"]["serverInfo"]["name"] == "perturbalign-mcp"
    assert "protocolVersion" in init_resp["result"]

    # 2. column tools/list
    list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    list_resp = server.handle_message(list_req)
    tools = list_resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "perturbalign_align" in tool_names
    assert "perturbalign_query_gene" in tool_names
    assert "perturbalign_predict_pathway_impact" in tool_names
    assert "perturbalign_query_epigenetics" in tool_names

    # 3. tools/call: perturbalign_align
    call_align = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "perturbalign_align", "arguments": {"perturbations": ["Trametinib 10nM 24h"]}},
    }
    resp_align = server.handle_message(call_align)
    text_res = resp_align["result"]["content"][0]["text"]
    parsed_json = json.loads(text_res)
    assert len(parsed_json) == 1
    assert parsed_json[0]["parsed_name"] == "Trametinib"
    assert parsed_json[0]["parsed_dose"] == 10.0

    # 4. tools/call: perturbalign_predict_pathway_impact
    call_impact = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "perturbalign_predict_pathway_impact", "arguments": {"perturbation": "Trametinib", "pathway": "Pathway:MAPK_Cascade"}},
    }
    resp_impact = server.handle_message(call_impact)
    impact_json = json.loads(resp_impact["result"]["content"][0]["text"])
    assert "MAPK/ERK Signaling Pathway" in impact_json["pathways"]
    assert impact_json["pathways"]["MAPK/ERK Signaling Pathway"]["direction"] == "inhibited"

    # 5. tools/call: perturbalign_query_epigenetics
    call_epi = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "perturbalign_query_epigenetics", "arguments": {"target": "TP53"}},
    }
    resp_epi = server.handle_message(call_epi)
    epi_json = json.loads(resp_epi["result"]["content"][0]["text"])
    assert epi_json["target"] == "TP53"
    assert epi_json["promoter_ccres"] >= 1


def test_extended_providers():
    providers = default_kg_registry.list_providers()
    # Verify 4 Provider
    assert "collectri" in providers
    assert "liana" in providers
    assert "drugcomb" in providers
    assert "human1_metabolism" in providers

    kg = HeterogeneousKnowledgeGraph(load_curated=False)

    # 1. CollecTRI (DoRothEA )
    n_col = kg.load_dataset("collectri")
    assert n_col >= 8
    paths_tf = kg.search_pathways("TP53", "CDKN1A")
    assert len(paths_tf) >= 1
    assert paths_tf[0].overall_effect_sign == 1

    # 2. LIANA ( -receptor network )
    n_lia = kg.load_dataset("liana")
    assert n_lia >= 5
    sub_tnf = kg.get_subgraph("TNF", radius=1)
    assert any(e.relation == "ligand_receptor_binding" for e in sub_tnf.edges)

    # 3. DrugComb ( network )
    n_dc = kg.load_dataset("drugcomb")
    assert n_dc >= 3
    paths_comb = kg.search_pathways("Trametinib", "Pictilisib")
    assert len(paths_comb) >= 1
    assert paths_comb[0].edges[0].relation == "synergizes_with"

    # 4. Human1 Metabolism (all gene - - network )
    n_met = kg.load_dataset("human1_metabolism")
    assert n_met >= 4
    paths_met = kg.search_pathways("IDH1", "TET2")
    assert len(paths_met) >= 1
    # IDH1 --(+1)--> 2-HG --(-1)--> TET2 => overall effect sign = -1 ( )
    assert paths_met[0].overall_effect_sign == -1

    # 5. DGIdb ( -gene target )
    assert "dgidb" in providers
    n_dg = kg.load_dataset("dgidb")
    assert n_dg >= 5
    paths_dg = kg.search_pathways("Trametinib", "MAP2K1")
    assert len(paths_dg) >= 1
    assert paths_dg[0].edges[0].relation == "inhibitor"

    # 6. PharmGKB ( gene gene )
    assert "pharmgkb" in providers
    n_pg = kg.load_dataset("pharmgkb")
    assert n_pg >= 5
    paths_pg = kg.search_pathways("Warfarin", "VKORC1")
    assert len(paths_pg) >= 1
    assert paths_pg[0].edges[0].relation == "pharmacogenomic_variant_association"

