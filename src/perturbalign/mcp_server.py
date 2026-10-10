"""PerturbAlign Model Context Protocol (MCP) Server.

Exposes PerturbAlign's biomedical entity alignment and causal knowledge graph capabilities
as standard Model Context Protocol (MCP) services for AI Agents (Claude Code, Antigravity,
Cursor, OpenAI Agents, LangChain).
"""
from __future__ import annotations

import json
import sys
from typing import Any, Callable, Optional

import perturbalign


class PerturbAlignMCPServer:
    """Standard MCP (Model Context Protocol) server."""

    def __init__(self) -> None:
        self.server_name = "perturbalign-mcp"
        self.server_version = perturbalign.__version__
        self._tools = self._build_tools_manifest()

    def _build_tools_manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "perturbalign_align",
                "description": "Clean, parse, and normalize single-cell perturbation labels (extracts drug name, dose, duration, controls, and combos)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "perturbations": {"type": "array", "items": {"type": "string"}, "description": "List of raw perturbation strings"},
                        "online": {"type": "boolean", "default": False, "description": "Whether to enable online API cascade retrieval"},
                    },
                    "required": ["perturbations"],
                },
            },
            {
                "name": "perturbalign_query_gene",
                "description": "Query comprehensive gene/protein knowledge and deep representations (NCBI, ESM2 embeddings, STRING PPI, DepMap dependency, Reactome)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "gene": {"type": "string", "description": "Gene symbol (e.g. 'TP53', 'EGFR')"},
                        "online": {"type": "boolean", "default": False},
                    },
                    "required": ["gene"],
                },
            },
            {
                "name": "perturbalign_query_compound",
                "description": "Query small molecule bioactivity and phenotypic profiles (PubChem, ChemBERTa embeddings, L1000 signatures, PRISM viability, JUMP morphology)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "compound": {"type": "string", "description": "Compound name or alias (e.g. 'Doxorubicin', 'Trametinib')"},
                        "online": {"type": "boolean", "default": False},
                    },
                    "required": ["compound"],
                },
            },
            {
                "name": "perturbalign_search_pathways",
                "description": "Search multi-hop causal regulatory paths and cumulative sign inference between source and target entities in the heterogeneous knowledge graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Source entity (e.g. 'Trametinib', 'TP53')"},
                        "target": {"type": "string", "description": "Target entity (e.g. 'CDKN1A', 'Apoptosis')"},
                        "max_hops": {"type": "integer", "default": 3, "description": "Maximum inference hops"},
                    },
                    "required": ["source", "target"],
                },
            },
            {
                "name": "perturbalign_predict_pathway_impact",
                "description": "Predict activation/inhibition causal impact and evidence chains on 14 core oncology signaling pathways from perturbations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "perturbation": {"type": "string", "description": "Perturbation entity (e.g. 'Trametinib', 'TP53')"},
                        "pathway": {"type": "string", "description": "Optional target pathway (e.g. 'Pathway:MAPK')"},
                    },
                    "required": ["perturbation"],
                },
            },
            {
                "name": "perturbalign_meta_path_search",
                "description": "Constrained meta-path search matching a sequence of node types (e.g. Compound->Gene->Pathway)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Source entity name"},
                        "meta_path": {"type": "array", "items": {"type": "string"}, "description": "Pattern sequence (e.g. ['Compound', 'Gene', 'Pathway'])"},
                        "max_paths": {"type": "integer", "default": 10},
                    },
                    "required": ["source", "meta_path"],
                },
            },
            {
                "name": "perturbalign_query_epigenetics",
                "description": "Query cis-regulatory elements (cCREs promoters/enhancers), binding motifs, and CRISPRi/a epigenetic perturbation impacts",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Target gene or perturbation (e.g. 'TP53', 'dCas9-KRAB:MYC')"},
                    },
                    "required": ["target"],
                },
            },
            {
                "name": "perturbalign_query_cell_line",
                "description": "Query tumor host cell line genetic background, driver mutations, DepMap dependencies, and cell model properties",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "cell_line": {"type": "string", "description": "Cell line name (e.g. 'A549', 'MCF7', 'K562', 'PC9')"},
                        "online": {"type": "boolean", "default": False},
                    },
                    "required": ["cell_line"],
                },
            },
            {
                "name": "perturbalign_predict_reversal_drugs",
                "description": "Compute Tau connectivity from CMap/L1000 signatures to recommend small molecules that reverse pathological expression signatures",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "up_genes": {"type": "array", "items": {"type": "string"}, "description": "Set of abnormally upregulated genes in disease state"},
                        "down_genes": {"type": "array", "items": {"type": "string"}, "description": "Set of abnormally downregulated genes in disease state"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["up_genes", "down_genes"],
                },
            },
            {
                "name": "perturbalign_evaluate_safety",
                "description": "Evaluate secondary pharmacology off-target liabilities and major organ toxicity risks (cardiac hERG/hepatic BSEP/black-box warnings) via SafetyScreen44",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Tested target or drug name (e.g. 'KCNH2', 'Doxorubicin', 'Trametinib')"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "perturbalign_enrich",
                "description": "Comprehensive perturbation knowledge enrichment and DoRothEA causal cascade analysis (outputs canonical name, SMILES, targets, master TFs, regulons, indications)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Perturbation text (e.g. 'AZD-6244 10uM', 'shTrp53', 'TNF-alpha', 'Hypoxia')"},
                        "organism": {"type": "string", "default": "auto", "description": "Experimental organism ('human', 'mouse', or 'auto')"},
                        "online": {"type": "boolean", "default": True, "description": "Whether to enable online dynamic retrieval"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "perturbalign_resolve_lab_code",
                "description": "Resolve pharmaceutical investigational codes (e.g. 'AZD-6244', 'GDC-0941', 'PD-0332991', 'BMS-354825') to INN generic name, targets, and developer",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Investigational code (e.g. 'AZD-6244', 'GDC-0941')"},
                    },
                    "required": ["code"],
                },
            },
            {
                "name": "perturbalign_align_ortholog",
                "description": "Cross-species gene to human ortholog symbol mapping (maps mouse/rat genes to official human HGNC Symbol and Ensembl/UniProt)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "gene": {"type": "string", "description": "Input gene symbol (e.g. 'Trp53')"},
                        "organism": {"type": "string", "default": "auto", "description": "Source organism (e.g. 'mouse', 'rat', or 'auto')"},
                        "online": {"type": "boolean", "default": True},
                    },
                    "required": ["gene"],
                },
            },
            {
                "name": "perturbalign_standardize_metadata",
                "description": "Batch validate and standardize perturbation tabular records into UPDS (Unified Perturbation Data Schema v1.0) format",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "records": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of dictionary records containing perturbation column",
                        },
                        "pert_col": {"type": "string", "default": "perturbation", "description": "Perturbation column key"},
                    },
                    "required": ["records"],
                },
            },
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Dispatch and execute specific PerturbAlign tool."""
        if name == "perturbalign_align":
            results = perturbalign.align(arguments.get("perturbations", []), online=arguments.get("online", False))
            return [r.to_dict() for r in results]
        elif name == "perturbalign_enrich":
            from .facade import enrich
            res = enrich(arguments["query"], organism=arguments.get("organism", "auto"), online=arguments.get("online", True))
            return res.as_dict()
        elif name == "perturbalign_resolve_lab_code":
            from .compound import resolve_investigational_code
            res = resolve_investigational_code(arguments["code"])
            return res or {"found": False, "query": arguments["code"]}
        elif name == "perturbalign_align_ortholog":
            from .ortholog import get_gene_orthologs
            return get_gene_orthologs(arguments["gene"], organism=arguments.get("organism", "auto"), online=arguments.get("online", True))
        elif name == "perturbalign_standardize_metadata":
            from .facade import enrich
            import pandas as pd
            records = arguments.get("records", [])
            pert_col = arguments.get("pert_col", "perturbation")
            if not records:
                return {"records": [], "count": 0}
            df = pd.DataFrame(records)
            if pert_col not in df.columns:
                return {"error": f"Column '{pert_col}' not found in records"}
            from .facade import standardize_anndata
            class SimpleStub:
                def __init__(self, obs):
                    self.obs = obs
                    self.uns = {}
            stub = SimpleStub(df.copy())
            out = standardize_anndata(stub, obs_col=pert_col, online=False)
            return {
                "records": out.obs.to_dict(orient="records"),
                "audit": out.uns.get("perturbalign", {}),
            }
        elif name == "perturbalign_query_gene":
            return perturbalign.query_gene(arguments["gene"], online=arguments.get("online", False))
        elif name == "perturbalign_query_compound":
            return perturbalign.query_compound(arguments["compound"], online=arguments.get("online", False))
        elif name == "perturbalign_search_pathways":
            paths = perturbalign.search_hetero_pathways(arguments["source"], arguments["target"], max_hops=arguments.get("max_hops", 3))
            return [p.to_dict() for p in paths]
        elif name == "perturbalign_predict_pathway_impact":
            return perturbalign.predict_pathway_impact(arguments["perturbation"], pathway=arguments.get("pathway"))
        elif name == "perturbalign_meta_path_search":
            paths = perturbalign.meta_path_search(arguments["source"], meta_path=arguments.get("meta_path"), max_paths=arguments.get("max_paths", 10))
            return [p.to_dict() for p in paths]
        elif name == "perturbalign_query_epigenetics":
            from .epigenetics import query_epigenetics
            return query_epigenetics(arguments["target"])
        elif name == "perturbalign_query_cell_line":
            from .cell_line import query_cell_line
            return query_cell_line(arguments["cell_line"], online=arguments.get("online", False))
        elif name == "perturbalign_predict_reversal_drugs":
            from .connectivity import predict_reversal_drugs
            return predict_reversal_drugs(arguments["up_genes"], arguments["down_genes"], top_k=arguments.get("top_k", 5))
        elif name == "perturbalign_evaluate_safety":
            from .safety import evaluate_target_safety
            return evaluate_target_safety(arguments["query"])
        raise KeyError(f"Unknown tool name: {name}")

    def handle_message(self, req: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Handle client requests conforming to JSON-RPC 2.0 / MCP specification."""
        msg_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": self.server_name, "version": self.server_version},
                    "capabilities": {"tools": {}},
                },
            }
        elif method == "notifications/initialized":
            return None
        elif method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
        elif method == "tools/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": self._tools}}
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            try:
                raw_res = self.call_tool(tool_name, tool_args)
                text_content = json.dumps(raw_res, ensure_ascii=False, indent=2)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": text_content}]},
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": f"Error: {str(e)}"}], "isError": True},
                }
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method {method} not found"}}

    def run_stdio(self) -> None:
        """Launch stdio long-running MCP server process."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self.handle_message(req)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                    sys.stdout.flush()
            except Exception as e:
                err_resp = {"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}}
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


def run_stdio_server() -> None:
    """Launch stdio long-running MCP server process."""
    server = PerturbAlignMCPServer()
    server.run_stdio()


if __name__ == "__main__":
    run_stdio_server()
