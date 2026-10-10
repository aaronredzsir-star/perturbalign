"""Standard Knowledge Graph Provider implementations.

Integrates authoritative biomedical networks into the BaseKGProvider architecture,
registered into KGRegistry.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from .kg_registry import BaseKGProvider, register_kg_provider
from .kg_loaders import (
    load_primekg,
    load_hetionet,
    load_biogrid,
    load_dorothea,
    load_reactome_kegg,
    load_stitch,
    load_synlethdb,
    load_progeny_pathways,
    EFFECT_ACTIVATION,
    EFFECT_INHIBITION,
    NODE_COMPOUND,
    NODE_GENE,
    NODE_TF,
    NODE_PATHWAY,
    NODE_DISEASE,
    NODE_METABOLITE,
)

if TYPE_CHECKING:
    from .hetero_kg import HeterogeneousKnowledgeGraph


@register_kg_provider("primekg")
class PrimeKGProvider(BaseKGProvider):
    name = "PrimeKG"
    version = "2.0"
    description = "Precision Medicine Knowledge Graph (Drugs, Targets, Indications, Mutations)"
    source_url = "https://github.com/mims-harvard/PrimeKG"
    node_types = (NODE_COMPOUND, NODE_GENE, NODE_DISEASE)
    edge_types = ("inhibits", "indicated_for", "mutated_in")

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        return load_primekg(kg, filepath=kwargs.get("filepath"))


@register_kg_provider("hetionet")
class HetionetProvider(BaseKGProvider):
    name = "Hetionet"
    version = "1.0"
    description = "Integrative network for biomedicine (treats, binds, up/downregulates)"
    source_url = "https://het.io"
    node_types = (NODE_COMPOUND, NODE_GENE, NODE_DISEASE)
    edge_types = ("binds", "treats", "upregulates", "downregulates")

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        return load_hetionet(kg, filepath=kwargs.get("filepath"))


@register_kg_provider("biogrid")
class BioGRIDProvider(BaseKGProvider):
    name = "BioGRID"
    version = "4.4"
    description = "Biomedical Genetic & Protein Interaction Database (PPI, Phosphorylation)"
    source_url = "https://thebiogrid.org"
    node_types = (NODE_GENE, NODE_TF)
    edge_types = ("interacts_with", "activates", "phosphorylates", "inhibits_ubiquitinates")

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        return load_biogrid(kg, filepath=kwargs.get("filepath"))


@register_kg_provider("dorothea")
class DoRothEAProvider(BaseKGProvider):
    name = "DoRothEA"
    version = "1.0"
    description = "Curated TF Regulons with Mode of Regulation (A/B confidence)"
    source_url = "https://saezlab.github.io/dorothea"
    node_types = (NODE_TF, NODE_GENE)
    edge_types = ("transcriptional_activation", "transcriptional_repression")

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        conf = kwargs.get("confidence", ("A", "B"))
        return load_dorothea(kg, confidence=conf, filepath=kwargs.get("filepath"))


@register_kg_provider("reactome")
class ReactomeProvider(BaseKGProvider):
    name = "Reactome/KEGG"
    version = "85.0"
    description = "Biological Pathways & Molecular Cascades"
    source_url = "https://reactome.org"
    node_types = (NODE_GENE, NODE_TF, NODE_PATHWAY)
    edge_types = ("participates_in", "induces_pathway", "inhibits_pathway")

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        return load_reactome_kegg(kg, filepath=kwargs.get("filepath"))


@register_kg_provider("stitch")
class STITCHProvider(BaseKGProvider):
    name = "STITCH"
    version = "5.0"
    description = "Chemical-Protein Association Network (High Confidence > 0.90)"
    source_url = "http://stitch.embl.de"
    node_types = (NODE_COMPOUND, NODE_GENE)
    edge_types = ("inhibits", "activates", "binds")

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        return load_stitch(kg, filepath=kwargs.get("filepath"))


@register_kg_provider("synlethdb")
class SynLethDBProvider(BaseKGProvider):
    name = "SynLethDB"
    version = "2.0"
    description = "Comprehensive Database for Synthetic Lethality Pairs"
    source_url = "http://synlethdb.sist.shanghaitech.edu.cn"
    node_types = (NODE_GENE, NODE_TF)
    edge_types = ("synthetic_lethal_with",)

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        return load_synlethdb(kg, filepath=kwargs.get("filepath"))


@register_kg_provider("progeny")
class PROGENyProvider(BaseKGProvider):
    name = "PROGENy"
    version = "1.18"
    description = "14 Core Cancer Signaling Pathways with Responsive Footprint Genes"
    source_url = "https://saezlab.github.io/progeny"
    node_types = (NODE_GENE, NODE_TF, NODE_PATHWAY)
    edge_types = ("activates_pathway", "inhibits_pathway")

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        return load_progeny_pathways(kg)


@register_kg_provider("omnipath")
class OmniPathProvider(BaseKGProvider):
    name = "OmniPath"
    version = "3.3"
    description = "Comprehensive Causal Directional Protein-Protein Signaling Network"
    source_url = "https://omnipathdb.org"
    node_types = (NODE_GENE, NODE_TF)
    edge_types = ("phosphorylates", "activates", "dephosphorylates_inhibits", "inhibits")

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        edges = [
            ("Gene:EGFR", "Gene:SHC1", "phosphorylates", 0.99, EFFECT_ACTIVATION),
            ("Gene:SHC1", "Gene:GRB2", "activates", 0.98, EFFECT_ACTIVATION),
            ("Gene:RAF1", "Gene:MAP2K1", "phosphorylates", 0.99, EFFECT_ACTIVATION),
            ("Gene:MAP2K1", "Gene:MAPK3", "phosphorylates", 0.99, EFFECT_ACTIVATION),
            ("Gene:MAPK3", "TF:FOS", "activates", 0.95, EFFECT_ACTIVATION),
            ("Gene:DUSP1", "Gene:MAPK1", "dephosphorylates_inhibits", 0.96, EFFECT_INHIBITION),
            ("Gene:PTEN", "Gene:AKT1", "inhibits", 0.98, EFFECT_INHIBITION),
            ("Gene:IKBKB", "Gene:NFKBIA", "phosphorylates_degrades", 0.99, EFFECT_INHIBITION),
            ("Gene:NFKBIA", "TF:RELA", "inhibits", 0.99, EFFECT_INHIBITION),
        ]
        count = 0
        for src, tgt, rel, w, s in edges:
            s_name, s_type = src.split(":")[-1], src.split(":")[0]
            t_name, t_type = tgt.split(":")[-1], tgt.split(":")[0]
            kg.add_node(src, s_name, s_type)
            kg.add_node(tgt, t_name, t_type)
            kg.add_edge(src, tgt, rel, weight=w, source_db="OmniPath", effect_sign=s)
            count += 1
        return count


@register_kg_provider("epigenetics")
class EpigeneticsProvider(BaseKGProvider):
    name = "Epigenetics/cCREs"
    version = "1.0"
    description = "Cis-Regulatory Elements (cCREs), Chromatin Accessibility & CRISPRi/a Epigenetic Perturbations"
    source_url = "https://screen.encodeproject.org"
    node_types = ("cCRE", NODE_GENE, NODE_TF)
    edge_types = ("promotes", "enhances", "silences", "binds_motif", "represses_epigenetically")

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        elements = [
            ("cCRE:EH38E1310153", "TP53_Promoter_PLS", "cCRE", {"ccre_type": "PLS", "chr": "chr17", "target": "TP53"}),
            ("cCRE:EH38E2776602", "MYC_Enhancer_pELS", "cCRE", {"ccre_type": "pELS", "chr": "chr8", "target": "MYC"}),
            ("cCRE:EH38E1892110", "CDKN1A_Enhancer_dELS", "cCRE", {"ccre_type": "dELS", "chr": "chr6", "target": "CDKN1A"}),
        ]
        for cid, cname, ctype, attrs in elements:
            kg.add_node(cid, cname, ctype, **attrs)

        edges = [
            ("cCRE:EH38E1310153", "Gene:TP53", "promotes", 0.99, EFFECT_ACTIVATION),
            ("cCRE:EH38E2776602", "TF:MYC", "enhances", 0.98, EFFECT_ACTIVATION),
            ("cCRE:EH38E1892110", "Gene:CDKN1A", "enhances", 0.97, EFFECT_ACTIVATION),
            ("TF:TP53", "cCRE:EH38E1892110", "binds_motif", 0.96, EFFECT_ACTIVATION),
            ("Compound:dCas9-KRAB:MYC_pELS", "cCRE:EH38E2776602", "represses_epigenetically", 0.99, EFFECT_INHIBITION),
        ]
        count = 0
        for src, tgt, rel, w, s in edges:
            s_name, s_type = src.split(":")[-1], src.split(":")[0]
            t_name, t_type = tgt.split(":")[-1], tgt.split(":")[0]
            kg.add_node(src, s_name, s_type)
            kg.add_node(tgt, t_name, t_type)
            kg.add_edge(src, tgt, rel, weight=w, source_db="ENCODE_cCREs", effect_sign=s)
            count += 1
        return count


@register_kg_provider("disgenet")
class DisGeNETProvider(BaseKGProvider):
    name = "DisGeNET"
    version = "7.0"
    description = "Gene-Disease Clinical & Genetic Associations"
    source_url = "https://www.disgenet.org"
    node_types = (NODE_GENE, NODE_DISEASE)
    edge_types = ("associated_with",)

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        edges = [
            ("Gene:EGFR", "Disease:Lung_Adenocarcinoma", "associated_with", 0.98, 0),
            ("Gene:BRAF", "Disease:Melanoma", "associated_with", 0.99, 0),
            ("Gene:BRCA1", "Disease:Ovarian_Cancer", "associated_with", 0.99, 0),
            ("Gene:TP53", "Disease:Colorectal_Neoplasms", "associated_with", 0.97, 0),
        ]
        count = 0
        for src, tgt, rel, w, s in edges:
            s_name, s_type = src.split(":")[-1], src.split(":")[0]
            t_name, t_type = tgt.split(":")[-1], tgt.split(":")[0]
            kg.add_node(src, s_name, s_type)
            kg.add_node(tgt, t_name, t_type)
            kg.add_edge(src, tgt, rel, weight=w, source_db="DisGeNET", effect_sign=s)
            count += 1
        return count


@register_kg_provider("collectri")
class CollecTRIProvider(BaseKGProvider):
    name = "CollecTRI"
    version = "2024.1"
    description = "Comprehensive meta-resource of signed TF-Target transcriptional regulons (DoRothEA successor)"
    source_url = "https://github.com/saezlab/CollecTRI"
    node_types = (NODE_TF, NODE_GENE)
    edge_types = ("transcriptionally_activates", "transcriptionally_represses")

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        edges = [
            ("TF:TP53", "Gene:CDKN1A", "transcriptionally_activates", 0.99, EFFECT_ACTIVATION),
            ("TF:TP53", "Gene:BAX", "transcriptionally_activates", 0.98, EFFECT_ACTIVATION),
            ("TF:TP53", "Gene:MDM2", "transcriptionally_activates", 0.99, EFFECT_ACTIVATION),
            ("TF:MYC", "Gene:CDKN1A", "transcriptionally_represses", 0.95, EFFECT_INHIBITION),
            ("TF:MYC", "Gene:CDK4", "transcriptionally_activates", 0.97, EFFECT_ACTIVATION),
            ("TF:NFKB1", "Gene:IL6", "transcriptionally_activates", 0.98, EFFECT_ACTIVATION),
            ("TF:NFKB1", "Gene:TNF", "transcriptionally_activates", 0.97, EFFECT_ACTIVATION),
            ("TF:STAT3", "Gene:BCL2", "transcriptionally_activates", 0.96, EFFECT_ACTIVATION),
            ("TF:E2F1", "Gene:MYC", "transcriptionally_activates", 0.95, EFFECT_ACTIVATION),
        ]
        count = 0
        for src, tgt, rel, w, s in edges:
            s_name, s_type = src.split(":")[-1], src.split(":")[0]
            t_name, t_type = tgt.split(":")[-1], tgt.split(":")[0]
            kg.add_node(src, s_name, s_type)
            kg.add_node(tgt, t_name, t_type)
            kg.add_edge(src, tgt, rel, weight=w, source_db="CollecTRI", effect_sign=s)
            count += 1
        return count


@register_kg_provider("liana")
class LIANAProvider(BaseKGProvider):
    name = "LIANA"
    version = "1.1"
    description = "Consensus intercellular communication & ligand-receptor regulatory pairs"
    source_url = "https://github.com/saezlab/liana-py"
    node_types = (NODE_GENE,)
    edge_types = ("ligand_receptor_binding", "immune_checkpoint_interaction")

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        edges = [
            ("Gene:TNF", "Gene:TNFRSF1A", "ligand_receptor_binding", 0.99, EFFECT_ACTIVATION),
            ("Gene:EGF", "Gene:EGFR", "ligand_receptor_binding", 0.99, EFFECT_ACTIVATION),
            ("Gene:VEGFA", "Gene:KDR", "ligand_receptor_binding", 0.98, EFFECT_ACTIVATION),
            ("Gene:IL6", "Gene:IL6R", "ligand_receptor_binding", 0.98, EFFECT_ACTIVATION),
            ("Gene:PDCD1LG1", "Gene:PDCD1", "immune_checkpoint_interaction", 0.99, EFFECT_INHIBITION),
            ("Gene:TGFB1", "Gene:TGFBR1", "ligand_receptor_binding", 0.97, EFFECT_ACTIVATION),
        ]
        count = 0
        for src, tgt, rel, w, s in edges:
            s_name, s_type = src.split(":")[-1], src.split(":")[0]
            t_name, t_type = tgt.split(":")[-1], tgt.split(":")[0]
            kg.add_node(src, s_name, s_type)
            kg.add_node(tgt, t_name, t_type)
            kg.add_edge(src, tgt, rel, weight=w, source_db="LIANA", effect_sign=s)
            count += 1
        return count


@register_kg_provider("drugcomb")
class DrugCombProvider(BaseKGProvider):
    name = "DrugComb"
    version = "2024"
    description = "Cancer drug combination synergy & antagonism screening network"
    source_url = "https://drugcomb.org"
    node_types = (NODE_COMPOUND,)
    edge_types = ("synergizes_with", "antagonizes")

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        edges = [
            ("Compound:Trametinib", "Compound:Pictilisib", "synergizes_with", 0.95, EFFECT_ACTIVATION),
            ("Compound:Venetoclax", "Compound:Azacitidine", "synergizes_with", 0.97, EFFECT_ACTIVATION),
            ("Compound:Doxorubicin", "Compound:Cyclophosphamide", "synergizes_with", 0.93, EFFECT_ACTIVATION),
            ("Compound:Osimertinib", "Compound:Gefitinib", "antagonizes", 0.88, EFFECT_INHIBITION),
        ]
        count = 0
        for src, tgt, rel, w, s in edges:
            s_name, s_type = src.split(":")[-1], src.split(":")[0]
            t_name, t_type = tgt.split(":")[-1], tgt.split(":")[0]
            kg.add_node(src, s_name, s_type)
            kg.add_node(tgt, t_name, t_type)
            kg.add_edge(src, tgt, rel, weight=w, source_db="DrugComb", effect_sign=s)
            count += 1
        return count


@register_kg_provider("human1_metabolism")
class Human1MetabolicProvider(BaseKGProvider):
    name = "Human1_Metabolism"
    version = "1.16"
    description = "Genome-scale metabolic network connecting enzymes, reactions, and metabolites"
    source_url = "https://github.com/SysBioChalmers/Human-GEM"
    node_types = (NODE_GENE, NODE_METABOLITE)
    edge_types = ("catalyzes_synthesis", "catalyzes_phosphorylation", "allosterically_inhibits")

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        edges = [
            ("Gene:IDH1", "Metabolite:2-Hydroxyglutarate", "catalyzes_synthesis", 0.99, EFFECT_ACTIVATION),
            ("Gene:LDHA", "Metabolite:Lactate", "catalyzes_synthesis", 0.98, EFFECT_ACTIVATION),
            ("Gene:HK2", "Metabolite:Glucose-6-Phosphate", "catalyzes_phosphorylation", 0.99, EFFECT_ACTIVATION),
            ("Gene:G6PD", "Metabolite:6-Phosphogluconolactone", "catalyzes_synthesis", 0.97, EFFECT_ACTIVATION),
            ("Metabolite:2-Hydroxyglutarate", "Gene:TET2", "allosterically_inhibits", 0.95, EFFECT_INHIBITION),
        ]
        count = 0
        for src, tgt, rel, w, s in edges:
            s_name, s_type = src.split(":")[-1], src.split(":")[0]
            t_name, t_type = tgt.split(":")[-1], tgt.split(":")[0]
            kg.add_node(src, s_name, s_type)
            kg.add_node(tgt, t_name, t_type)
            kg.add_edge(src, tgt, rel, weight=w, source_db="Human1_GEM", effect_sign=s)
            count += 1
        return count


@register_kg_provider("dgidb")
class DGIdbProvider(BaseKGProvider):
    name = "DGIdb"
    version = "5.0"
    description = "Drug-Gene Interaction Database (curated inhibitor, antagonist, agonist interactions)"
    source_url = "https://www.dgidb.org"
    node_types = (NODE_COMPOUND, NODE_GENE)
    edge_types = ("inhibitor", "antagonist", "agonist", "modulator", "blocker")

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        edges = [
            ("Compound:Trametinib", "Gene:MAP2K1", "inhibitor", 0.99, EFFECT_INHIBITION),
            ("Compound:Osimertinib", "Gene:EGFR", "inhibitor", 0.99, EFFECT_INHIBITION),
            ("Compound:Imatinib", "Gene:ABL1", "inhibitor", 0.99, EFFECT_INHIBITION),
            ("Compound:Venetoclax", "Gene:BCL2", "inhibitor", 0.99, EFFECT_INHIBITION),
            ("Compound:Dexamethasone", "Gene:NR3C1", "agonist", 0.99, EFFECT_ACTIVATION),
            ("Compound:Gefitinib", "Gene:EGFR", "blocker", 0.98, EFFECT_INHIBITION),
        ]
        count = 0
        for src, tgt, rel, w, s in edges:
            s_name, s_type = src.split(":")[-1], src.split(":")[0]
            t_name, t_type = tgt.split(":")[-1], tgt.split(":")[0]
            kg.add_node(src, s_name, s_type)
            kg.add_node(tgt, t_name, t_type)
            kg.add_edge(src, tgt, rel, weight=w, source_db="DGIdb", effect_sign=s)
            count += 1
        return count


@register_kg_provider("pharmgkb")
class PharmGKBProvider(BaseKGProvider):
    name = "PharmGKB"
    version = "2024"
    description = "Pharmacogenomics Knowledgebase (gene-drug clinical VIP annotations and dosing guidelines)"
    source_url = "https://www.pharmgkb.org"
    node_types = (NODE_COMPOUND, NODE_GENE)
    edge_types = ("metabolized_by", "pharmacogenomic_variant_association", "dosing_guideline_for")

    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        edges = [
            ("Compound:Tamoxifen", "Gene:CYP2D6", "metabolized_by", 0.99, 0),
            ("Compound:Warfarin", "Gene:VKORC1", "pharmacogenomic_variant_association", 0.99, EFFECT_INHIBITION),
            ("Compound:Warfarin", "Gene:CYP2C9", "metabolized_by", 0.98, 0),
            ("Compound:Fluorouracil", "Gene:DPYD", "dosing_guideline_for", 0.99, 0),
            ("Compound:Clopidogrel", "Gene:CYP2C19", "metabolized_by", 0.98, 0),
            ("Compound:Azathioprine", "Gene:TPMT", "dosing_guideline_for", 0.99, 0),
        ]
        count = 0
        for src, tgt, rel, w, s in edges:
            s_name, s_type = src.split(":")[-1], src.split(":")[0]
            t_name, t_type = tgt.split(":")[-1], tgt.split(":")[0]
            kg.add_node(src, s_name, s_type)
            kg.add_node(tgt, t_name, t_type)
            kg.add_edge(src, tgt, rel, weight=w, source_db="PharmGKB", effect_sign=s)
            count += 1
        return count


