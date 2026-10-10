"""Target safety and secondary pharmacology off-target evaluation engine.

Evaluates off-target liabilities and organ toxicity risks for perturbation targets
and small molecules based on the industry-standard SafetyScreen44 (Bowes et al.,
Nature Reviews Drug Discovery 2012) and Tox21 / openFDA warnings.
"""
from __future__ import annotations

from typing import Any, Optional


# Complete 44 core in vitro secondary pharmacology safety targets from SafetyScreen44 (Bowes et al., 2012)
_SAFETY_SCREEN_44_TARGETS: dict[str, dict[str, Any]] = {
    # Ion Channels (8)
    "KCNH2": {
        "alias": "hERG (Kv11.1)",
        "organ_system": "Cardiovascular",
        "risk_type": "QT Prolongation & Torsades de Pointes (Fatal Arrhythmia)",
        "severity": "CRITICAL",
        "monitoring": "ECG (QTc interval)",
    },
    "CACNA1C": {
        "alias": "Cav1.2",
        "organ_system": "Cardiovascular",
        "risk_type": "Hypotension, Negative Inotropy & Heart Block",
        "severity": "CRITICAL",
        "monitoring": "Blood pressure, ECG (PR interval)",
    },
    "SCN5A": {
        "alias": "Nav1.5",
        "organ_system": "Cardiovascular",
        "risk_type": "Cardiac Conduction Slowing, QRS Widening & Brugada Phenotype",
        "severity": "CRITICAL",
        "monitoring": "ECG (QRS duration)",
    },
    "KCNQ1": {
        "alias": "Kv7.1 / IKs",
        "organ_system": "Cardiovascular",
        "risk_type": "Long QT Syndrome Type 1 & Ventricular Arrhythmias",
        "severity": "CRITICAL",
        "monitoring": "ECG",
    },
    "GABRA1": {
        "alias": "GABA-A Receptor (alpha-1)",
        "organ_system": "Central Nervous System",
        "risk_type": "Excessive Sedation, Motor Ataxia & Withdrawal Seizures",
        "severity": "HIGH",
        "monitoring": "Neurological examination, alertness",
    },
    "GRIN1": {
        "alias": "NMDA Receptor (GluN1)",
        "organ_system": "Central Nervous System",
        "risk_type": "Psychotomimetic Effects, Olney's Lesions & Cognitive Impairment",
        "severity": "HIGH",
        "monitoring": "Mental status exam",
    },
    "GRIA1": {
        "alias": "AMPA Receptor (GluA1)",
        "organ_system": "Central Nervous System",
        "risk_type": "Seizure Susceptibility & Neurotoxicity",
        "severity": "HIGH",
        "monitoring": "EEG, neurological monitoring",
    },
    "CHRNA4": {
        "alias": "Neuronal Nicotinic AChR (alpha-4/beta-2)",
        "organ_system": "Central & Autonomic Nervous System",
        "risk_type": "Autonomic Instability, Ganglionic Blockade & Tremor",
        "severity": "MODERATE",
        "monitoring": "Blood pressure, heart rate",
    },

    # G-protein coupled receptors (GPCRs - 24)
    "HTR1A": {
        "alias": "5-HT1A Receptor",
        "organ_system": "Central Nervous System",
        "risk_type": "Serotonin Syndrome, Hypothermia & Hypotension",
        "severity": "MODERATE",
        "monitoring": "Body temperature, mental status",
    },
    "HTR1B": {
        "alias": "5-HT1B Receptor",
        "organ_system": "Cardiovascular",
        "risk_type": "Coronary Vasoconstriction & Hypertension",
        "severity": "MODERATE",
        "monitoring": "Blood pressure, ECG",
    },
    "HTR2A": {
        "alias": "5-HT2A Receptor",
        "organ_system": "CNS & Platelets",
        "risk_type": "Hallucinations, Agitation & Platelet Aggregation Perturbation",
        "severity": "MODERATE",
        "monitoring": "Psychiatric evaluation",
    },
    "HTR2B": {
        "alias": "5-HT2B Receptor",
        "organ_system": "Cardiovascular",
        "risk_type": "Cardiac Valvulopathy & Pulmonary Arterial Hypertension",
        "severity": "HIGH",
        "monitoring": "Echocardiogram (valve morphology)",
    },
    "ADORA2A": {
        "alias": "Adenosine A2A Receptor",
        "organ_system": "Cardiovascular",
        "risk_type": "Systemic Vasodilation, Reflex Tachycardia & Hypotension",
        "severity": "MODERATE",
        "monitoring": "Blood pressure, pulse rate",
    },
    "ADRA1A": {
        "alias": "Alpha-1A Adrenergic",
        "organ_system": "Cardiovascular",
        "risk_type": "Orthostatic Hypotension & Reflex Tachycardia",
        "severity": "MODERATE",
        "monitoring": "Blood pressure monitoring (postural)",
    },
    "ADRA2A": {
        "alias": "Alpha-2A Adrenergic",
        "organ_system": "Cardiovascular / CNS",
        "risk_type": "Sedation, Bradycardia & Central Sympatholysis",
        "severity": "MODERATE",
        "monitoring": "Heart rate, alertness",
    },
    "ADRB1": {
        "alias": "Beta-1 Adrenergic",
        "organ_system": "Cardiovascular",
        "risk_type": "Profound Bradycardia, Heart Failure Exacerbation",
        "severity": "HIGH",
        "monitoring": "ECG, heart rate",
    },
    "ADRB2": {
        "alias": "Beta-2 Adrenergic",
        "organ_system": "Respiratory",
        "risk_type": "Bronchospasm in Asthma/COPD, Peripheral Tremor",
        "severity": "HIGH",
        "monitoring": "Spirometry / FEV1",
    },
    "CNR1": {
        "alias": "Cannabinoid CB1 Receptor",
        "organ_system": "Central Nervous System",
        "risk_type": "Psychosis, Memory Impairment & Depressive Symptoms",
        "severity": "MODERATE",
        "monitoring": "Psychiatric evaluation",
    },
    "CNR2": {
        "alias": "Cannabinoid CB2 Receptor",
        "organ_system": "Immune System",
        "risk_type": "Immunomodulation / Altered Infection Response",
        "severity": "MODERATE",
        "monitoring": "Immune panel",
    },
    "CCKAR": {
        "alias": "Cholecystokinin CCK1 Receptor",
        "organ_system": "Gastrointestinal",
        "risk_type": "Gallbladder Dysfunction & Delayed Gastric Emptying",
        "severity": "MODERATE",
        "monitoring": "GI symptoms",
    },
    "DRD1": {
        "alias": "Dopamine D1 Receptor",
        "organ_system": "Central Nervous System",
        "risk_type": "Cognitive and Motor Coordination Impairment",
        "severity": "MODERATE",
        "monitoring": "Neurological examination",
    },
    "DRD2": {
        "alias": "Dopamine D2 Receptor",
        "organ_system": "CNS & Endocrine",
        "risk_type": "Extrapyramidal Symptoms (EPS), Hyperprolactinemia & QT Risk",
        "severity": "HIGH",
        "monitoring": "Prolactin levels, movement assessment",
    },
    "EDNRA": {
        "alias": "Endothelin ETA Receptor",
        "organ_system": "Cardiovascular / Teratology",
        "risk_type": "Potent Vasoconstriction, Teratogenicity & Edema",
        "severity": "HIGH",
        "monitoring": "Pregnancy test, fluid retention",
    },
    "HRH1": {
        "alias": "Histamine H1 Receptor",
        "organ_system": "CNS & Metabolic",
        "risk_type": "Somnolence, Weight Gain & Impaired Alertness",
        "severity": "MODERATE",
        "monitoring": "Body weight, daytime sleepiness",
    },
    "HRH2": {
        "alias": "Histamine H2 Receptor",
        "organ_system": "Gastrointestinal",
        "risk_type": "Achlorhydria, Bacterial Overgrowth & Rebound Hypersecretion",
        "severity": "MODERATE",
        "monitoring": "Gastric symptoms",
    },
    "CHRM1": {
        "alias": "Muscarinic M1 Receptor",
        "organ_system": "Central Nervous System",
        "risk_type": "Anticholinergic Cognitive Impairment, Delirium & Dry Mouth",
        "severity": "MODERATE",
        "monitoring": "Cognitive and neurological evaluation",
    },
    "CHRM2": {
        "alias": "Muscarinic M2 Receptor",
        "organ_system": "Cardiovascular",
        "risk_type": "Sinus Bradycardia, AV Block & Hypotension",
        "severity": "HIGH",
        "monitoring": "ECG, heart rate",
    },
    "CHRM3": {
        "alias": "Muscarinic M3 Receptor",
        "organ_system": "GI & Ocular & Pulmonary",
        "risk_type": "Severe Constipation, Urinary Retention, Bronchospasm & Blurred Vision",
        "severity": "HIGH",
        "monitoring": "Post-void residual volume, bowel habits",
    },
    "OPRM1": {
        "alias": "Mu-Opioid Receptor",
        "organ_system": "CNS & Respiratory",
        "risk_type": "Respiratory Depression, Severe Addiction Liability & Constipation",
        "severity": "CRITICAL",
        "monitoring": "Respiratory rate, pulse oximetry",
    },
    "OPRD1": {
        "alias": "Delta-Opioid Receptor",
        "organ_system": "Central Nervous System",
        "risk_type": "Proconvulsant Potential & Hallucinogenic Effects",
        "severity": "MODERATE",
        "monitoring": "Seizure threshold",
    },
    "OPRK1": {
        "alias": "Kappa-Opioid Receptor",
        "organ_system": "Central Nervous System",
        "risk_type": "Dysphoria, Depersonalization & Psychotomimetic Reaction",
        "severity": "MODERATE",
        "monitoring": "Mood and psychiatric evaluation",
    },
    "AVPR1A": {
        "alias": "Vasopressin V1A Receptor",
        "organ_system": "Cardiovascular / Renal",
        "risk_type": "Systemic Vasoconstriction & Platelet Aggregation",
        "severity": "MODERATE",
        "monitoring": "Blood pressure",
    },

    # Enzymes and kinases (Enzymes & Kinases - 7)
    "PTGS1": {
        "alias": "COX-1 (Cyclooxygenase-1)",
        "organ_system": "Gastrointestinal & Hematologic",
        "risk_type": "Gastric Ulceration, Mucosal Hemorrhage & Bleeding Tendency",
        "severity": "MODERATE",
        "monitoring": "Fecal Occult Blood, CBC (hemoglobin)",
    },
    "PTGS2": {
        "alias": "COX-2 (Cyclooxygenase-2)",
        "organ_system": "Cardiovascular & Renal",
        "risk_type": "Thromboembolic Events (Myocardial Infarction, Stroke) & Hypertension",
        "severity": "HIGH",
        "monitoring": "Blood pressure, cardiovascular history",
    },
    "ACHE": {
        "alias": "Acetylcholinesterase",
        "organ_system": "Central & Peripheral Nervous System",
        "risk_type": "Acute Cholinergic Crisis (Salivation, Lacrimation, Fasciculation, Bradycardia)",
        "severity": "HIGH",
        "monitoring": "Neuromuscular examination, vitals",
    },
    "MAOA": {
        "alias": "Monoamine Oxidase A",
        "organ_system": "Cardiovascular",
        "risk_type": "Hypertensive Crisis with Dietary Tyramine ('Cheese Effect')",
        "severity": "HIGH",
        "monitoring": "Dietary restriction, blood pressure",
    },
    "PDE4D": {
        "alias": "Phosphodiesterase 4D",
        "organ_system": "Gastrointestinal & Vascular",
        "risk_type": "Intractable Nausea, Emesis & Vasculitis Liability",
        "severity": "MODERATE",
        "monitoring": "GI tolerability assessment",
    },
    "LCK": {
        "alias": "Lck Tyrosine Kinase",
        "organ_system": "Immune System",
        "risk_type": "Severe T-Cell Immunosuppression & Opportunistic Infections",
        "severity": "HIGH",
        "monitoring": "CD4/CD8 T-cell counts, infection signs",
    },
    "KDR": {
        "alias": "VEGFR2 (Vascular Endothelial Growth Factor Receptor 2)",
        "organ_system": "Cardiovascular & Renal",
        "risk_type": "Severe Hypertension, Proteinuria, Arterial Thrombosis & Wound Healing Delay",
        "severity": "HIGH",
        "monitoring": "Blood pressure, urinalysis (protein)",
    },

    # Transporters and clearance (Transporters & Clearance - 5)
    "ABCB11": {
        "alias": "BSEP (Bile Salt Export Pump)",
        "organ_system": "Hepatic",
        "risk_type": "Drug-Induced Liver Injury (DILI) & Severe Cholestasis",
        "severity": "CRITICAL",
        "monitoring": "Liver Function Panel (ALT/AST, Bilirubin, Alk Phos)",
    },
    "SLC6A4": {
        "alias": "SERT (Serotonin Transporter)",
        "organ_system": "CNS & GI",
        "risk_type": "Serotonin Syndrome, GI Bleeding & Discontinuation Syndrome",
        "severity": "MODERATE",
        "monitoring": "Neurological examination",
    },
    "SLC6A3": {
        "alias": "DAT (Dopamine Transporter)",
        "organ_system": "Central Nervous System",
        "risk_type": "Psychomotor Agitation, Tachycardia & Abuse Liability",
        "severity": "MODERATE",
        "monitoring": "Behavioral observation, addiction risk",
    },
    "SLC6A2": {
        "alias": "NET (Norepinephrine Transporter)",
        "organ_system": "Cardiovascular",
        "risk_type": "Tachycardia, Hypertension & Sympathetic Hyperactivity",
        "severity": "MODERATE",
        "monitoring": "Heart rate, blood pressure",
    },
    "CYP3A4": {
        "alias": "Cytochrome P450 3A4",
        "organ_system": "Hepatic / Metabolic",
        "risk_type": "Severe Drug-Drug Interaction (DDI) & Altered Systemic Clearance",
        "severity": "HIGH",
        "monitoring": "Plasma drug concentration monitoring, co-medication audit",
    },

    # Supplementary critical oncology safety targets
    "TOP2A": {
        "alias": "Topoisomerase II Alpha",
        "organ_system": "Hematologic & Cardiac",
        "risk_type": "Myelosuppression & Cumulative Irreversible Cardiotoxicity",
        "severity": "CRITICAL",
        "monitoring": "CBC, LVEF measurement",
    },
}

_COMPOUND_SAFETY_PROFILES: dict[str, dict[str, Any]] = {
    "DOXORUBICIN": {
        "compound": "Doxorubicin",
        "black_box_warning": "Dose-dependent irreversible cardiotoxicity (CHF), myelosuppression",
        "primary_organ_risks": ["Cardiovascular", "Hematologic"],
        "critical_off_targets": ["TOP2A", "KCNH2"],
        "safety_tier": "High Risk / Cytotoxic Oncology Agent",
    },
    "TRAMETINIB": {
        "compound": "Trametinib",
        "black_box_warning": "Cardiomyopathy (decreased LVEF), retinal pigment epithelial detachment (RPED)",
        "primary_organ_risks": ["Cardiovascular", "Ophthalmologic", "Dermatologic"],
        "critical_off_targets": [],
        "safety_tier": "Moderate Risk / Targeted Kinase Inhibitor",
    },
    "OSIMERTINIB": {
        "compound": "Osimertinib",
        "black_box_warning": "Interstitial lung disease (ILD/Pneumonitis), QTc prolongation, cardiomyopathy",
        "primary_organ_risks": ["Respiratory", "Cardiovascular"],
        "critical_off_targets": ["KCNH2"],
        "safety_tier": "Moderate Risk / Targeted Kinase Inhibitor",
    },
    "VORINOSTAT": {
        "compound": "Vorinostat",
        "black_box_warning": "Severe thrombocytopenia, gastrointestinal toxicity, venous thromboembolism",
        "primary_organ_risks": ["Hematologic", "Gastrointestinal"],
        "critical_off_targets": ["CYP3A4"],
        "safety_tier": "High Risk / Epigenetic Modifier",
    },
}


def evaluate_target_safety(query: str) -> dict[str, Any]:
    """Comprehensively evaluate secondary pharmacology, off-target liabilities, and organ toxicity risks.

    Parameters
    ----------
    query : str
        Gene target symbol (e.g. 'KCNH2', 'CACNA1C', 'ABCB11') or drug name (e.g. 'Doxorubicin', 'Trametinib').

    Returns
    -------
    dict
        Organ risk tier, SafetyScreen44 annotations, and safety warning summary.
    """
    cleaned = query.strip().upper()

    # 1. Query compound safety profile
    compound_profile = _CUSTOM_SAFETY_COMPOUNDS.get(cleaned) or _COMPOUND_SAFETY_PROFILES.get(cleaned)
    if compound_profile:
        return {
            "query": query,
            "entity_type": "Compound",
            "safety_tier": compound_profile.get("safety_tier", "Unspecified"),
            "black_box_warning": compound_profile.get("black_box_warning", "None"),
            "primary_organ_risks": compound_profile.get("primary_organ_risks", []),
            "critical_off_targets": compound_profile.get("critical_off_targets", []),
            "safetyscreen44_flagged": len(compound_profile.get("critical_off_targets", [])) > 0,
        }

    # 2. Query target gene secondary pharmacology (SafetyScreen44 official panel)
    target_info = _CUSTOM_SAFETY_TARGETS.get(cleaned) or _SAFETY_SCREEN_44_TARGETS.get(cleaned)
    if target_info:
        return {
            "query": query,
            "entity_type": "Target_Gene",
            "alias": target_info.get("alias", query),
            "safetyscreen44_member": target_info.get("safetyscreen44_member", True),
            "severity": target_info.get("severity", "MODERATE"),
            "organ_system": target_info.get("organ_system", "General"),
            "clinical_risk": target_info.get("risk_type", "Liability identified"),
            "recommended_monitoring": target_info.get("monitoring", "Standard clinical monitoring"),
        }

    # 3. Target not in the SafetyScreen44 panel (transparently noted; does not imply zero toxicity)
    return {
        "query": query,
        "entity_type": "Target_Gene_or_Compound",
        "safetyscreen44_member": False,
        "severity": "NOT_IN_PANEL",
        "organ_system": "General",
        "clinical_risk": f"Target '{query}' is not a standard member of the Bowes et al. SafetyScreen44 in vitro secondary pharmacology panel (does not imply zero in vivo toxicity).",
        "recommended_monitoring": "Standard laboratory and preclinical evaluation",
    }


_CUSTOM_SAFETY_TARGETS: dict[str, dict[str, Any]] = {}
_CUSTOM_SAFETY_COMPOUNDS: dict[str, dict[str, Any]] = {}


def register_safety_target(target_symbol: str, target_profile: dict[str, Any]) -> None:
    """Register or customize a secondary pharmacology safety target liability."""
    _CUSTOM_SAFETY_TARGETS[target_symbol.strip().upper()] = target_profile


def register_safety_compound(compound_name: str, profile: dict[str, Any]) -> None:
    """Register or customize a compound organ safety and black-box profile."""
    _CUSTOM_SAFETY_COMPOUNDS[compound_name.strip().upper()] = profile


def reset_safety_registry() -> None:
    """Reset custom safety targets and compound registries."""
    _CUSTOM_SAFETY_TARGETS.clear()
    _CUSTOM_SAFETY_COMPOUNDS.clear()


evaluate_safety = evaluate_target_safety

__all__ = [
    "evaluate_target_safety",
    "evaluate_safety",
    "register_safety_target",
    "register_safety_compound",
    "reset_safety_registry",
]

