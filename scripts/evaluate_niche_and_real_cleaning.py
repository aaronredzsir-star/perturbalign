"""
PerturbAlign 数据清洗能力真实与冷门数据集全面评测
覆盖：
1. 真实 Tahoe-100M 复杂药物与盐型
2. 真实单细胞组合扰动 (Combosciplex / Norman 2019 / Gehring-Pachter)
3. 真实冷门微环境物理胁迫与生物大分子 (Hypoxia, Heat shock, TNF, IL-6, 单抗)
4. 真实医药制剂与试剂商标注 (Broad Repurposing Hub / DailyMed / openFDA / Selleck / MCE)
5. 真实复杂遗传扰动与同源基因 (CRISPRi, CRISPRa, 双敲除, 突变, 小鼠同源)
6. 官方协议数据 (LINCS L1000 500条真实签名 / PRISM / JUMP-CP)
7. 单细胞 AnnData 端到端 UPDS-v1.0 清洗标准化
"""
import sys
import os
import time
import gzip
import json
import pandas as pd
import numpy as np

# 确保 UTF-8 编码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import perturbalign as pa

def run_all_evaluations():
    print("=" * 80)
    print(f" PerturbAlign 真实与冷门多组学扰动数据清洗能力全景测评 (v{pa.__version__})")
    print("=" * 80)

    # -------------------------------------------------------------
    # 评测 1: 真实 Tahoe-100M 单细胞药物大筛选
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("【评测 1】真实 Tahoe-100M 大规模扰动数据集 (65,618 细胞条件 / 378 种药物)")
    print("-" * 75)
    tahoe_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "perturbalign_data_archive", "reference", "tahoe_conditions.parquet"))
    df_tahoe = pd.read_parquet(tahoe_path)
    unique_drugs = df_tahoe['drug'].unique().tolist()
    
    t0 = time.time()
    tahoe_res = pa.align(unique_drugs, online=False)
    t_elapsed = time.time() - t0
    
    salt_stripped = sum(1 for d, r in zip(unique_drugs, tahoe_res) if ('(' in d or '-' in d) and r.parsed_name != d)
    print(f"  * 样本总数: {len(unique_drugs)} 种唯一样本药物标签 (覆盖 65,618 个单细胞实验条件)")
    print(f"  * 清洗耗时: {t_elapsed*1000:.1f} ms (吞吐量: {len(unique_drugs)/(t_elapsed+1e-6):.1f} 条/秒)")
    print(f"  * 成功去除盐型、异构体、制剂杂质的标签数: {salt_stripped} 种")
    
    tahoe_demos = [
        '(R)-Verapamil (hydrochloride)',
        '(S)-Crizotinib',
        'Almonertinib (hydrochloride)',
        'Almonertinib (mesylate)',
        'Palbociclib (isethionate)',
        'Balsalazide (sodium hydrate)',
        'AZD-7648',
        'AZD-8055',
        'Abiraterone acetate',
        'DMSO'
    ]
    for d in tahoe_demos:
        r = pa.align([d], online=False)[0]
        print(f"    - [原始] {d:<32} -> [标准名] {r.parsed_name:<16} [模态] {r.modality:<14} [对照] {str(r.is_control):<5}")

    # -------------------------------------------------------------
    # 评测 2: 冷门微环境胁迫与生物大分子/细胞因子 (Microenvironment & Biologics)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("【评测 2】冷门微环境物理/化学胁迫、细胞因子与生物药 (8大经典应激体系)")
    print("-" * 75)
    niche_env_cases = [
        "Hypoxia 1% O2 12h",
        "Heat shock 42C 30min",
        "Glucose starvation 6h",
        "Acidosis pH 6.5",
        "Hyperosmotic stress 200mM NaCl",
        "Cold shock 32C 24h",
        "Serum starvation 48h",
        "TNF-alpha 50ng/ml",
        "IL-6 100ng/ml 24h",
        "IFN-gamma 100U/ml",
        "Trastuzumab 10ug/ml",
        "Pembrolizumab 5mg/kg",
        "LNP-mRNA-luciferase"
    ]
    
    for case in niche_env_cases:
        r = pa.align([case], online=False)[0]
        e = pa.enrich(case, online=False)
        dose_s = f"{r.parsed_dose} {r.parsed_unit}" if r.parsed_dose is not None else "-"
        time_s = f"{r.parsed_duration} {r.parsed_duration_unit}" if r.parsed_duration is not None else "-"
        tfs = e.properties.get("sensor_transcription_factors") or e.properties.get("receptors") or []
        tf_str = ";".join(tfs[:2]) if tfs else "-"
        print(f"    - [输入] {case:<32} -> [模态] {r.modality:<14} [名称] {r.parsed_name:<16} [剂量] {dose_s:<12} [时间] {time_s:<10} [关键TF/受体] {tf_str}")

    # -------------------------------------------------------------
    # 评测 3: 真实多药鸡尾酒与微滴组合筛选 (Combinatorial Perturbations)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("【评测 3】复杂多药联用、微滴鸡尾酒与双基因敲除 (Combosciplex & Norman 2019)")
    print("-" * 75)
    combo_cases = [
        "Trametinib 10nM + Pictilisib 1uM",
        "Sorafenib 5uM + Doxorubicin 100nM 24h",
        "sgBCL2L1+sgMCL1",
        "sgMAPK1+sgMAP2K1",
        "DUSP9+ETS2",
        "1:1 Trametinib:Pictilisib",
        "M1 macrophage",
        "Th17 differentiation",
        "T-cell activation",
        "PMA ionomycin"
    ]
    for case in combo_cases:
        r = pa.align([case], online=False)[0]
        if r.parts:
            parts_desc = " + ".join([f"{p.parsed_name} ({p.parsed_dose} {p.parsed_unit})" if p.parsed_dose else p.parsed_name for p in r.parts])
        else:
            parts_desc = r.note or "-"
        print(f"    - [组合] {case:<38} -> [模态] {r.modality:<10} [拆解组分] {parts_desc}")

    # -------------------------------------------------------------
    # 评测 4: 极难医药制剂、试剂商代号与反离子保护 (Dirty Clinical / Formulations)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("【评测 4】极难商业制剂代码、试剂商前缀剥离与反离子保护 (DailyMed / Repurposing Hub)")
    print("-" * 75)
    dirty_formulations = [
        ("Diamox sr 250mg", "缓释代号 sr 剥离与剂量解析"),
        ("Selleck S1001 10µM", "试剂商前缀 Selleck 剥离"),
        ("MCE HY-10020 5uM", "试剂商前缀 MCE 剥离"),
        ("Alendronate sodium trihydrate", "金属阳离子与水合物剥离"),
        ("Bisindolylmaleimide IX (Ro 31-8220 Mesylate)", "化学盐型与实验室代号剥离"),
        ("Obatoclax Mesylate (GX15-070)", "网罗多层括号代号与甲磺酸盐"),
        ("Patupilone (EPO906, Epothilone B)", "多重同义词消除歧义"),
        ("Sodium Chloride", "无机反离子保护 (防止将食盐误剥成无意义片段)"),
        ("Potassium Chloride", "无机反离子保护"),
        ("DMSO 0.1%", "溶剂阴性对照消歧"),
        ("Saline (vehicle)", "载体阴性对照消歧"),
        ("Vehicle", "空白载体对照消歧")
    ]
    for inp, desc in dirty_formulations:
        r = pa.align([inp], online=False)[0]
        dose_s = f"{r.parsed_dose} {r.parsed_unit}" if r.parsed_dose is not None else "-"
        print(f"    - [输入] {inp:<44} -> [清洗结果] {r.parsed_name:<20} [剂量] {dose_s:<10} [对照] {str(r.is_control):<5} ({desc})")

    # -------------------------------------------------------------
    # 评测 5: 真实复杂基因工程、蛋白降解靶向与跨物种对齐 (Perturb-seq & Degron)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("【评测 5】真实单细胞 CRISPR、突变耐药、靶向降解与跨物种同源对齐")
    print("-" * 75)
    genetic_cases = [
        ("sgTP53", "经典 CRISPR-KO"),
        ("sgMYC_1", "带探针编号 sgRNA"),
        ("dCas9-KRAB:TP53", "CRISPRi 抑制"),
        ("dCas9-VPR:MYC", "CRISPRa 激活"),
        ("shBRCA1", "RNAi 敲低"),
        ("MYC_OE", "质粒过表达"),
        ("EGFR_T790M", "耐药突变体"),
        ("BCR-ABL_T315I", "融合基因耐药突变"),
        ("dTAG-KRAS_G12D", "dTAG 蛋白靶向降解"),
        ("AID-MYC", "生长素诱导降解 Degron"),
        ("sgTrp53", "小鼠同源基因 (Trp53 -> TP53)"),
        ("shBrca1", "小鼠同源基因 (Brca1 -> BRCA1)"),
        ("Non-Targeting sgRNA", "sgRNA 阴性对照"),
        ("sgCtrl_1", "sgRNA 编号对照"),
        ("sgAAVS1", "人类安全港 AAVS1 阴性对照"),
        ("sgRosa26", "小鼠安全港 Rosa26 阴性对照"),
        ("sgCCR5", "安全港位点阴性对照")
    ]
    for inp, desc in genetic_cases:
        e = pa.enrich(inp, online=False)
        r = pa.align([inp], online=False)[0]
        ortho_h = ""
        if e.modality == "genetic":
            orthos = e.properties.get("orthologs", {})
            ortho_h = orthos.get("human", {}).get("symbol") or e.identifiers.get("symbol", "")
        print(f"    - [输入] {inp:<22} -> [模态] {r.modality:<10} [机制] {str(e.properties.get('genetic_action')):<6} [人类同源] {ortho_h:<8} [对照] {str(r.is_control):<5} ({desc})")

    # -------------------------------------------------------------
    # 评测 6: 官方基准协议解析 (LINCS L1000 500 条实测 / PRISM / JUMP)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("【评测 6】高通量官方筛选协议真实数据解析 (LINCS L1000 500条 / PRISM / JUMP-CP)")
    print("-" * 75)
    lincs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "perturbalign_data_archive", "drug_name_corpus", "raw", "lincs_sig_info.txt.gz"))
    with gzip.open(lincs_path, "rt", encoding="utf-8", errors="ignore") as f:
        df_lincs = pd.read_csv(f, sep="\t", nrows=500)
    
    t0 = time.time()
    lincs_res = [pa.adapt_protocol("lincs", row.to_dict(), online=False) for _, row in df_lincs.iterrows()]
    t_lincs = time.time() - t0
    
    l_modalities = {}
    l_ctrls = 0
    l_doses = 0
    for r in lincs_res:
        l_modalities[r.modality] = l_modalities.get(r.modality, 0) + 1
        if r.is_control:
            l_ctrls += 1
        if r.parsed_dose is not None:
            l_doses += 1
            
    print(f"  * LINCS 500 条真实签名解析耗时: {t_lincs*1000:.1f} ms (吞吐量: {500/(t_lincs+1e-6):.1f} 条/秒)")
    print(f"  * 模态分布: {l_modalities}")
    print(f"  * 准确识别对照组: {l_ctrls} 条 (ctl_vehicle / ctl_vector)")
    print(f"  * 成功抽取浓度剂量: {l_doses} 条")

    # PRISM / JUMP
    prism_res = pa.from_prism("BRD-K70511574-001-08-9::2.5::HTS", online=False)
    print(f"  * PRISM 复合标识符解析: Name={prism_res.parsed_name}, Dose={prism_res.parsed_dose} {prism_res.parsed_unit}, Modality={prism_res.modality}")
    
    jump_res = pa.from_jump_cp({"Metadata_JCP2022": "JCP2022_012345", "Metadata_pert_name": "Doxorubicin", "Metadata_pert_type": "compound"}, online=False)
    print(f"  * JUMP-CP 协议解析: Name={jump_res.parsed_name}, Modality={jump_res.modality}, Ctrl={jump_res.is_control}")

    # -------------------------------------------------------------
    # 评测 7: 单细胞 AnnData 容器清洗对齐与 UPDS-v1.0 标准化落盘
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("【评测 7】单细胞 AnnData 容器清洗对齐与 UPDS-v1.0 标准化输出 (10大受控字段)")
    print("-" * 75)
    
    class MockAnnData:
        def __init__(self, obs_df):
            self.obs = obs_df.copy()
            self.uns = {}
            self.obsm = {}
        def copy(self):
            m = MockAnnData(self.obs.copy())
            m.uns = dict(self.uns)
            m.obsm = dict(self.obsm)
            return m
            
    raw_obs_data = [
        {"cell_barcode": "cell_001", "raw_pert": "Doxorubicin HCl 10nM 24h", "cell_line": "K562"},
        {"cell_barcode": "cell_002", "raw_pert": "Trametinib 10nM + Pictilisib 1uM", "cell_line": "A549"},
        {"cell_barcode": "cell_003", "raw_pert": "AZD-6244 500nM", "cell_line": "A549"},
        {"cell_barcode": "cell_004", "raw_pert": "sgTrp53 KO", "cell_line": "NIH3T3"},
        {"cell_barcode": "cell_005", "raw_pert": "dCas9-KRAB:TP53", "cell_line": "K562"},
        {"cell_barcode": "cell_006", "raw_pert": "Hypoxia 1% O2 12h", "cell_line": "HepG2"},
        {"cell_barcode": "cell_007", "raw_pert": "TNF-alpha 50ng/ml", "cell_line": "MCF7"},
        {"cell_barcode": "cell_008", "raw_pert": "sgCtrl (non-targeting)", "cell_line": "K562"},
        {"cell_barcode": "cell_009", "raw_pert": "DMSO 0.1%", "cell_line": "K562"},
        {"cell_barcode": "cell_010", "raw_pert": "Vehicle", "cell_line": "K562"},
    ]
    df_raw = pd.DataFrame(raw_obs_data)
    adata = MockAnnData(df_raw)
    
    std_adata = pa.standardize_anndata(
        adata,
        obs_col="raw_pert",
        cell_line_col="cell_line",
        online=False
    )
    
    cols_to_display = [
        "raw_pert",
        "pert_canonical_name",
        "pert_modality",
        "pert_is_control",
        "pert_dose_norm_um",
        "pert_duration_h",
        "pert_human_ortholog",
        "pert_primary_targets"
    ]
    for idx, row in std_adata.obs[cols_to_display].iterrows():
        dose_norm = f"{row['pert_dose_norm_um']:.4f} uM" if pd.notna(row['pert_dose_norm_um']) else "-"
        dur_norm = f"{row['pert_duration_h']:.1f} h" if pd.notna(row['pert_duration_h']) else "-"
        print(f"    - [Cell {idx+1:02d}] {row['raw_pert']:<34} -> 标准名: {row['pert_canonical_name']:<14} 模态: {row['pert_modality']:<14} 归一浓度: {dose_norm:<12} 归一时间: {dur_norm:<8} 靶点: {row['pert_primary_targets']}")

    audit = std_adata.uns['perturbalign']['audit_summary']
    print(f"\n  * adata.uns 审计总结: 细胞数={audit['n_cells']}, 唯一扰动数={audit['n_unique_perturbations']}, 对照组数={audit['n_controls']}, 模态={audit['modalities']}")
    print("=" * 80)
    print(" 全部 7 大层级真实与冷门多组学扰动数据清洗测评完成！")
    print("=" * 80)

if __name__ == "__main__":
    run_all_evaluations()
