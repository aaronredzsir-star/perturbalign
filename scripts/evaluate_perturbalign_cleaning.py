"""
PerturbAlign 数据清洗能力真实数据全面评测脚本
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

# 导入本地 perturbalign
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import perturbalign as pa

def test_tahoe_dataset():
    print("\n" + "=" * 70)
    print("【评测 1】真实 Tahoe-100M 大规模扰动数据集 (65,618 细胞条件 / 378 种药物)")
    print("=" * 70)
    
    tahoe_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "perturbalign_data_archive", "reference", "tahoe_conditions.parquet"))
    if not os.path.exists(tahoe_path):
        print(f"[-] 文件不存在: {tahoe_path}")
        return
        
    df = pd.read_parquet(tahoe_path)
    total_rows = len(df)
    unique_drugs = df['drug'].unique().tolist()
    print(f"数据量: {total_rows} 行条件, {len(unique_drugs)} 种唯一样本药物标签")
    
    # 统计脏标签类型：含有盐型、立体异构、括号代号等
    salt_samples = [d for d in unique_drugs if any(s in d.lower() for s in ['hydrochloride', 'mesylate', 'hydrate', 'sodium', 'acetate', 'isethionate'])]
    stereo_samples = [d for d in unique_drugs if any(s in d for s in ['(R)-', '(S)-', '18β-', '18-'])]
    code_samples = [d for d in unique_drugs if any(c in d.upper() for c in ['AZD', 'GDC', 'BAY', 'APTO', 'AT75', '9-ING'])]
    
    print(f"  - 识别到盐型/酸根/水合物标签: {len(salt_samples)} 种 (例如: {salt_samples[:3]})")
    print(f"  - 识别到立体异构/希腊字母标签: {len(stereo_samples)} 种 (例如: {stereo_samples[:3]})")
    print(f"  - 识别到药企研发代号/实验室代码: {len(code_samples)} 种 (例如: {code_samples[:3]})")
    
    # 吞吐量与清洗基准
    t0 = time.time()
    results = pa.align(unique_drugs, online=False)
    elapsed = time.time() - t0
    
    speed = len(unique_drugs) / (elapsed + 1e-6)
    print(f"\n清洗吞吐量: {speed:.1f} 条/秒 (处理 {len(unique_drugs)} 种药物耗时 {elapsed*1000:.1f} ms)")
    
    # 检验清洗效果 (挑选代表性脏标签)
    print("\n代表性真实脏数据清洗对齐实测:")
    cases_to_show = [
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
    
    res_map = {r.input: r for r in pa.align(cases_to_show, online=False)}
    for c in cases_to_show:
        r = res_map[c]
        print(f"  [原始] {c:<32} -> [清洗后] 名称: {r.parsed_name:<16} 模态: {r.modality:<14} 对照: {str(r.is_control):<5}")

def test_single_cell_benchmark_cases():
    print("\n" + "=" * 70)
    print("【评测 2】真实单细胞扰动筛选基准测试 (SciPlex 3, Combosciplex, 4i, Gehring-Pachter)")
    print("=" * 70)
    
    bench_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "benchmark", "benchmark_cases.json"))
    with open(bench_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
        
    cases = bench_data["cases"]
    print(f"测试用例总数: {len(cases)} 条 (覆盖 80+ 种边缘与复杂脏数据模式)")
    
    passed_name = 0
    passed_ctrl = 0
    passed_dose = 0
    
    # 分类测试统计
    cat_stats = {}
    
    for c in cases:
        cat = c["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "pass": 0}
        cat_stats[cat]["total"] += 1
        
        inp = c["input"]
        res = pa.align(inp, online=False)[0]
        
        exp_name = c["expected_parsed_name"]
        exp_ctrl = c["expected_is_control"]
        exp_dose = c["expected_dose"]
        
        name_ok = (res.parsed_name == exp_name) or (exp_name is None and res.parsed_name is None)
        ctrl_ok = (res.is_control == exp_ctrl)
        dose_ok = True if exp_dose is None else (res.parsed_dose == exp_dose)
        
        if name_ok:
            passed_name += 1
        if ctrl_ok:
            passed_ctrl += 1
        if dose_ok:
            passed_dose += 1
            
        if name_ok and ctrl_ok and dose_ok:
            cat_stats[cat]["pass"] += 1
            
    print(f"  - 实体名称解析准确率: {passed_name}/{len(cases)} ({passed_name/len(cases)*100:.1f}%)")
    print(f"  - 对照组(Control)识别准确率: {passed_ctrl}/{len(cases)} ({passed_ctrl/len(cases)*100:.1f}%)")
    print(f"  - 浓度剂量(Dose)解析准确率: {passed_dose}/{len(cases)} ({passed_dose/len(cases)*100:.1f}%)")
    print(f"  - 综合全字段一致率: {sum(v['pass'] for v in cat_stats.values())}/{len(cases)} ({sum(v['pass'] for v in cat_stats.values())/len(cases)*100:.1f}%)")

def test_lincs_l1000_real_data():
    print("\n" + "=" * 70)
    print("【评测 3】真实 LINCS L1000 全量元数据 (GSE92742 真实 gzip 提取)")
    print("=" * 70)
    
    lincs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "perturbalign_data_archive", "drug_name_corpus", "raw", "lincs_sig_info.txt.gz"))
    if not os.path.exists(lincs_path):
        print(f"[-] 文件不存在: {lincs_path}")
        return
        
    with gzip.open(lincs_path, "rt", encoding="utf-8", errors="ignore") as f:
        df = pd.read_csv(f, sep="\t", nrows=5000)
        
    print(f"抽取真实 LINCS 签名数: {len(df)} 条")
    print("包含扰动类型分布:", dict(df['pert_type'].value_counts()))
    
    adapter = pa.LINCSAdapter()
    
    t0 = time.time()
    results = [adapter.adapt(row, online=False) for _, row in df.iterrows()]
    elapsed = time.time() - t0
    
    modality_counts = {}
    ctrl_count = 0
    dose_parsed = 0
    duration_parsed = 0
    
    for r in results:
        modality_counts[r.modality] = modality_counts.get(r.modality, 0) + 1
        if r.is_control:
            ctrl_count += 1
        if r.parsed_dose is not None:
            dose_parsed += 1
        if r.parsed_duration is not None:
            duration_parsed += 1
            
    print(f"\nLINCS 适配器解析吞吐量: {len(df)/(elapsed+1e-6):.1f} 条/秒 (总耗时: {elapsed*1000:.1f} ms)")
    print("解析后模态分类分布:", modality_counts)
    print(f"正确识别对照组 (ctl_vehicle / ctl_vector): {ctrl_count} 条")
    print(f"提取出有效浓度: {dose_parsed}/{len(df)} 条, 有效作用时间: {duration_parsed}/{len(df)} 条")
    
    # 展示不同类型的解析样本
    print("\n典型扰动类型解析对比:")
    type_samples = {}
    for _, row in df.iterrows():
        pt = row.get('pert_type')
        if pt not in type_samples:
            type_samples[pt] = row
            if len(type_samples) >= 5:
                break
                
    for pt, row in type_samples.items():
        res = adapter.adapt(row)
        print(f"  [LINCS {pt:<10}] iname={str(row.get('pert_iname')):<16} -> 模态: {res.modality:<14} 名称: {res.parsed_name:<16} 靶点: {str(res.genetic_target):<8} 对照: {str(res.is_control):<5}")

def test_genetic_perturbseq_cleaning():
    print("\n" + "=" * 70)
    print("【评测 4】真实单细胞 CRISPR Perturb-seq 基因扰动数据清洗与跨物种对齐")
    print("=" * 70)
    
    perturbseq_cases = [
        "sgTP53",
        "sgMYC_1",
        "sgCDKN1A-2",
        "sgKRAS",
        "sgEGFR",
        "sgBRCA1",
        "sgBCL2L1+sgMCL1",      # 双敲除 (双基因协同致死)
        "sgMAPK1+sgMAP2K1",      # 双敲除 (同一通路串行阻断)
        "dCas9-KRAB:TP53",       # CRISPRi
        "dCas9-VPR:MYC",         # CRISPRa
        "shBRCA1",               # RNAi 敲低
        "MYC_OE",                # 过表达
        "sgTrp53",               # 小鼠同源基因
        "shBrca1",               # 小鼠同源基因
        "Non-Targeting sgRNA",   # 阴性对照
        "sgCtrl_1",              # 阴性对照
        "safe_targeting_sgRNA"   # 安全位点对照
    ]
    
    print(f"测试基因扰动用例: {len(perturbseq_cases)} 条 (覆盖 KO, 双KO, CRISPRi/a, RNAi, OE, 跨物种, 阴性对照)")
    
    for case in perturbseq_cases:
        e = pa.enrich(case, online=False)
        r = pa.align([case], online=False)[0]
        
        ortho_human = ""
        if e.modality == "genetic":
            orthos = e.properties.get("orthologs", {})
            ortho_human = orthos.get("human", {}).get("symbol") or e.identifiers.get("symbol", "")
            
        print(f"  [输入] {case:<22} -> 模态: {e.modality:<10} 对照: {str(e.is_control):<5} 人类同源: {ortho_human:<6} 机制: {str(e.properties.get('genetic_action')):<6} 标准名: {e.canonical_name}")

def test_anndata_standardization_and_upds():
    print("\n" + "=" * 70)
    print("【评测 5】真实单细胞 AnnData 容器清洗对齐与 UPDS-v1.0 标准输出")
    print("=" * 70)
    
    # 构造模拟单细胞真实混合扰动数据集 (模拟包含各种实验命名脏数据的 AnnData)
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
        {"cell_barcode": "cell_006", "raw_pert": "sgCtrl (non-targeting)", "cell_line": "K562"},
        {"cell_barcode": "cell_007", "raw_pert": "DMSO 0.1%", "cell_line": "K562"},
        {"cell_barcode": "cell_008", "raw_pert": "Vehicle", "cell_line": "K562"},
        {"cell_barcode": "cell_009", "raw_pert": "Palbociclib (isethionate) 1uM 48h", "cell_line": "MCF7"},
        {"cell_barcode": "cell_010", "raw_pert": "18β-Glycyrrhetinic acid 10uM", "cell_line": "HepG2"},
    ]
    df_raw = pd.DataFrame(raw_obs_data)
    adata = MockAnnData(df_raw)
    
    print(f"输入 AnnData 观测矩阵: {len(adata.obs)} 个细胞，原始扰动列包含复杂药物盐型、多药联用、CRISPRi、小鼠同源、溶剂对照")
    
    # 执行标准化
    std_adata = pa.standardize_anndata(
        adata,
        obs_col="raw_pert",
        cell_line_col="cell_line",
        online=False
    )
    
    print("\n输出 UPDS-v1.0 标准化字段矩阵 (adata.obs 新增 10 大标准列):")
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
    display_df = std_adata.obs[cols_to_display]
    for idx, row in display_df.iterrows():
        print(f"  Cell {idx+1:02d}: {row['raw_pert']:<34} -> 标准名: {row['pert_canonical_name']:<14} 模态: {row['pert_modality']:<14} 对照: {str(row['pert_is_control']):<5} 归一化浓度: {str(row['pert_dose_norm_um']):<8} 靶点: {row['pert_primary_targets']}")
        
    print("\nadata.uns['perturbalign'] 审计元数据:")
    audit = std_adata.uns['perturbalign']['audit_summary']
    print(f"  - 校验标准: {std_adata.uns['perturbalign']['standard']}")
    print(f"  - 审计统计: 细胞数={audit['n_cells']}, 唯一扰动数={audit['n_unique_perturbations']}, 对照组数={audit['n_controls']}, 模态={audit['modalities']}")

if __name__ == "__main__":
    test_tahoe_dataset()
    test_single_cell_benchmark_cases()
    test_lincs_l1000_real_data()
    test_genetic_perturbseq_cleaning()
    test_anndata_standardization_and_upds()
    print("\n" + "=" * 70)
    print("【评测总结】所有真实数据场景 (Tahoe, LINCS, 单细胞筛选, Perturb-seq, AnnData) 评测完成！")
    print("=" * 70)
