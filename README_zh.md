# PerturbAlign 🧬

[English](README.md) | [中文](README_zh.md)

**PerturbAlign** 是一个面向单细胞扰动组学（Perturb-seq、Sci-Plex、Mix-seq、L1000）与多组学因果知识体系的扰动数据清洗、实体对齐、因果知识富集与内嵌图数据库存储工具。

核心能力：
1. **扰动元数据标准化**：将实验中杂乱的原始标签（包含盐型、溶剂、浓度、时间、商品名、药企研发代号等）清洗消歧为符合 **UPDS (Unified Perturbation Data Schema v1.0)** 规范的受控字段。
2. **因果调控知识富集**：将扰动作用靶点连接到下游**激酶 $\to$ 转录因子 (TFs) $\to$ DoRothEA 权威响应基因群**的真实因果链，并支持宿主细胞系突变背景特异性剪枝。
3. **内嵌 SQLite 图数据底座**：内置 ACID 事务性图数据库（`SQLiteGraphStore`），支持 SQL 递归通用表表达式 (Recursive CTE) 进行毫秒级多跳因果路径推断，免去复杂外部图服务运维。
4. **多模态连续表征与对齐生态**：原生支持基于 ESM2 蛋白质语言模型与 ChemBERTa 分子结构的连续深度嵌入；内置可插拔 Provider 架构，无缝适配 AnnData / MuData 单细胞生态与下游深度学习管线。

> 📖 **完整交互式教程**：请参阅配套的 Jupyter Notebook 实例：[notebooks/quickstart_tutorial.ipynb](notebooks/quickstart_tutorial.ipynb)。

---

## 1. 支持的扰动类型与输入形态

### 1.1 扰动类型（Modalities）

PerturbAlign 依据生物学机制将扰动严格归类为以下受控模态：

| 扰动模态 (`pert_modality`) | 说明与支持的命名形式 | 典型输入示例 | 标准化目标 |
| :--- | :--- | :--- | :--- |
| **小分子化合物**<br>`small_molecule` | 通用名 (INN)、商品名、化学结构 (SMILES)、药企研发代号 (Investigational codes) 及试剂商编号 | `Doxorubicin HCl 10nM 24h`<br>`AZD-6244 1uM`<br>`Selleck-S1001` | 剥除盐型与前缀，消歧药企代号，统一输出通用名（如 `Selumetinib`）、SMILES 与 HGNC 靶点 |
| **遗传扰动**<br>`crispr_ko` / `crispr_a` / `crispr_i`<br>`rnai` / `overexpression` | CRISPR 基因敲除/激活/抑制、RNAi 敲低、质粒过表达；支持跨物种基因符号 | `sgTrp53 KO`<br>`MYC_OE`<br>`shBRCA1` | 识别操作机制（KO/a/i/OE/shRNA），提取目标基因，跨物种对齐至人类 HGNC Symbol（`TP53`） |
| **生物大分子与细胞因子**<br>`biologic` | 重组细胞因子、白介素、干扰素、单克隆抗体（`-mab`）、受体融合蛋白（`-cept`） | `TNF-alpha 50ng/ml`<br>`IL-6 100ng/ml`<br>`Trastuzumab` | 规范命名，匹配对应细胞膜表面受体（如 `TNFRSF1A`）与下行信号通路 |
| **微环境与物理胁迫**<br>`environmental` | 氧浓度、温度、pH 值、营养匮乏等细胞培养环境胁迫 | `Hypoxia 1% O2 12h`<br>`Heat shock 42C`<br>`Glucose starvation` | 归一化至 MSigDB Hallmark 8 大经典应激体系，关联环境感应转录因子（如 `HIF1A`, `HSF1`） |
| **联合扰动**<br>`combination` | 多药联用、复方鸡尾酒或双基因敲除 | `Trametinib + Pictilisib`<br>`sgCDKN1A + sgTP53` | 依据安全分隔符规则拆解组分，分别解析并构建协同调控图谱 |
| **阴性对照**<br>`control` | 溶剂对照、空白载体、非靶向 sgRNA 对照 | `DMSO 0.1%`<br>`Vehicle`<br>`sgCtrl (non-targeting)` | 标记为对照（`pert_is_control=True`），基线特征置零 |

### 1.2 支持的数据结构

- **`AnnData`**：单细胞标准容器，自动从 `adata.obs` 提取扰动列，并将标准列与 `adata.uns["perturbalign"]` 知识命名空间写回；
- **Pandas `Series` / `DataFrame`**：常规表格数据，支持批量向量化处理；
- **Python 列表 / 字符串**：单条或批量扰动名称。

---

## 2. 扰动数据清洗与对齐（外置词典体系）

生命科学实验命名存在大量历史遗留差异与噪音。PerturbAlign 将所有清洗、消歧与判定规则解耦为 **22 份透明、可独立维护的离线 TSV 词典**，避免硬编码黑盒：

### 2.1 22 份外置词典分工

```
resources/
├── 命名消歧与前缀剥离
│   ├── investigational_codes.tsv  # 180+ 药企研发代号 (AZD, GDC, BMS, ABT, PD, PF 等) 映射通用名与权威 SMILES
│   ├── vendor_prefixes.tsv        # 28 家商业化学试剂商前缀 (Selleck, MCE, Tocris, Sigma 等) 自动剥离
│   └── formulation_codes.tsv      # 药品剂型与管制代码 (XR, SR, CIV) 剥离
├── 化学去盐与反离子保护
│   ├── salt_suffixes.tsv          # 盐型/酸根后缀 (HCl, mesylate, tartrate, sodium 等) 识别与安全剥除
│   ├── metal_cations.tsv          # 金属阳离子盐前缀识别 (Sodium, Potassium 等)
│   ├── inorganic_compounds.tsv    # 无机化合物反离子保护 (避免将 NaCl、KCl 误剥成无意义残片)
│   ├── stereo_tokens.tsv          # 立体化学标记 ((R)-, (S)-, cis-, trans-) 规范化
│   └── greek_transliteration.tsv  # 希腊字母规范转写 (α -> alpha, β -> beta)
├── 遗传扰动机制与对照识别
│   ├── genetic_actions.tsv        # 遗传操作机制词汇 (KO, CRISPRa, CRISPRi, RNAi, OE) 识别
│   ├── genetic_control_targets.tsv# 遗传安全阴性对照靶点 (NT, Scramble, AAVS1, Luciferase) 识别
│   ├── sgrna_reference.tsv        # sgRNA 命名与前缀参考表
│   └── control_tokens.tsv         # 溶剂与载体对照 (DMSO, Vehicle, Untreated, Water 等) 识别
├── 生物药、递送与微环境
│   ├── biologics_tokens.tsv       # 细胞因子与生物大分子词元表 (TNF, IL-6, Interferon 等)
│   ├── biologic_suffixes.tsv      # 生物药 INN 词尾 (-mab 单抗, -cept 融合蛋白) 识别
│   ├── delivery_vehicles.tsv      # 递送载体 (LNP, Liposome, Electroporation) 识别
│   ├── environmental_tokens.tsv   # 8 大物理/微环境应激词表 (Hypoxia, Starvation, Acidosis 等)
│   └── protac_e3_ligands.tsv      # PROTAC 降解剂与 E3 连接酶配体识别
└── 时空量纲与联用拆解
    ├── dose_units.tsv             # 浓度量纲与单位换算系数 (nM, uM, mM, M, mg/ml)
    ├── duration_units.tsv         # 时间量纲与单位换算系数 (h, hr, min, d, days)
    ├── combo_separators.tsv       # 联合用药安全分隔符 (+, /, ;, AND) 质量门规则
    ├── cocktails.tsv              # 经典多药鸡尾酒方案 (如 R-CHOP, FOLFOX) 识别
    └── perturbation_protocols.tsv # 公开扰动协议元数据分类 (LINCS L1000, scPerturb)
```

### 2.2 对齐输出：UPDS (Unified Perturbation Data Schema v1.0) 规范

经 `perturbalign.standardize_anndata(adata)` 处理后，会在 `adata.obs` 中回填 10 大标准列：

1. `pert_canonical_name`：标准化官方名称（全大写标准词，如 `SELUMETINIB`, `TP53`）
2. `pert_modality`：受控枚举（`small_molecule`, `crispr_ko`, `crispr_a`, `crispr_i`, `rnai`, `overexpression`, `biologic`, `environmental`, `combination`, `control`）
3. `pert_is_control`：布尔值标记是否为对照（`True` / `False`）
4. `pert_dose_val`：原始解析的数值
5. `pert_dose_unit`：原始剂量单位
6. `pert_dose_norm_um`：**严格统一换算为微摩尔 ($\mu\text{M}$)**（支持 nM, mM 自动无损换算）
7. `pert_duration_h`：**严格统一换算为小时数 ($h$)**（支持 min, d 自动无损换算）
8. `pert_human_ortholog`：**跨物种对齐的人类 HGNC Symbol**（小鼠 `Trp53` $\to$ `TP53`）
9. `pert_primary_targets`：分号分隔的标准靶点列表（如 `MAP2K1;MAP2K2`）
10. `pert_smiles`：立体无歧义标准 SMILES 结构式

---

## 3. 多维知识富集与因果传导网络

清洗对齐后，PerturbAlign 将离散标签接入真实生命科学知识网络，推导扰动的生物学因果链路。

### 3.1 整合的知识库

- **化学与药理**：PubChem（理化参数/SMILES）、DrugBank（靶点/MoA）、ChEMBL（临床期别/适应症）、openFDA（不良反应）、BindingDB（亲和力常数）；
- **基因与通路**：NCBI Gene（Entrez ID/染色体）、Reactome（信号通路）、STRING（蛋白质互作 PPI）、DepMap（CRISPR 依赖性得分）；
- **因果调控与宿主背景**：DoRothEA（Level A/B 权威转录因子-靶基因 Regulons）、Cellosaurus / DepMap（细胞系驱动突变与遗传背景）。

### 3.2 整合机制：因果信号传导链路 (Causal Cascade)

扰动对细胞状态的改变并非黑盒。PerturbAlign 构建了从直接作用靶点到下游转录输出的因果调控链：

$$\text{直接扰动靶点} \longrightarrow \text{激酶/受体信号通路} \longrightarrow \text{核心汇聚转录因子 (TFs)} \longrightarrow \text{DoRothEA 标志响应基因群 (Regulons)}$$

### 3.3 宿主细胞系突变背景剪枝 (Context-Specific Pruning)

不同细胞系遗传背景不同，相同的扰动会产生截然不同的下游响应。PerturbAlign 支持宿主细胞系上下文感知：

- **TP53 缺陷剪枝**：在 **TP53 缺失/突变的细胞系**（如 `K562`, `Jurkat`, `HL-60`）中，若使用 DNA 损伤类药物（如 Doxorubicin / TOP2A 抑制剂），系统自动将失效的 `TP53` 分支从活跃转录因子中剪枝，移入 `pruned_tfs`，从响应基因集中剔除 p53 独占调控靶基因，并将置信度下调至 `0.65 (Context-Attenuated)`；
- **驱动突变吻合增强**：当靶点与宿主驱动基因吻合时（如 A549 肺癌背景下抑制 MEK/KRAS 通路），置信度相应提升至 `0.97`。

---

## 4. 快速上手

### 4.1 基本代码示例

```python
import scanpy as sc
import perturbalign

# 1. 单细胞 AnnData 标准化 (输出 UPDS 10 大标准列)
adata = sc.read_h5ad("raw_screen.h5ad")
adata = perturbalign.standardize_anndata(
    adata,
    obs_col="perturbation",      # 原始扰动列
    cell_line_col="cell_line",   # 宿主细胞系列 (如 K562, A549)
    online=False,                # 离线环境优先走本地外置词典与缓存
)

# 2. 推导因果传导网络并进行细胞系剪枝
cascade = perturbalign.infer_causal_cascade(["TOP2A"], cell_line="K562")
print("剪枝旁路分支:", cascade["pruned_tfs"])            # 标注 TP53 因突变被剪枝
print("因果置信度得分:", cascade["cascade_confidence_score"]) # 0.65

# 3. 药企研发代号纯离线解析
drug = perturbalign.query_compound("AZD-6244", online=False)
print("通用名:", drug["pubchem"]["canonical_name"])      # Selumetinib
print("SMILES:", drug["pubchem"]["canonical_smiles"])    # 标准立体结构式

# 4. 跨物种同源基因映射
gene = perturbalign.query_gene("Trp53", organism="mouse")
print("人类同源基因:", gene["ncbi"]["symbol"])            # TP53
```

### 4.2 内嵌图数据库与 SQL 因果路径遍历
```python
# 构建或加载知识图谱至内嵌 SQLite 图数据库
kg = perturbalign.HeterogeneousKnowledgeGraph(db_path="perturbation_kg.sqlite")

# 利用 SQL 递归通用表表达式 (CTE) 极速查询多跳因果调控路径
paths = kg.find_paths_sql("Compound:Trametinib", "Gene:MAP2K1", max_hops=2)
for p in paths:
    print(p.format_path())
```

### 4.3 教程 Notebook

更多完整端到端实战（包括内嵌 SQLite 图数据库操作、多跳因果路径检索等），请直接在 Jupyter 环境运行：
```bash
jupyter notebook notebooks/quickstart_tutorial.ipynb
```

---

## 5. 热插拔与微内核可扩展架构 (Pluggable Architecture)

PerturbAlign 在各核心层均引入解耦的注册表抽象机制，便于无缝热插拔私有模型、企业自建数据库、新型实验 Assay 与自定义流水线：

### 5.1 深度表征模型热插拔 (`register_embedding_provider`)
灵活拔插或替换扰动嵌入模型（如 Uni-Mol、MolFormer、Geneformer、scGPT 或自建表征）：
```python
from perturbalign import BaseEmbeddingProvider, register_embedding_provider, MODALITY_SMALL_MOLECULE

class UniMolProvider(BaseEmbeddingProvider):
    name = "unimol"
    def embed(self, target: str, model=None, cache=None, **kwargs) -> list[float]:
        return custom_unimol_encoder(target)

register_embedding_provider(MODALITY_SMALL_MOLECULE, UniMolProvider())
```

### 5.2 动态实验表型提供器 (`register_profile_provider`)
接入新型实验技术谱图（如 DIA 蛋白质组学、单细胞 ATAC 染色质开放特征或私有筛选结果）：
```python
from perturbalign import BaseProfileProvider, register_profile_provider

class ProteomicsProvider(BaseProfileProvider):
    assay_name = "proteomics"
    def get_profile(self, target: str, cache=None, **kwargs):
        return {"assay": "proteomics", "target": target, "hits": ["AKT1", "MTOR"]}

register_profile_provider("proteomics", ProteomicsProvider(), aliases=["dia_proteomics"])
```

### 5.3 通用领域词典与研发代号注册器 (`register_vocabulary`)
任何领域词表（研发代号 `investigational_codes`、供应商前缀 `vendor_prefixes`、对照组词条 `control_tokens`、盐型后缀 `salt_suffixes`、生物制剂 `biologics_tokens` 或自定义私有实体表）均可直接通过通用词典注册器动态扩展或覆盖（支持 TSV 文件路径或内存 dict/set/list）：
```python
from perturbalign import register_vocabulary, get_vocabulary, resolve_investigational_code, align

# 1. 注册/覆盖自研药物代号字典
register_vocabulary("investigational_codes", {
    "CUSTOM-999": {
        "code": "CUSTOM-999",
        "canonical_name": "Antigravib",
        "targets": ["BRAF", "KRAS"],
        "class_name": "Kinase Inhibitor",
        "developer": "InHouse",
    }
})
# 对齐与代号解析引擎立即可用
res = resolve_investigational_code("CUSTOM 999")
assert res["canonical_name"] == "Antigravib"

# 2. 扩展对照组词条或供应商前缀
register_vocabulary("vendor_prefixes", ["MYVENDOR-", "CHEM-"])

# 3. 注册任意自定义业务词典
register_vocabulary("my_disease_indications", {"AML": "Acute Myeloid Leukemia"})
```

### 5.4 表观调控元件、因果网络与安全性评价扩展
- **顺式调控元件 (cCRE)**：`register_ccre(gene, ccre)` 注册自定义增强子与启动子元件
- **信号通路与 Regulons**：`register_target_cascade(kinase, tfs, pathway)` 与 `register_dorothea_regulon(tf, targets)`
- **宿主模型与安全性靶点**：`register_cell_line(entity)`、`register_safety_target(symbol, profile)` 与 `register_safety_compound(drug, profile)`
- **参考转录签名**：`register_reference_signature(drug, signature_dict)`

---

## 6. 生态架构与选型

- **`perturbalign`**（主包）：全功能包，包含 AnnData 特征化、基于真实预训练权重的连续深度嵌入、异构因果网络推断与内嵌 SQLite 图数据存储（`SQLiteGraphStore`）。
- **`perturbalign-core`**（子包）：**零重量级依赖**，仅依赖 Python 3.9+ 标准库与 22 份离线 TSV 词典，适合高吞吐生产微服务与数据清洗 ETL。
- **AI 智能体支持 (MCP Server)**：提供标准 Model Context Protocol 服务，支持通过 `perturbalign mcp` 命令一键启动，供 Cursor、Claude Desktop 等 LLM 工具直接调用。

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源发布。

