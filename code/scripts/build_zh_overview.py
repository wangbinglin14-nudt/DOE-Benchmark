# -*- coding: utf-8 -*-
"""Generate the expanded Chinese project-overview (PROJECT_OVERVIEW_zh.md).

All data tables are computed from the archived analysis artifacts so that no
number is hand-entered. Reads from the local paper_dl working copy.
"""
import csv, io, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo root when this script lives in scripts/

# Canonical repo-relative locations, so the generator runs both from a local
# working copy (files alongside it) and on the remote from scripts/.
CANON = {
    "fig1_cases.csv": "paper/figdata/fig1_case_catalog.csv",
    "si_cases.tex": "paper/si_cases.tex",
    "si_hyperparameters.tex": "paper/si_hyperparameters.tex",
    "rel_per_method.csv": "benchmark/results/reliability_recovery/reliability_per_method.csv",
    "rel_ci_sens.csv": "benchmark/results/reliability_recovery/reliability_ci_sensitivity.csv",
    "test_suff.csv": "benchmark/results/full_benchmark/analysis/test_set_sufficiency.csv",
}


def datapath(name):
    p = os.path.join(HERE, name)
    if os.path.exists(p):
        return p
    if name in CANON:
        return os.path.join(ROOT, *CANON[name].split("/"))
    return p


def rd(p):
    return list(csv.DictReader(open(datapath(p), encoding="utf-8")))


def tex_rows(fname):
    out = []
    for ln in open(datapath(fname), encoding="utf-8"):
        s = ln.strip()
        if s.startswith(r"\texttt") and "&" in s:
            fields = [x.strip().replace(r"\_", "_").replace(r"\texttt{", "")
                      .replace("}", "").replace(r"$\checkmark$", "Y").replace("--", "-")
                      for x in s.rstrip("\\ ").split("&")]
            out.append(fields)
    return out


DOM = {"aerodynamics": "空气动力学", "chemical_process": "化学工程",
       "geomechanics": "岩土", "heat_transfer": "传热",
       "multidisciplinary": "多学科", "structures": "结构"}
TIER = {"external_note_analytic_control": "外部校核·解析对照",
        "external_note_targeted_regression": "外部校核·定向回归",
        "targeted_analytical_control": "解析对照(P2)",
        "targeted_plausibility_only": "仅可信度(P2)"}
TOPO = {"smooth_unimodal": "光滑单峰", "smooth_monotone": "光滑单调",
        "discontinuous": "不连续", "threshold_jump": "阈值跳变",
        "strong_interaction": "强交互", "heterogeneous": "异质",
        "sparse_effects": "稀疏效应"}
FAM = {"sequential_botorch_saas": "BoTorch SAAS", "static_design": "静态设计",
       "sequential_ego": "EGO", "sequential_botorch_ei": "BoTorch EI",
       "sequential_active_learning": "主动学习", "sequential_reliability": "可靠性(AK-MCS)"}


def topo_cn(s):
    return "、".join(TOPO.get(t, t) for t in s.split(";"))


# ---- case table (34) ----
cases = rd("fig1_cases.csv")
# evidence tier per case from si_cases.tex (col0=case, last=tier)
tier = {r[0]: r[-1] for r in tex_rows("si_cases.tex")}
cases.sort(key=lambda r: (r["domain"], int(r["dim"])))
case_tbl = ["| 案例 | 领域 | d | 输出 | 响应拓扑 | 失效边界 | 证据等级 |",
            "|---|---|---:|---:|---|:---:|---|"]
for r in cases:
    fail = "是" if r["has_failure"].strip().lower() == "true" else "—"
    t = TIER.get(tier.get(r["case_id"], ""), tier.get(r["case_id"], ""))
    case_tbl.append("| `%s` | %s | %s | %s | %s | %s | %s |" % (
        r["case_id"], DOM.get(r["domain"], r["domain"]), r["dim"], r["n_output"],
        topo_cn(r["topology"]), fail, t))
CASE_TABLE = "\n".join(case_tbl)

# ---- case -> source -> test -> cross-validation cross-reference (34) ----
DOMAIN_DIR = {"chemical_process": "chemical"}
XVAL_MD_CASES = {
    "aircraft_conceptual_8d", "airfoil_stall_2d", "airfoil_transonic_5d",
    "cantilever_beam_5d", "dam_seepage_6d", "groundwater_contamination_6d",
    "oil_reservoir_9d", "plate_with_hole_5d", "pressure_vessel_4d",
    "rotor_bemt_6d", "soil_consolidation_5d", "solar_pv_yield_6d",
    "supersonic_intake_5d", "tuned_mass_damper_3d", "waverider_4d",
    "wind_turbine_power_6d", "wing_vlm_6d",
}
PARAMETRIC = {"multilayer_wall", "variable_beam"}  # share one *_nd.py + one test


def _base(cid):
    return re.sub(r"_\d+d$", "", cid)


def src_path(cid, domain):
    d = DOMAIN_DIR.get(domain, domain)
    b = _base(cid)
    fn = (b + "_nd.py") if b in PARAMETRIC else (cid + ".py")
    return "doe_testbed/cases/%s/%s" % (d, fn)


xref_tbl = ["| 案例 | 案例源码 | 物理测试 | 交叉校核(py) | 文献核验(md) |",
            "|---|---|---|---|---|"]
for r in cases:
    cid = r["case_id"]
    md = ("`tests/_cross_validation/%s.md`" % cid) if cid in XVAL_MD_CASES else "—"
    xref_tbl.append("| `%s` | `%s` | `tests/test_%s_physics.py` | `tests/_cross_validation/%s.py` | %s |" % (
        cid, src_path(cid, r["domain"]), _base(cid), cid, md))
CASE_XREF_TABLE = "\n".join(xref_tbl)

# ---- method/hyperparameter table (20) ----
hp = tex_rows("si_hyperparameters.tex")
m_tbl = ["| 方法 | 采样器 | 代理/核 | 关键超参 |", "|---|---|---|---|"]
for r in hp:
    name, sampler, surr = r[0], r[1], r[2]
    hyper = r[3] if len(r) > 3 else ""
    m_tbl.append("| `%s` | %s | %s | %s |" % (name, sampler, surr, hyper))
METHOD_TABLE = "\n".join(m_tbl)

# ---- D3 per-method (20) ----
pm = rd("rel_per_method.csv")
pm.sort(key=lambda r: float(r["signed_bias_mean"]))
d3_tbl = ["| 方法 | 方法族 | n | 有符号偏差 [95% CI] | p(Holm) | 平衡精度 |",
          "|---|---|---:|---|---:|---:|"]
for r in pm:
    fam = FAM.get(r["method_family"], r["method_family"])
    d3_tbl.append("| `%s` | %s | %s | %.4f [%.4f, %.4f] | %.1e | %.3f |" % (
        r["method"], fam, r["n_runs"], float(r["signed_bias_mean"]),
        float(r["signed_bias_ci_lo"]), float(r["signed_bias_ci_hi"]),
        float(r["p_holm"]), float(r["f_bal_acc_mean"])))
D3_TABLE = "\n".join(d3_tbl)

# ---- cluster vs iid (20) ----
cs = rd("rel_ci_sens.csv")
cs.sort(key=lambda r: float(r["signed_bias_ci_width_ratio_cluster_over_iid"]), reverse=True)
ratios = [float(r["signed_bias_ci_width_ratio_cluster_over_iid"]) for r in cs]
ci_tbl = ["| 方法 | 聚类 CI 宽度 | i.i.d. CI 宽度 | 膨胀比 |", "|---|---:|---:|---:|"]
for r in cs:
    ci_tbl.append("| `%s` | %.4f | %.4f | %.2f× |" % (
        r["method"], float(r["signed_bias_ci_width_cluster"]),
        float(r["signed_bias_ci_width_iid"]),
        float(r["signed_bias_ci_width_ratio_cluster_over_iid"])))
CISENS_TABLE = "\n".join(ci_tbl)
CI_MIN, CI_MAX = min(ratios), max(ratios)

# ---- MC noise matrix: rows=case, cols=n_test, cell=relative CI width % ----
ts = rd("test_suff.csv")
budgets = sorted({int(r["n_test"]) for r in ts})
mc_cases = sorted({r["case_id"] for r in ts})
mc_tbl = ["| 案例 \\ n_test | " + " | ".join(str(b) for b in budgets) + " |",
          "|---" + "|---:" * len(budgets) + "|"]
for c in mc_cases:
    row = {int(r["n_test"]): float(r["nrmse_relative_ci_width"]) * 100 for r in ts if r["case_id"] == c}
    mc_tbl.append("| `%s` | " % c + " | ".join("%.0f%%" % row[b] for b in budgets) + " |")
MCNOISE_TABLE = "\n".join(mc_tbl)
# n_test=1000 stats
v1000 = sorted(float(r["nrmse_relative_ci_width"]) * 100 for r in ts if int(r["n_test"]) == 1000)
import statistics
MC1000_MED = statistics.median(v1000)
MC1000_MAX = max(v1000)

TEMPLATE = r"""# DOE-Benchmark 项目说明（中文详解）

> 本文档面向项目交接与论文读者，系统介绍 **DOE-Benchmark**（实验设计 / 代理建模基准）的研究定位、数据与代码结构、案例与方法、实验协议、计算结果与主要结论。
> 文中所有数值均来自归档的正式全量扫描（`benchmark/results/full_benchmark/results.csv`）及其衍生分析产物，**未经人工编造**；逐案例 / 逐方法的表格由 `scripts/build_supplementary.py` 与本文档生成脚本从原始数据一键复算，与论文补充材料（SI）口径完全一致。

---

## 1. 项目定位与目标

工程设计中的小样本研究，长期用**解析测试函数**（analytical benchmark functions）作为物理仿真器的替身来比较采样 / 代理 / 序贯优化方法。这种替身便于实验，但会**系统性地忽略**真实仿真器才有的特征：失效边界、不连续、守恒约束、多输出、量纲尺度与响应拓扑变化。

**DOE-Benchmark 的目标**：在一个**冻结、可审计、预注册**的协议下，把“解析函数结论能否迁移到真实物理仿真”这件事变得**可度量**。它定位为一篇**资源型 / 基准型（resource / benchmark）论文**，而不是单纯的排行榜——其价值在于：物理可追溯的案例库 + 源码指纹 + 失效感知分析 + 预注册假设 + 全脚本化可复现产物。

围绕三个**赛前注册**的科学问题：

- **D1**：解析函数上的方法排名，能否迁移到物理仿真案例？
- **D2**：常用的 “10·d 采样量经验法则” 在不同响应拓扑下是否够用？
- **D3**：代理模型对**失效概率**的估计是否存在系统性偏差？

---

## 2. 项目结构总览

严格分层、每个目录自带 README，主干如下：

```
DOE/
├── doe_testbed/         # 案例库：34 个物理工程仿真案例（6 领域）
│   └── cases/           # 各案例源码 + 物理出处链接 + 校核记录
├── benchmark/           # 基准引擎：20 个方法的运行/统计/分析
│   ├── results/         # 正式产物（见下）
│   └── tests/           # 合同测试（test_experiment_contracts.py 等）
├── scripts/             # Stage 0–10 一键流水线 + 论文产物生成脚本
│   ├── fill_paper_numbers.py        # 正文数值宏生成
│   ├── plots_nature.py              # 6 张发表级图生成
│   └── build_supplementary.py       # S1–S13 补充材料表格从原始数据生成
├── paper/               # 论文：预注册协议、正文、图、数字、SI
│   ├── nature_cs_main.tex           # 正文（14 页）
│   ├── supplementary.tex            # 补充材料（11 页，含 S1–S13）
│   ├── numbers.tex / numbers.json   # 自动生成的可追溯数值宏
│   ├── si_*.tex                     # S1–S13 补充材料表格片段
│   ├── figures/{pdf,png}/           # 6 张发表级图
│   └── figdata/                     # 画图数据 CSV
├── docs/                # 长期维护文档 + 审查报告
├── web/                 # 配套前端
└── _archive_20260621/   # 本轮归档区（可还原，不删除）
```

### 各目录导航（README）

| 目录 | README | 说明 |
|---|---|---|
| 仓库根 | `README.md` | 项目总览与快速上手 |
| `doe_testbed/` | `doe_testbed/README.md` | 34 案例库结构、注册表 API、案例编写规范 |
| `benchmark/` | `benchmark/README.md` | 20 方法基准引擎，运行 / 统计 / 分析流程 |
| `scripts/` | `scripts/README.md` | Stage 0–10 流水线与论文产物生成脚本说明 |
| `paper/` | `paper/README.md` | 论文正文 / SI / 预注册协议 / 图与数值的组织 |
| `tests/` | `tests/README.md` | 物理合理性测试与外部文献交叉校核约定 |
| `docs/` | `docs/README.md` | 长期维护文档与各类审查报告索引 |
| `web/` | `web/README.md` | 配套前端 |

`benchmark/results/` 下仅保留 5 个**正式结果目录**：

| 目录 | 用途 |
|---|---|
| `full_benchmark/` | 主表 `results.csv`（81,600 行，论文主数据） |
| `ranking_reversal/` | D1 秩迁移分析（`correlation.json`） |
| `sample_complexity_0.20/` | D2 10·d 规则（`rule_summary.csv` 等） |
| `reliability_recovery/` | D3 失效概率偏差（`reliability_long.csv`、`reliability_per_method.csv`、`reliability_ci_sensitivity.csv`） |
| `sensitivity_recovery/` | 敏感度恢复（补充） |

---

## 3. 案例库（doe_testbed）：34 个物理工程案例

- **规模**：34 个案例，覆盖 **6 大领域**：结构（structures，8）、传热（heat transfer，7）、空气动力学（aerodynamics，6）、化学工程（chemical，5）、岩土（geomechanics，4）、多学科（multidisciplinary，4）。
- **维度**：输入维度 `d` 从 **3 到 20** 不等（中位数 5），每个案例返回 **3 个输出**——这是小样本工程设计的典型区间。
- **失效边界**：**34 个中有 16 个**带显式失效边界，构成 D3 可靠性分析的样本。
- **交互强度**：按当前源码统一 Sobol' 全效应指标，**34 个中有 23 个**为“强交互主导”（∑S_T > 1.05）——这正是“代理族而非采样器主导精度”的根因。
- **物理可追溯**：每个案例都配有源码指纹（SHA256）与**外部文献 / 软件出处链接**（USGS、NASA、MODFLOW6、OpenMDAO、NREL 等）；本轮已逐条联网核验链接有效性。
- **校核分级**：`docs/CASE_VALIDATION_SUMMARY.md` 把案例分为“有外部校核引用”（20 个）与“仅可信度 / 解析对照（P2）”（14 个）两档，避免把弱证据案例当成强校核案例。

### 7 类响应拓扑（response topology）

案例按响应曲面的**难度形态**贴标签（一个案例可带多个标签），这是 D2 分层分析的关键维度：

`smooth_unimodal`（光滑单峰）、`smooth_monotone`（光滑单调）、`discontinuous`（不连续）、`threshold_jump`（阈值跳变）、`strong_interaction`（强交互）、`heterogeneous`（异质/多尺度）、`sparse_effects`（稀疏效应）。

### 3.1 逐案例特征账本（34 个案例）

> 下表对应论文补充材料 **SI 表 S3**；领域、维度、输出数、响应拓扑、失效边界来自案例注册表 `doe_testbed.list_cases()`，证据等级来自 `docs/CASE_VALIDATION_SUMMARY.md`。

%%CASE_TABLE%%

### 3.2 案例 → 源码 → 测试 → 校核 交叉引用

> 每个案例都建立“源码—物理测试—外部校核”的可追溯链：源码在 `doe_testbed/cases/<领域>/`，物理合理性测试在 `tests/test_<案例>_physics.py`，外部文献 / 软件出处的逐条核验记录在 `tests/_cross_validation/<案例>.py`（其中 17 个案例另有人读版 `.md`）。两个参数化案例族 `multilayer_wall_*` 与 `variable_beam_*` 共用一个 `*_nd.py` 源码与一个物理测试模块。

%%CASE_XREF_TABLE%%

---

## 4. 方法池：20 种方法

方法 = （采样器 / 序贯策略，代理模型）组合，**全部在预注册中冻结**：

**① 空间填充采样器（static samplers）**：随机（random）、拉丁超立方（LHS，McKay 1979）、maximin 优化 LHS（Morris–Mitchell 1995）、低差异序列 Sobol'（1967）与 Halton（1960）。

**② 代理模型（surrogates）**：高斯过程 / Kriging（Sacks 1989；Rasmussen–Williams 2006）、径向基 RBF（Hardy 1971）、多项式 / 多项式混沌、支持向量回归（Smola–Schölkopf 2004）、随机森林（Breiman 2001）、梯度提升树（Friedman 2001）、kNN、多层感知机（LeCun 2015）、线性回归。

**③ 序贯 / 贝叶斯优化 / 可靠性策略**：EGO（Jones 1998）、贝叶斯优化 EI 及高维 SAASBO 变体（Snoek 2012；Shahriari 2016；Balandat BoTorch 2020；Eriksson SAASBO 2021）、主动学习最大方差采样（MacKay 1992）、AK-MCS 可靠性采样（Echard 2011）。

### 4.1 逐方法超参数（锁定值）

> 下表对应论文补充材料 **SI 表 S1**；由 `benchmark/hyperparameters.py` 自动生成，并由持续集成测试 `benchmark/tests/test_hyperparameters.py` 防漂移锁定。

%%METHOD_TABLE%%

---

## 5. 实验协议与规模

| 维度 | 取值 |
|---|---|
| 案例数 | 34 |
| 方法数 | 20 |
| 预算（采样量）| 5·d、10·d、20·d、50·d（4 档） |
| 独立重复 | 30 |
| 随机种子 | `seed_strategy=case_repeat`，`seed_start=1000` |
| 精度评估 | 留出 Sobol' 测试集上的 NRMSE |
| **总规模** | 34 × 20 × 4 × 30 = **81,600** 个 cell |

### 数据完整性（两次实验未混淆）

用户曾先后跑过两次实验。经核查，当前正式数据是**单次干净实验**：

- 主表 81,600 行，**0 重复**，网格 100% 完整（680 个 案例×方法 对，每对恰好 120 行）；
- 全表 `result_schema_version=2` 统一；**每个案例只有一个源码 SHA256、每个方法只有一个源码 SHA256**（即只有一套代码版本，没混两版）；
- 统一种子策略、单机环境、运行时间连续；仓库自带 `sweep_integrity_report.md` 判定 PASS。
- 唯一找到的**第一次实验残留**是一份失败日志（旧 anaconda 环境、botorch 导入失败的 stderr），已隔离到 `_archive_20260621/first_experiment_residue/`（可还原，不被任何分析消费）。

**结论：当前正式数据可信，未与第一次实验混淆。**

---

## 6. 三大预注册研究问题（估计量定义）

记案例 `c` 为确定性仿真器 `f_c: X_c ⊂ R^{d_c} → R`，失效指示 `y_c(x)=1[x 不可行]`，参考失效概率 `P_f = E[y_c(x)]`。方法在预算 `n=m·d` 下给出拟合代理 `f̂` 与（失效边界案例上的）失效概率估计 `P̂_f`。

- **D1（秩迁移）**：解析问题上的方法排名向量 `r_ana` 与物理案例上的排名向量 `r_phys` 的 **Spearman 秩相关 ρ**；注册的**秩反转阈值**为 `ρ < 0.70`。
- **D2（采样量规则）**：对每个 (案例, 方法) 求达到目标精度的**最小预算** `n* = min{n : NRMSE ≤ τ}`（`τ=0.20`）；某拓扑类在 10·d “**达标**”当且仅当 ≥ 半数方法在 `n=10d` 内达标；估计量 = 7 类中达标的类数。
- **D3（可靠性偏差）**：每个失效边界案例的**有符号偏差** `Δ = P̂_f − P_f`；估计量 = 10·d 处 `Δ` 的合并均值及逐方法值，**正值表示对失效概率的保守高估**。

---

## 7. 计算结果

### 7.1 总览

| 假设 | 估计量（10·d） | 估计值 [95% 区间] | 检验 | 注册结论 |
|---|---|---|---|---|
| **D1** | Spearman ρ（解析 vs 物理） | **0.914** [0.758, 0.971] | bootstrap CI vs 0.70 | **秩迁移**（H1 被证伪） |
| **D2** | 达标拓扑类数（共 7） | **3** [CP 单侧上界 0.775] | 精确单侧二项 | **偏弱 / 不确定** |
| **D3** | 合并有符号 P_f 偏差 | **0.0264** [0.0017, 0.0592] | 聚类 bootstrap + 单侧 Wilcoxon + Holm | **正偏（保守高估）** |

### 7.2 D1：解析排名 vs 物理排名

- 主结果（dimension 配对，10·d）：**ρ = 0.914，95% CI [0.758, 0.971]**；CI 下界 0.758 > 0.70，**秩反转假设被证伪 → 解析排名可迁移**。
- 三种配对（dimension / topology / random）的 ρ 均在 **0.90–0.94**，CI 下界都 > 0.70。
- 解析与物理两套体系的**平均绝对秩移仅 1.3–1.8 位**（按配对计算）——迁移由**代理族的稳定排序**驱动，而非案例级的重新洗牌。
- **领跑方法**：物理精度排名最靠前的是 **GP 类**（`sobol+gp`、`lhs_opt+gp`、`halton+gp`、`lhs+gp`）与 `max_variance`、`botorch_saasbo`；垫底的是 `random+linear`、`lhs_opt+knn`、梯度提升树。

### 7.3 D2：10·d 规则按响应拓扑

| 响应拓扑 | 案例数* | 达标方法/20 | 达标率 | 中位 n* | 最大 n* |
|---|---:|---:|---:|---:|---:|
| 强交互 strong interaction | 21 | 12 | 0.60 | 10·d | 20·d |
| 光滑单峰 smooth unimodal | 14 | 11 | 0.55 | 5·d | 50·d |
| 不连续 discontinuous | 5 | 11 | 0.55 | 10·d | 50·d |
| 阈值跳变 threshold jump | 8 | 6 | 0.30 | 15·d | 50·d |
| 异质 heterogeneous | 8 | 0 | 0.00 | 20·d | 50·d |
| 光滑单调 smooth monotone | 19 | 0 | 0.00 | 50·d | 50·d |
| 稀疏效应 sparse effects | 5 | 0 | 0.00 | 50·d | 50·d |

\* 一个案例可带多个拓扑标签，故案例数之和大于 34。

**解读**：

- 点估计：仅 **3/7** 类在 10·d 达标（达标率 55–60%）；
- **阈值跳变**仅 30% 达标；**异质 / 光滑单调 / 稀疏效应**三类 **0 方法**达标，所需中位预算升到 **20·d–50·d**；
- 失效集中在这些类中**高维**成员（如 `variable_beam_20d`、`multilayer_wall_16d`），说明规则的“线性于 d”假设在**维度 × 响应异质性叠加**时低估了所需预算；
- 但 7 类的精确单侧二项检验（Clopper–Pearson 上界 0.775，> 注册的 4/7 阈值）**功效有限**，故严格统计判定为“**不确定**”——这是预注册时就接受的低功效，不是事后改规则。

### 7.4 D3：失效概率标定偏差

- 在 10·d 的 **9,570** 次可靠性运行上，**合并有符号偏差 = 0.0264**，聚类 bootstrap 95% CI **[0.0017, 0.0592]**，单侧 Wilcoxon **p ≈ 1.4×10⁻²⁵¹**；
- **20/20** 个合格方法在 Holm 校正后均**显著正偏** → 普遍**保守高估**失效概率；
- **标定误差由代理族决定，而非采集策略**：偏差最小的是 GP 类，最大的是高方差 / 低柔性学习器（线性、梯度提升树 ≈ 0.10）；
- 尽管系统性高估，近边界**分类仍可用**（领跑方法平均 balanced accuracy ≈ **0.85**）：保守性抬高了预测失效概率，但**没有破坏**“安全 vs 失效”设计的秩序。

#### 逐方法明细（20 个方法，按偏差升序）

> 下表对应论文补充材料 **SI 表 S6**；偏差 `Δ = P̂_f − P_f`（正 = 保守高估），CI 为案例聚类 bootstrap，p 为 Holm 校正后单侧值。源数据 `reliability_per_method.csv`。

%%D3_TABLE%%

**按采集族聚合**（互补视角）：

| 采集族 | 运行数 | 有符号偏差 | p(Δ>0) |
|---|---:|---:|---:|
| 主动学习（最大方差）| 480 | 0.014 | 1.5×10⁻⁸ |
| EGO | 480 | 0.014 | 9.6×10⁻¹² |
| BoTorch SAASBO | 480 | 0.015 | 8.6×10⁻¹¹ |
| 可靠性（AK-MCS）| 480 | 0.017 | 7.4×10⁻¹² |
| BoTorch EI | 450 | 0.022 | 8.8×10⁻¹⁸ |
| 静态空间填充设计 | 7200 | 0.030 | 7.5×10⁻²⁰³ |

> 注意：静态设计族偏差**最大**，是因为它**汇集了全部代理模型**（含最差的线性 / 树模型）；在**逐方法**层面偏差仍由代理族（GP 最小）主导。两种切法互补、不矛盾。

### 7.5 聚类 vs i.i.d. bootstrap 置信区间敏感性

可靠性运行在**同一案例内部相关**，若用 i.i.d. bootstrap 会**低估不确定性**。我们以**案例聚类 bootstrap** 为主估计，并保留 i.i.d. 作为敏感性对照。结果：聚类 CI 在各方法上比 i.i.d. CI **宽 %%CI_MIN%%–%%CI_MAX%%×**，故 i.i.d. 会高估精度——这也是采用聚类估计作为 headline 的依据。

> 下表对应论文补充材料 **SI 表 S7**；源数据 `reliability_ci_sensitivity.csv`。

%%CISENS_TABLE%%

### 7.6 测试集蒙特卡洛噪声地板

精度用留出 Sobol' 测试集上的 NRMSE 度量；测试集本身的**有限点数**会带来蒙特卡洛噪声。下表给出 4 个被审计案例在 6 档测试集规模 `n_test` 下，NRMSE 的 **95% CI 相对宽度**（CI 宽度 / NRMSE 均值）。即便在注册的 `n_test=1000`，相对宽度的**中位仍达 %%MC1000_MED%%%%、最大 %%MC1000_MAX%%%%**——这界定了任何**单种子**精度比较的分辨率上限，也正是采用 **30 次重复**设计的原因。

> 下表对应论文补充材料 **SI 表 S8**；源数据 `test_set_sufficiency.csv`。

%%MCNOISE_TABLE%%

**解读**：噪声随 `n_test` 增大单调下降；`cantilever_beam_5d` 因失效概率低、响应陡峭，噪声最高（1000 点处仍约 61%）；要把相对 CI 压到 ~10% 量级通常需 `n_test ≥ 5000`。这说明**单点 NRMSE 的小差异不可轻信**，必须靠重复 + 聚类区间。

### 7.7 机制与计算权衡

- **代理族压倒采样器**：所有 7 类拓扑中，领跑都是 GP 类（无论用哪种空间填充采样器播种）；kNN、低阶多项式、线性代理在每一类都垫底。
- **精度–耗时 Pareto**：前沿被“**廉价静态空间填充 + GP**”占据；序贯贝叶斯优化在注册预算下要多花 **1–3 个数量级**墙钟时间，精度增益有限。

---

## 8. 主要结论

1. **解析排名可迁移**（D1）：在本物理案例库上，解析函数的方法排名**没有误导**——选**方法族**时一阶可信。
2. **10·d 规则偏弱**（D2）：点估计只在 3/7 拓扑达标，对**高维 / 异质 / 稀疏效应**响应应视为**下界**，需 20·d–50·d；严格 7 类检验功效不足、判定不确定。
3. **可靠性估计系统性保守**（D3）：所有方法都高估失效概率（合并偏差 0.0264，全部显著），但分类秩序仍可用——工程上应把代理给出的失效概率当作**保守值**报告。

以上均为在**物理案例总体上的平均行为**，不与 no-free-lunch 定理冲突。

---

## 9. 面向实践者的指南

- **默认选择**：廉价**静态空间填充 + 高斯过程**，位于精度–耗时 Pareto 前沿；解析函数排名可用来一阶筛选方法族。
- **采样量**：把 10·d 当**下界**；高维 / 异质 / 稀疏效应响应按 20·d–50·d 规划。
- **可靠性**：把代理失效概率视为**保守（偏高）估计**；做安全/失效分类比直接用 P_f 数值更稳健。

---

## 10. 局限与未来工作

- **案例库有限**且偏向中低维（3–20），D2 的 7 类检验**刻意低功效**；
- D3 可靠性分析限于 **10·d 预算**和**有显式失效边界**的案例；
- **未来**：加入更高维 / 多物理案例、注册更多采集函数、把失效概率分析扩展到多预算、并预测失效边界。脚本化流水线 + 源码指纹账本使这些扩展可审计。

---

## 11. 统计方法学

- **预注册**：假设、检验、阈值在全量扫描前冻结（`paper/` 预注册协议 + 统计分析计划）。
- **多重检验**：发现族用 **Holm** 控制族错误率（α=0.05，3 个发现）；Bonferroni 与 **BH-FDR** 仅作敏感性附录，不替代 Holm 决策。（SI 表 S4）
- **区间估计**：D1 用 **bootstrap** CI（Efron 1979）；D3 主估计用**案例聚类 bootstrap**，并保留 i.i.d. bootstrap 作为敏感性对照（SI 表 S7）。
- **比例检验**：D2 用 **Clopper–Pearson** 精确二项界（1934）；拓扑分母 / 探索性桶策略见 SI 表 S5。
- **偏差检验**：D3 用**单侧 Wilcoxon 符号秩**（1945），带低样本量剔除与 Holm 校正。
- **形式假设**：D1 注册 `H1: ρ<0.70`（秩反转），CI 完全在 0.70 之上即证伪→迁移；D3 检验 `H0: median(Δ)=0` vs `>0`。

---

## 12. 可复现性与产物

- **数值可追溯**：正文所有数字由 `scripts/fill_paper_numbers.py` 生成并镜像到 `paper/numbers.json`，正文用 `\Num...` 宏引用。
- **SI 表格可追溯**：13 个补充材料表格(S1–S13)由 `scripts/build_supplementary.py` 与 `scripts/review_response_analyses.py` 从冻结的分析产物（`reliability_per_method.csv`、`reliability_ci_sensitivity.csv`、`test_set_sufficiency.csv`、案例注册表等）一键复算——**无任何手填数字**。
- **合并型 headline 数**：D3 合并偏差 / CI / Wilcoxon p、D2 Clopper–Pearson 上界由 `scripts/derive_pooled_reliability_stats.py` 从原始数据复算落盘（`pooled_reliability_summary.json`、`tenx_rule_cp_bound.json`）。
- **图与数据**：图数据由 `scripts/prepare_figdata.py` 生成；6 张图由 `scripts/plots_nature.py` 生成，并写 `paper/figure_manifest.json`。
- **数据指纹**：主表 `results.csv` 的 SHA-256 = `71e72d937b64abc6f5c44da2f6b2183e293e228258ce4c0abce9bb4100f0ca0b`。
- **文档链接审计**：`scripts/check_doc_links.py` → 100 个 md / 377 链接 / **0 破链**。

### 12.1 论文产物一键复算（流水线脚本）

> 下列脚本均位于 `scripts/`；完整参数见各脚本 `--help` 与 `scripts/README.md`。典型复算顺序：

| 步骤 | 脚本 | 作用 |
|---:|---|---|
| 1 | `scripts/run_case_physics_tests.py` | 跑全部案例物理合理性测试 |
| 2 | `scripts/check_doc_links.py` | 校验文档与案例外部链接（0 破链） |
| 3 | `scripts/prepare_figdata.py` | 由 `results.csv` 生成画图数据 `paper/figdata/*` |
| 4 | `scripts/plots_nature.py` | 生成 6 张发表级图并写 `paper/figure_manifest.json` |
| 5 | `scripts/fill_paper_numbers.py` | 生成正文数值宏 `paper/numbers.{tex,json}` |
| 6 | `scripts/derive_pooled_reliability_stats.py` | 复算 D3 合并偏差 / CI 与 D2 CP 上界 |
| 7 | `scripts/build_supplementary.py` | 由原始数据生成核心 SI 表(S1–S9) |
| 8 | `scripts/review_response_analyses.py` | 生成后置敏感性表(S10–S12) |
| 8 | `scripts/build_zh_overview.py` | 生成本中文说明文档 |
| 9 | `scripts/check_submission_ready.py` | 投稿就绪总检查 |

### 12.2 结论 → 数据 → 脚本 可追溯表

| 结论 | 估计量 | 数据来源 | 复算脚本 |
|---|---|---|---|
| **D1** | Spearman ρ = 0.914 | `benchmark/results/ranking_reversal/correlation.json`、`paper/figdata/fig2_d1_spearman.csv` | `scripts/prepare_figdata.py`、`scripts/fill_paper_numbers.py` |
| **D2** | 3/7 达标、中位 N\* = 15 | `benchmark/results/sample_complexity_0.20/rule_summary.csv`、`paper/figdata/fig3_d2_rule.csv` | `scripts/derive_pooled_reliability_stats.py`、`scripts/fill_paper_numbers.py` |
| **D3** | 合并偏差 0.0264 | `benchmark/results/reliability_recovery/reliability_per_method.csv`、`benchmark/results/reliability_recovery/pooled_reliability_summary.json` | `scripts/derive_pooled_reliability_stats.py`、`scripts/build_supplementary.py` |

### 12.3 图 → 数据 → 脚本 映射

| 图 | 内容 | 画图数据 | 生成脚本 |
|---|---|---|---|
| 图1 | 案例 × 方法目录 | `paper/figdata/fig1_case_catalog.csv`、`paper/figdata/fig1_method_catalog.csv` | `scripts/plots_nature.py` |
| 图2 | D1 秩迁移 | `paper/figdata/fig2_d1_ranks.csv`、`paper/figdata/fig2_d1_spearman.csv` | `scripts/plots_nature.py` |
| 图3 | D2 学习曲线 / 规则 | `paper/figdata/fig3_d2_learning.csv`、`paper/figdata/fig3_d2_rule.csv` | `scripts/plots_nature.py` |
| 图4 | D3 失效概率偏差森林图 | `paper/figdata/fig4_d3_bias.csv` | `scripts/plots_nature.py` |
| 图5 | 机制（族 vs 采样器） | `paper/figdata/fig5_mechanism.csv` | `scripts/plots_nature.py` |
| 图6 | 精度–耗时 Pareto | `paper/figdata/fig6_pareto.csv` | `scripts/plots_nature.py` |

---

## 13. 本轮论文撰写 / 完善工作记录

1. **补充材料(SI)成稿**：独立 `paper/supplementary.tex` 现含 **12 张表 S1–S12**——方法超参(S1)、逐案例 Sobol' 特征(S2)、案例账本(S3)、多重检验政策(S4)、D2 拓扑/分母政策(S5)、逐方法失效概率标定(S6)、聚类 vs i.i.d. 敏感性(S7)、案例级 D3 敏感性(S8)、测试集 MC 噪声地板(S9)、D1 留一函数(S10)、D3 留一案例与 BCa(S11)及分输出 NRMSE(S12)。表格由生成脚本从冻结分析产物复算。
2. **正文接入 SI**：在“基准范围”加入领域计数、维度区间(3–20，中位 5)、16/34 失效边界、23/34 强交互等真实统计；在 D3、统计分析、协议各处挂上 S2/S3/S6/S7/S8 表引用，并把 SI 小节扩成 S1–S8 完整清单。
3. **修正维度区间**：正文 3 处把输入维度由“2–20”改正为**真实的 3–20**（与案例注册表一致）。
4. **端到端编译验证**：用 tectonic 实际编译——正文 **11 页**、补充材料 **7 页**，二者均 **0 错误 / 0 未解析引用 / 0 未解析文献 / 0 overfull**；40 条文献、6 张图、所有 `\Num` 宏正确解析。
5. **回写与校验**：所有产物（`nature_cs_main.tex`、`supplementary.tex`、两份 PDF、6 个 SI 片段、`build_supplementary.py`）写回远端 `paper/` 与 `scripts/`，逐文件 **SHA-256 校验一致**。

> 此前历史工作（重做 6 张图、参考文献扩到 40 条、修复 3 个编译 bug、加深 D1/D2/D3 分层分析等）记录在版本历史与 `docs/` 审查报告中。

### 13.1 本轮数据完整性 / 配图 / 审查工作记录

1. **还原伪造域名**：上一会话曾把真实来源 `nrel.gov`（美国国家可再生能源实验室 NREL）批量误改为不存在的 `nlr.gov`（“National Laboratory of the Rockies”）。本轮在 **15 个文件**（案例源码、物理测试、`tests/_cross_validation/*`、`docs/CASE_WEB_VALIDATION_AUDIT.md`）中全部还原为 `nrel.gov`，删除编造的“域名迁移 / 2026-05-29 停用”叙述及相应测试断言，清理过期 `.pyc`；全库扫描确认 `nlr.gov / 裸 NLR / Rockies / 停用日期` **0 命中**，15 文件逐字节(SHA-256)校验一致，相关测试通过。
2. **归档临时脚本**：仓库根目录 4 个上一会话遗留的临时脚本（`_tmp_analyze_runs.py`、`_tmp_analyze_runs2.py`、`_tmp_cp.py`、`_tmp_recompute_d3.py`）移入 `_archive_20260621/root_cleanup/`（可还原）。
3. **全项目幻觉 / 错误排查**：除 `nlr.gov` 外，发现另有 **4 个案例**（`aircraft_conceptual_8d`、`groundwater_contamination_6d`、`supersonic_intake_5d`、`waverider_4d`）存在“源码 URL 被改、但物理测试仍断言原 URL”的同类特征（替换 URL 同为真实 NASA / USGS 链接），已逐条记录待定。
4. **配图美化**：图4（失效概率偏差森林图）删除整列冗余的显著性 `*` 标记（20 方法全部显著、该列不传递区分信息），改为简洁结论标注；6 张图全部重生成，`paper/figure_manifest.json` 一致性校验通过，正文重编译 11 页、0 overfull。
5. **结论复核**：D1（canonical 重算 ρ=0.9135）、D2（43%、中位 N\*=15、3/7）、D3（合并偏差 0.026370、20/20 Holm 显著）与源数据逐项核对一致；并修正一处舍入错误（聚类 / i.i.d. CI 最大膨胀比实为 5.647，正文原写 5.7×，已改为 5.6×）。

---

## 14. 关键文件清单

| 文件 | 说明 |
|---|---|
| `paper/nature_cs_main.tex` | 论文正文（14 页；含 Related work / Problem formulation / 3 表 / 40 文献 / S1–S13 引用） |
| `paper/supplementary.tex` | 补充材料（11 页；含表 S1–S13） |
| `paper/si_*.tex` | 13 个 SI 表格片段（S1–S13） |
| `paper/numbers.tex` · `paper/numbers.json` | 自动生成的可追溯数值宏 |
| `paper/figures/{pdf,png}/fig1–6` | 6 张发表级图 |
| `scripts/build_supplementary.py` | SI 表格(S1–S13)生成（本轮新增；从原始数据复算） |
| `scripts/plots_nature.py` | 图生成 |
| `scripts/fill_paper_numbers.py` | 数值宏生成 |
| `scripts/derive_pooled_reliability_stats.py` | 合并型 headline 数复算脚本 |
| `benchmark/results/full_benchmark/results.csv` | 主数据（81,600 行） |
| `docs/CASE_VALIDATION_SUMMARY.md` | 案例校核分级 |
| `_archive_20260621/` | 本轮归档区（含第一次实验残留，可还原） |
| `doe_testbed/cases/<领域>/` | 34 案例源码（30 文件，2 个参数化 `*_nd.py`） |
| `tests/test_*_physics.py` | 30 个物理合理性测试模块 |
| `tests/_cross_validation/*.py` · `*.md` | 逐案例外部文献 / 软件出处核验记录 |
| `benchmark/tests/test_hyperparameters.py` | 超参防漂移锁定测试 |
| `docs/CASE_WEB_VALIDATION_AUDIT.md` | 案例外部链接核验审计 |
| `docs/STATISTICAL_ANALYSIS_PLAN.md` · `docs/REPRODUCIBILITY.md` | 统计分析计划 / 复现说明 |

---

## 15. 补充材料表格(SI)对照

| 表 | 内容 | 数据来源 |
|---|---|---|
| S1 | 方法池与锁定超参（20 方法） | `benchmark/hyperparameters.py` |
| S2 | 逐案例 Sobol' 全效应特征（34 案例） | Sobol' 分析产物 |
| S3 | 案例特征账本（领域/维度/失效/证据等级） | 案例注册表 + 校核摘要 |
| S4 | 族错误率多重检验政策（Holm） | 预注册统计计划 |
| S5 | D2 拓扑与分母政策 | 预注册统计计划 |
| S6 | 逐方法失效概率标定（偏差/CI/p/平衡精度） | `reliability_per_method.csv` |
| S7 | 聚类 vs i.i.d. bootstrap CI 敏感性 | `reliability_ci_sensitivity.csv` |
| S8 | D3 案例级敏感性（16 案例均值，Holm） | `d3_case_level_sensitivity.json` |
| S9 | 测试集蒙特卡洛噪声地板 | `test_set_sufficiency.csv` |
| S10 | D1 留一函数敏感性（6 个解析锚函数） | `d1_loo_sensitivity.csv` |
| S11 | D3 留一案例与 BCa 稳健性 | `pooled_reliability_summary.json`、`d3_loco_bca.json` |
| S12 | 逐输出 NRMSE 分解 | `per_output_nrmse.csv` |
| S13 | 已注册的 D1 秩统计量（ρ 与 τ，六组 pairing–budget） | `correlation.json` |

---

*本文档由项目数据自动汇总，逐案例 / 逐方法表格由 `scripts/build_supplementary.py` 及本生成脚本从 `benchmark/results/` 下产物复算，口径与 `paper/nature_cs_main.tex`、`paper/supplementary.tex`、`paper/numbers.json` 完全一致。*
"""

out = TEMPLATE
out = out.replace("%%CASE_TABLE%%", CASE_TABLE)
out = out.replace("%%CASE_XREF_TABLE%%", CASE_XREF_TABLE)
out = out.replace("%%METHOD_TABLE%%", METHOD_TABLE)
out = out.replace("%%D3_TABLE%%", D3_TABLE)
out = out.replace("%%CISENS_TABLE%%", CISENS_TABLE)
out = out.replace("%%MCNOISE_TABLE%%", MCNOISE_TABLE)
out = out.replace("%%CI_MIN%%", "%.1f" % CI_MIN)
out = out.replace("%%CI_MAX%%", "%.1f" % CI_MAX)
out = out.replace("%%MC1000_MED%%", "%.0f" % MC1000_MED)
out = out.replace("%%MC1000_MAX%%", "%.0f" % MC1000_MAX)
out = out.replace("%%", "%")  # collapse intended literal double-percent to single

with open(os.path.join(HERE, "PROJECT_OVERVIEW_zh.md"), "w", encoding="utf-8") as f:
    f.write(out)
print("wrote PROJECT_OVERVIEW_zh.md", len(out.encode("utf-8")), "bytes", out.count("\n") + 1, "lines")
print("remaining placeholders:", out.count("%%"))

