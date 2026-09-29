# Chaord 审核指南（供远程评审 · 评审 2 计划完成后更新版）

日期：2026-09-29 · 分支 master · 已全部推送
仓库：https://github.com/WhiteCrosstheRiver/Chaord

---

## 一、当前状态（一句话）

**v0.1 原型，验收 14/14 全过（真实物理重建、零 build 失败）。** 四个阶段：M0-M8 功能开发 → 诚实核心（独立参考数据 + 严格验收器 + 独立验证）→ FAIL 修复 → **评审 2 两波八流（T1-T8）+ 五个 A5 修复包**。所有数字来自最新一次全量验收（`reports/acceptance.json`），每个修复都有修前失败的复现测试，两项判定学权衡如实记录在 gate 报告（非隐藏）。

## 二、四轮进展总览

| 阶段 | 验收 | 关键变化 |
|---|---|---|
| 第一轮（M0-M8） | 自称 14/14 | 外部评审指出验证循环：CI 失败、无序帧非物理、验收器弱于 PLAN |
| 第二轮（诚实核心） | 9/14 | 独立 MD 参考数据（已发表势）；验收器逐字对齐 PLAN + 变异测试；独立验证者报告；真守恒；阈值卫生 |
| 第三轮（FAIL 修复） | 12/14 | A2/A3 固溶体（SRO 显著性 + 不变性）；A5 LJ 案例（physics=True）；A9 大帧；CSL 直方图法 |
| **第四轮（评审 2：两波八流 + A5 修复）** | **14/14** | A4 热缺陷 22/22；A5 六案例 6/6 全过零 build 失败；ASE 分子后端（已发表势）；参考数据 v2；段优先管线 stage 1（字节等价） |

## 三、14/14 逐条结果（最新全量验收）

| ID | 结果 | 要点 |
|---|---|---|
| A1 语法与格式 | ✅ | 10 示例全过；10,000 生成程序幂等 |
| A2 不变性 | ✅ | 9/9（含随机固溶体）；18/18 刚体变换字节一致 |
| A3 精确往返 | ✅ | 9/9 文本字节一致 + 结构拟合 |
| **A4 缺陷恢复** | **✅ 新过** | **22/22 格 P=R=1.000**（受约束热淬火：LJ+谐和位点弹簧的最小化后再做 Wigner-Seitz；此前 18/22、最差 0.09） |
| **A5 统计往返** | **✅ 新过** | **6/6 带地板案例全过、12/12 重建零失败**：liquid ×0.6/×1.0、liquid_large(2048) ×0.6/×1.1、solid_liquid ×1.4/×1.3、glass ×0.6/×0.7、nacl_aq ×1.4/×1.0、water ×0.9/×0.9 |
| A6 守恒 | ✅ | 126/126 三方计数+电荷精确 |
| A7 相分割 | ✅ | 10/10 帧 ≥95%（最差 0.999） |
| A8 反应 census | ✅ | 2 个独立构造案例精确 |
| A9 压缩比 | ✅ | 1372 原子帧，最差比 0.49%（限 2%） |
| A10 确定性 | ✅ | 重复 lift 字节一致；同 seed build 坐标一致 |
| A11 速度 | ✅ | 10 万原子 lift 3.7 秒（限 120 秒） |
| A12 静态检查 | ✅ | 四类种子错误全捕获 |
| A13 无崩溃 | ✅ | 125/125 帧零异常 |
| A14 文档 | ✅ | 35/35 方言键覆盖（新增 `model` 键有示例） |

## 四、本轮（评审 2）修复详情——审核重点

### Wave 1（T1-T4，已推送 76f4eff）

| 流 | 内容 |
|---|---|
| T1 验收诚实化 | A5 逐行记录 T/backend/md_steps；physics_off 变异；玻璃地板诚实披露（跨淬火地板 + 慢测试） |
| T2 精确 join-count | Cliff-Ord 矩精确零假设方差（替代 bootstrap），68-168× 加速，10 次重排字节稳定 |
| T3 Σ5 公度双晶 | 公度盒（5a×5a×6a，520 原子）零重叠；Σ13/Σ17 也对 |
| T4 无静默跳过 | 未知 backend 抛错；玻璃无 history → 方言默认协议并记录为假设 |

### Wave 2（T5-T8）+ 五个 A5 修复包

**T5/T6（ASE 后端 + 参考数据 v2）**：TIP4P/JC 离子/表格 EAM 的 ASEBackend；玻璃 15 帧（3 次独立淬火，地板 0.165）；水/NaCl 帧距 5/10 ps；新增 2048 原子 lj_liquid_large。

**T7 热缺陷（A4 18/22→22/22）**：根因——位点容差锚在热帧最小近邻距（热碰撞+植入间隙原子把它压低），~40 个真实热位移原子被误读为 frenkel 对。修复：受约束能量最小化（自由最小化被实验证伪：金刚石/岩盐在单 σ LJ 下重构）；容差改锚拟合晶格自身 d_NN。修前 P=0.10/0.75 的两个复现测试在案。

**T8 管线 stage 1**：`lift/pipeline.py` 七阶段编排 + 快路径，9/9 帧与 legacy 字节一致，`mode="pipeline"` 可用。

**A5 修复包 1（多物种界面，P1）**：三个根因——(a) `is_single_phase` 的 M3 分子捷径把 Cu 板吞成伪分子 `Cu384`；(b) 薄液膜统计窗为空 → `np.concatenate([])` 崩溃；(c) metal 方言无 slab 阈值。修复：键图连通分量规则（新方言键）；薄膜 margin/rmax 比例降级（程序注释可见）；metal 0.2.4 补 11 键（lj 语义×Cu 尺度换算表在 docs）；多物种 decompile + build（逐区域弛豫，无已发表 Cu-水跨项——记录在案的决定）。

**A5 修复包 2（已发表势覆盖，P3）**：Ar（Hansen-Verlett 1969）/N₂（Murthy-Singer 1980 双位点）LJ 进方言 + 通用 LJ 流体计算器；TIP4P 核向量化（与 ase.calculators.tip4p <1e-8 等价）+ 耦合 Newton SHAKE：nacl 重建 450s→46s。

**A5 修复包 3（平衡协议，P4）**：RSA 起构的抹除是扩散过程，t_mix∝N^(2/3)：2400 步够 N=500、N=2048 需 ≥6144。方言 md 表 7400 步（实测步数-cn_tv 曲线钉住；md_steps 元数据=实跑步数）。

**A5 修复包 4（水模型一致性，P5）**：参考是刚性 SPC/E（Berendsen 1987）+JC 离子，重建却用 TIP4P 几何 → 分子内 g(r) 峰错位 ×46。语言新增 `model <name>` 物理语句（lift 按守恒的中位 O-H 键长分类）；SPC/E 按参考自己的截断方案实现（LJ 移位 + Wolf/DSF，Fennell-Gezelter 2006）；nacl ×46→×1.0。

### 两项判定学权衡（gate_a_prime.md "Recorded trade-offs"，审核时应专门看）

1. **分子内排除**：刚性分子的分子内 g(r) 峰是几何常数；SPC/E 的 O-H=1.0000 Å 恰在 bin 边界（参考帧自己的 1080 条键就因浮点舍入掷硬币分落两 bin），判据数学上不可满足。分子内排除（方言门控，partial g(r) 早有同款决定）；地板两侧同定义重算（非分子案例逐字节不变）；**几何错误改由更强 的精确测试钉死**（`test_solution_model.py`：lift 必含 `model spce`、重建 O-H 中位数必 ≈1.0000）；四个分子金丝雀仍全红。
2. **金属-水界面逐区域弛豫**：无已发表 Cu-水跨项势，界面不跨弛豫（代码注释+provenance 记录）。cu_water 是合成无地板案例（6-11s 完成构建，诚实跳过评分）。

## 五、建议审核路线

### P0 —— 数字与不弱化

1. **跑一遍验收**：`python tools/acceptance.py`（~25 分钟）应 14/14，与 `reports/acceptance.json` 一致。
2. **变异测试**：`pytest tests/acceptance -q` 应 **22/22** 全绿（每个判据的种子变异必须翻红；本轮新增分子案例变异）。
3. **参考数据未被动过**：`grep -r "import chaord" bench/reference/` 仍为空；`check_sanity.py` 6/7 过（cu 密度漂移为记录在案的已知限制）。
4. **快测全绿**：`pytest tests -m "not slow"` 应 **486 过**。
5. 逐案例数字：`reports/acceptance_details.md`（A4 的 22 格 P/R、A5 每案例距离/地板比、A9 逐案例）。

### P1 —— 深入审查点（本轮改动最大的地方）

6. **A4 热淬火**：`src/chaord/lift/defects.py` 的 `_quench_thermal`（受约束最小化的物理依据注释）+ `tests/test_defects.py` 两个修前失败复现。
7. **多物种界面**：`src/chaord/lift/slab.py` 的 `_liquid_bulk_windows`（薄膜降级）与多物种分支；`src/chaord/build/slab.py` 的逐区域弛豫；`tests/test_interface_species.py`（19 项）。
8. **水模型**：`tests/test_solution_model.py`（13 项，修前 11/12 失败）；`molecular.yaml` 的 `water_models` 表（带文献引用）；`cv/noise.py` 分子内排除的门控与注释。
9. **判定学权衡**：读 gate_a_prime.md 的 "Recorded trade-offs" 两节——是否同意这两个决定。
10. **管线 stage 1**：`src/chaord/lift/pipeline.py` + `tests/test_pipeline_equiv.py`（9/9 字节等价）。

### P2 —— 抽查

11. 方言键换算表：`docs/reference.md` 的 metal 0.2.4 节（lj 语义 × Cu 尺度）。
12. 确定性：A10 + 各修复包的同种子双跑断言。
13. 阈值纪律：`tools/check_magic_numbers.py` clean（本轮所有新参数都在方言 YAML 带出处）。

## 六、已知限制（全部有出处）

| # | 限制 | 位置 |
|---|---|---|
| 1 | cu_solid_liquid 参考帧固/液密度漂移 ±16%/±5%（共存 MD 相分数交换体积） | bench/reference sanity（记录在案 + 测试断言） |
| 2 | 分子内排除后 A5 的 g(r) 不再抓水模型几何错误（由精确测试接管） | gate_a_prime.md trade-off 1 |
| 3 | Cu-水界面不跨弛豫（无已发表跨项势） | gate_a_prime.md trade-off 2 |
| 4 | `test_interface_width_within_noise_floor` 单帧噪声敏感（负载下 flaky，钉住的案例过） | tests/test_surfaces.py |
| 5 | MACE/LAMMPS/PLUMED 属可选外接 | docs/licenses.md |
| 6 | LLM 50 提示用确定性 writer（非真 LLM 评估） | tools/llm_prompt_suite.py |
| 7 | 干净机器验收靠 nightly workflow 产物（Actions artifact） | .github/workflows/ci.yml |

## 七、复现命令

```bash
git clone https://github.com/WhiteCrosstheRiver/Chaord.git && cd Chaord
python -m venv .venv
.venv/bin/python -m pip install -e .[test]        # Linux/macOS
# Windows: .venv\Scripts\python -m pip install -e .[test]

PY=.venv/bin/python        # Windows: .venv\Scripts\python

$PY -m pytest tests -m "not slow"              # 快测（~8 分钟，486 项）
$PY -m pytest tests/acceptance -q              # 变异测试（22 项）
$PY tools/acceptance.py                        # 验收 A1–A14（~25 分钟）
$PY tools/check_magic_numbers.py               # 阈值纪律门禁
$PY bench/reference/check_sanity.py            # 参考数据物理健全性
```

## 八、关键文件索引

| 顺序 | 文件 | 内容 |
|---|---|---|
| 1 | `reports/gate_a_prime.md` | **主报告**：14/14 + 修复链 + 两项权衡 + 已知限制 |
| 2 | `reports/acceptance.json` / `acceptance_details.md` | 最新验收机器输出 + 逐案例数字 |
| 3 | `reports/verification_2026-09-28.md` | 独立验证者报告 |
| 4 | `reports/noise_floors.json` | 全部噪声地板（含 provenance 注记） |
| 5 | `bench/reference/` | 独立 MD 参考数据 + provenance（7 案例） |
| 6 | `docs/reference.md` | 方言键文档（metal 0.2.4 换算表、水模型表、平衡协议） |
| 7 | `src/chaord/lift/pipeline.py` | 段优先管线 stage 1（设计已批） |
| 8 | `AGENTS.md` / `PLAN.md` | 规则与计划 |
