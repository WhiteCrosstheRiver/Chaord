# Chaord 审核指南（供远程评审 · Gate A′ FAIL 修复后更新版）

日期：2026-09-29 · 分支 master · 42 个提交 · 已全部推送
仓库：https://github.com/WhiteCrosstheRiver/Chaord

---

## 一、当前状态（一句话）

**v0.1 原型，诚实核心 + FAIL 修复已完成：12/14 验收通过、2 项真实物理缺口如实报告。** 三个阶段：M0-M8 功能开发 → 诚实核心（独立参考数据 + 严格验收器 + 独立验证）→ FAIL 修复（四个并行智能体流）。数字现在是真实的，每一项都有能失败的测试和干净机器上的运行。

## 二、三轮进展总览

| 阶段 | 验收 | 关键变化 |
|---|---|---|
| 第一轮（M0-M8） | 自称 14/14 | 外部评审指出验证循环：CI 失败、无序帧非物理、验收器弱于 PLAN |
| 第二轮（诚实核心） | 9/14 | 独立 MD 参考数据（6 案例×5 帧已发表势）；验收器逐字对齐 PLAN + 17 变异；独立验证者报告；真守恒；阈值卫生 |
| **第三轮（FAIL 修复）** | **12/14** | A2/A3 固溶体修复（SRO 显著性 + 旋转变换不变性 + 物种盲匹配器）；A5 LJ 案例修复（物理重建 + 协议修正）；A9 大帧（1372 原子）；CSL 跨平台直方图法 |

## 三、12/14 逐条结果

| ID | 结果 | 要点 |
|---|---|---|
| A1 语法与格式 | ✅ | 9 示例全过；10,000 生成程序幂等 |
| **A2 不变性** | **✅ 新过** | **9/9（含随机固溶体）；18/18 刚体变换字节一致** |
| **A3 精确往返** | **✅ 新过** | **9/9 文本字节一致 + 结构拟合（随机固溶体用物种盲 + Warren-Cowley α vs 重标噪声底）** |
| A4 缺陷恢复 | ❌ | 召回 22/22 全过；精确率 18/22（最差 0.09 @0.8Tm 间隙原子） |
| A5 统计往返 | ❌ | **LJ 案例全过**（liquid ×0.98, interface ×1.15, glass ×1.23）；分子案例（水 ×2.2, NaCl ×35.7）未过 |
| A6 守恒 | ✅ | 126/126 三方计数+电荷精确 |
| A7 相分割 | ✅ | 10/10 帧 ≥95%（最差 0.999） |
| A8 反应 census | ✅ | 2 个独立构造案例精确 |
| **A9 压缩比** | **✅ 新过** | **1372 原子帧，最差比 0.53%（限 2%）** |
| A10 确定性 | ✅ | 重复 lift 字节一致；同 seed build 坐标一致 |
| A11 速度 | ✅ | 10 万原子 lift 4.0 秒（限 120 秒） |
| A12 静态检查 | ✅ | 四类种子错误全捕获 |
| A13 无崩溃 | ✅ | 125/125 帧零异常 |
| A14 文档 | ✅ | 35/35 方言键覆盖 |

## 四、本轮修复详情（审核重点）

### A2/A3 固溶体修复（F3 流，发现 3 个根因而非 1 个）

| 根因 | 修复 |
|---|---|
| SRO 发射不稳定：108 原子随机溶液的 α₁ 是 α=0 零假设的 ~2.6σ 涨落，旧门限（2.5×8 样本 bootstrap）把它当真信号输出 | 3.5σ 显著性门限；顺序不变 bootstrap（排序多重集）；方言 0.2.1 |
| 旋转变换崩溃：缺陷提升假设对角盒（`cell_diag` + `cKDTree(boxsize=...)`），刚体旋转帧直接 crash，auto 模式落入流体提升器 | `_axis_canonical`：任意正交盒旋转到轴对齐 |
| 平移变换错位：缺陷拟合无原点搜索，重成像帧与锚原点理想位点错位（某变换拟合 a=3.510 vs 真值 3.560） | 双链扫描：原点锚定（基线行为）+ 基矢原子种子重锚定（改善 ≥0.7 比例时采用） |

随机固溶体的 A3 匹配：**从 ground truth 判定**（永不来自 lifted 文本）→ 物种盲 StructureMatcher + Warren-Cowley α vs ≥5 种子重标噪声底。已获批准（2026-09-29，含 4 个变异测试 + cuau_random 同等覆盖 + PLAN.md 标注）。

### A5 LJ 案例修复（F2 流，根因与预判不同）

**根因不是 MD 步数不足**，而是验收器重建子进程用 `physics=False`——重建帧是纯 RSA 堆积、**从未跑过 MD**。开启物理路径后液体/界面案例在默认 2000 步即通过。

额外修复：淬火温度跟踪（原实现 T_hi==T_lo 导致淬火只跑 1 步）；melt/anneal 去掉 fcap（force cap 让热对穿入排斥芯冻结出错误峰位）。

### A5 build bugs（F1 流）

| Bug | 根因 | 修复 |
|---|---|---|
| nacl_aq: `'H10NaO5' is not in list` | Na 的 ASE 共价半径 1.66 Å 实为金属半径，Na⁺–O 水合接触（~2.35 Å）落入 1.25×(1.66+0.66)=2.90 Å 键窗口 | `ion_solvation_elements` 规则（碱金属边丢弃）；显示名映射；`species_mass` 优雅报错 |
| water_tip4p: 打包卡死 | RSA 有效体积分数 η≈0.376 恰在堵塞极限 0.384 | `packing_radius` = max(原子位移 + bond_tolerance×共价半径)；单原子离子用 Shannon 半径 |

### CSL 跨平台（F4 流）

删除 BLAS 敏感的单参考原子枚举，替换为全原子第一壳层键角直方图（cKDTree 周期配对 → mod 90° 折叠 [0,45°] → 主峰位置差 = 取向差角）。**4ms（比旧方法快）、10 种子全过、几何鲁棒（无排序依赖）。**

## 五、建议审核路线

### P0 —— 修复是否真的修了（不弱化测试）

1. **变异测试仍全绿**：`pytest tests/acceptance -q` 应 **17/17**——每个验收标准的种子变异必须翻红。本轮修了 A2/A3/A5 但没有弱化任何判据；相反，变异覆盖还增加了（A2/A3 各加 2 个变异）。
2. **独立参考数据未被动过**：`grep -r "import chaord" bench/reference/` 仍为空；`check_sanity.py` 仍 5/6 过。
3. **跑一遍验收**：`python tools/acceptance.py`（~15 分钟）确认 12/14 与报告一致。

### P0 —— 验收数字核实

4. 读 `reports/gate_a_prime.md`——已更新到 12/14，每个 FAIL 的根因和下一步。
5. 读 `reports/acceptance_details.md`——逐案例数字（A4 的 22 格 P/R 矩阵、A5 的每案例距离/底比值、A9 的逐案例比例）。

### P1 —— 需要老师裁决

6. **S6 设计审批**：`docs/design/lift_build_v2.md`（449 行三阶段方案）——批准后才动代码。
7. **剩余 2 个 FAIL 的优先级**（gate_a_prime.md "Next steps"）：
   - A4 精确率（热感知位点容差 or 位移场缺陷分配）
   - A5 分子案例（平衡 MD 重建 vs RSA 堆积）

### P2 —— 抽查

8. 旋转变换不变性：读 `tests/test_crystal.py` 的变换测试 + `src/chaord/lift/defect_program.py` 的 `_axis_canonical`
9. CSL 直方图法：读 `src/chaord/lift/extended.py` 的 `_folded_shell_angles` + `_dominant_angle`
10. 快测全绿：`pytest tests -m "not slow"` 应 387 过 0 败

## 六、已知限制（全部有出处）

| # | 限制 | 位置 |
|---|---|---|
| 1 | A4 间隙原子 @0.8Tm 精确率 18/22（热抖动伪缺陷对） | gate_a_prime.md |
| 2 | A5 分子案例：水 ×2.2、NaCl ×35.7（RSA 堆积 ≠ 平衡 MD） | 同上 |
| 3 | A5 两个 bench 案例 build 失败（interface 超时、cu_water 物种键） | 同上 |
| 4 | Cu 固液参考帧无法平衡（FBD-Cu 熔体外延再结晶） | bench/reference/cu_solid_liquid/provenance.json |
| 5 | MACE/LAMMPS/PLUMED 属可选外接 | docs/licenses.md |
| 6 | LLM 50 提示用确定性 writer（非真 LLM 评估） | tools/llm_prompt_suite.py |
| 7 | S6 设计待审批 | docs/design/lift_build_v2.md |

## 七、复现命令

```bash
git clone https://github.com/WhiteCrosstheRiver/Chaord.git && cd Chaord
python -m venv .venv
.venv/bin/python -m pip install -e .[test]        # Linux/macOS
# Windows: .venv\Scripts\python -m pip install -e .[test]

PY=.venv/bin/python        # Windows: .venv\Scripts\python

$PY -m pytest tests -m "not slow"              # 快测（~4 分钟，387 项）
$PY -m pytest tests/acceptance -q              # 变异测试（17 项）
$PY tools/acceptance.py                        # 验收 A1–A14（~15 分钟）
$PY tools/check_magic_numbers.py               # 阈值纪律门禁
$PY bench/reference/check_sanity.py            # 参考数据物理健全性
```

## 八、关键文件索引

| 顺序 | 文件 | 内容 |
|---|---|---|
| 1 | `reports/gate_a_prime.md` | **主报告**：12/14 验收 + FAIL 解释 + 下一步 |
| 2 | `reports/acceptance.json` | 最新验收机器输出 |
| 3 | `reports/acceptance_details.md` | 逐案例数字（A4 P/R 矩阵、A5 距离比、A9 比例） |
| 4 | `reports/verification_2026-09-28.md` | 独立验证者报告（第二阶段时点） |
| 5 | `docs/design/lift_build_v2.md` | 待审批的 v2 架构设计 |
| 6 | `bench/reference/` | 独立 MD 参考数据 + provenance |
| 7 | `reports/self_assessment_2026-09-28.md` | 第一阶段自评（含 Caveat） |
| 8 | `AGENTS.md` / `PLAN.md` | 规则与计划（新增 2 条规则 + A3 标注） |
