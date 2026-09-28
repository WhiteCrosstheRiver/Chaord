# Chaord 审核指南（供远程评审）

日期：2026-09-28 · 分支 master · 17 个提交 · 验收 A1–A14 全部通过

---

## 一、项目一句话

Chaord 是原子体系的领域语言：`chaord build` 把程序编译成坐标，`chaord lift`
把坐标反编译回程序。一个程序描述宏观态（一族构型），坐标文件只是它的一个
微观态采样；有序部分精确书写，无序部分用统计书写——从晶体到气体。
**正确性判据是往返**：晶体要求字节级一致，流体/玻璃要求落在实测噪声底内。

## 二、完成状态总览

| 里程碑 | 内容 | 状态 |
| --- | --- | --- |
| M0 | 语法（Lark）、Pydantic IR、规范化 fmt、方言加载器、CV 注册表、CLI、原型移植 | ✅ |
| M1 | 13 种晶体原型、取向/超胞构建、不变性套件（旋转/平移/重排/重成像 → 字节一致） | ✅ |
| M2 | Kröger–Vink 点缺陷（空位/反位/间隙/Frenkel）、SRO/SQS、植入缺陷验收 | ✅ |
| M3 | 分子模板、RSA 堆积、流体 CV、噪声底库、统计往返（g(r) 1.1× 底） | ✅ |
| M4 | 相分割（1-D 剖面 + 真 3-D 区域生长）、界面对象、Miller 表面、Wood 记号、吸附/覆盖度 | ✅ |
| M5 | melt/quench/anneal/deposit 协议、约束（密度/SRO/cn）、环/Voronoi 统计、最短程序控制器 | ✅ |
| M6 | 物种 census、反应界面端到端（dissociate 语句）、位错（Burgers 至机器精度）、CSL Σ 双晶 | ✅ |
| M7 | Chaord-Bench 25 案例 × 5 帧、验收运行器 A1–A14、参考手册、三教程 | ✅ |
| M8 | LLM JSON schema、脚本层、Laya 差分编码器、50 提示验证套件、主动学习钩子 | ✅ |
| 补强波 | 解析式 Finnis–Sinclair EAM 后端（Cu/Fe/Ni）、电荷守恒检查、CI/LICENSE/nightly/许可审计 | ✅ |

**量化指标**：源码 ~7,560 行、测试 ~3,970 行（307 项测试：299 快 + 8 慢标记，
全绿）；bench 125 帧带 ground truth；9 个规范示例全部字节稳定。

## 三、证据地图（每个主张在哪里验证）

| 主张 | 证据 | 复现命令 |
| --- | --- | --- |
| 语法健全 + 规范形 | `tests/test_grammar_examples.py`（9 示例字节稳定、30 错误行号） | `python -m pytest tests/test_grammar_examples.py -q` |
| fmt 幂等（10000 例性质测试） | `tests/test_fmt_property.py` | 同上换文件名 |
| 不变性（旋转/平移/重排/重成像 → 同文本） | `tests/test_crystal.py::test_invariance_*` | `python -m pytest tests/test_crystal.py -q` |
| 晶体精确往返（lift→build→lift 字节一致） | `tests/test_crystal.py::test_exact_round_trip`（12 原型参数化） | 同上 |
| 植入缺陷召回 ≥0.95 | `tests/test_defects.py` + 真实 0.8·Tm 热回复（`tests/test_thermal_recovery.py`，recall 1.00） | `python -m pytest tests/test_defects.py tests/test_thermal_recovery.py -q -m slow` |
| 统计往返 ≤1.5× 噪声底 | `tests/test_fluids.py`（g(r) 1.08×）、`tests/test_amorphous.py`（玻璃） | `python -m pytest tests -m slow` |
| 守恒（原子/物种/电荷） | `src/chaord/check/statics.py`；每次 lift 后自动跑 | `python -m pytest tests/test_charge.py -q` |
| 分割 ≥95% + 真 3-D | `tests/test_segment3d.py`（斜界面 3-D 1.000 vs 1-D 0.74） | `python -m pytest tests/test_segment3d.py -q` |
| 反应 census 精确 + 端到端 | `tests/test_extended.py`、`tests/test_reactive_interface.py` | `python -m pytest tests/test_reactive_interface.py -q` |
| 位错 Burgers 精确 | `tests/test_extended.py`（分量误差 ~1e-16·a） | `python -m pytest tests/test_extended.py -q` |
| 压缩比 | 验收 A9：2 万原子 0.03%（限 2%） | `python tools/acceptance.py` |
| 10 万原子性能 | `tests/test_thermal_recovery.py`：实测 3.5 s（限 120 s） | 同上 slow |
| 阈值纪律（无魔法数） | `tools/check_magic_numbers.py`（CI 门禁） | `python tools/check_magic_numbers.py` |
| LLM 套件 50 提示 | `tools/llm_prompt_suite.py`（确定性 writer 50/50） | `python tools/llm_prompt_suite.py` |
| 总验收 | `reports/acceptance.json`（14/14）、`reports/nightly_2026-09-28.md`（Overall PASS） | `python tools/acceptance.py`、`python tools/nightly.py` |

## 四、建议的审核路线（按优先级）

### P0 —— 核心设计规则是否兑现（AGENTS.md 十条）

1. **阈值只进方言**：抽查 `src/chaord/lift/`、`src/chaord/build/` 任何数值，
   应全部经 `dialect.threshold("名字")` 或带 `# dialect-exempt: <理由>` 标注；
   运行 `python tools/check_magic_numbers.py` 应输出 clean。
2. **一结构一文本**：读 `src/chaord/lang/fmt.py`（规范化打印）+
   `tests/test_crystal.py::test_invariance_rotation_translation_reorder_reimage`。
3. **一个量一个定义**：CV 注册表 `src/chaord/cv/registry.py`（measure/restrain/
   check 三用同源）；抽查 Warren–Cowley α₁ 在 `build/defects.py` 只定义一次。
4. **不丢原子**：`check/statics.py::conservation_check`；residual 机制
   （`lift/slab.py` 中未解释原子写入 residual 块）。

### P0 —— 往返哲学的两个层级

5. **晶体层（精确）**：`tests/test_crystal.py` 的 exact round trip 与不变性；
   注意其中 spglib 约定容差搜索（轴排列/原点平移/极性孪晶/平局物种）是难点所在。
6. **统计层（噪声底）**：`src/chaord/cv/noise.py` 的设计——噪声底是同一模拟
   两帧之间的距离，容差 = 1.5× 底，**从不拍脑袋**。对照
   `tests/test_fluids.py::test_lj_liquid_statistical_round_trip` 的输出。

### P1 —— 关键数值证据

7. 验收报告逐条读：`reports/acceptance.json`（14/14，每条带证据串）。
8. 慢速统计套件：`python -m pytest tests -m slow`（约 5–8 分钟，7 项）。
9. Gate B 报告：`reports/gate_b.md`（含 8 条如实记录的已知限制——请特别审
   "Known limitations" 一节，这是诚实性声明）。

### P1 —— 工程纪律

10. 随机性全部 seed（`np.random.default_rng`）；测试确定性：
    `tests/test_bench.py` 有跨进程逐数组比对。
11. CI：`.github/workflows/ci.yml`（PR 双 OS 矩阵 + nightly slow/acceptance）；
    `tools/nightly.py` 已在本机全量跑通（报告见 reports/）。
12. 许可：LICENSE（MIT）+ `docs/licenses.md` 第三方审计（OVITO 非商用限制、
    MACE/icet 需人工确认两条已标出）。

### P2 —— 扩展能力抽查（可选）

13. EAM 后端：`src/chaord/realize/eam.py`（解析式 FS，Cu/Fe/Ni 按
    a/E_coh/B 标定，力经有限差分校验）+ `tests/test_eam.py`。
14. 反应界面：`tests/test_reactive_interface.py`（rutile+水+OH/H 的完整 lift）。
15. 最短程序控制器：`src/chaord/check/shortest.py`（贪心删除搜索，conserve
    永不删）。

## 五、需要老师裁决/签字的点（AGENTS.md 规定的人工审批项）

| # | 事项 | 位置 | 需要的决定 |
| --- | --- | --- | --- |
| 1 | **Gate B 签字** | `reports/gate_b.md` | 按 AGENTS.md 由人类评审放行 v1.0 |
| 2 | 方言阈值变更（三处） | `metal.yaml`（版本 0.1.0→0.2.0，新增 EAM 参数与 Burgers 行匹配阈值）、`glass.yaml`（新增 md 块与 deposit 键）、`lj.yaml`（lattice_scan_steps 17→57，热振动下 d_NN 中值下移 ~10% 的必要校准） | 批准与否 |
| 3 | spec 扩充 | 第 8（deposit）、第 9（rutile 表面）示例；语法把 molecule/ion/atom 改为双栖终结符（修多原子 residual 不可重解析缺陷） | 批准与否 |
| 4 | 发布动作 | PyPI/GitHub 组织/域名占位 | 人工执行 |
| 5 | 真 LLM ≥90% 首试率 | `tools/llm_prompt_suite.py --writer file` | 接 API 后补测 |

## 六、已知限制（诚实声明，全部在 gate_b.md）

MACE 后端属外接；热弛豫位错核仍到家族级；大分子近液密度 RSA 饱和（物理先验
负责弛豫）；表面 lift 内政识别限 fcc/bcc（diamond 走重构辅助路径）；反应界面
程序可 parse/fmt 但尚无 crystal+liquid+vacuum 组合 builder；SiO₂/CuZr 玻璃与
石墨烯/水 bench 需核心外势函数（记为 v1.0 范围外）；LLM 首试率用确定性 writer
验证管线。

## 七、复现命令总表

```bash
# 环境
python -m venv .venv && .venv/Scripts/python -m pip install -e .[test]

# 快测（~2 分钟，299 项）
.venv/Scripts/python -m pytest tests -m "not slow"

# 慢速统计往返（~5 分钟）
.venv/Scripts/python -m pytest tests -m slow

# 验收 A1–A14（~15 分钟，写 reports/acceptance.json）
.venv/Scripts/python tools/acceptance.py

# 夜间全量（慢测+验收 → reports/nightly_<date>.md）
.venv/Scripts/python tools/nightly.py

# 门禁
.venv/Scripts/python tools/check_magic_numbers.py
.venv/Scripts/python tools/sketch_check.py spec/examples/*.chaord
.venv/Scripts/python tools/llm_prompt_suite.py
```

## 八、仓库结构速查

```
src/chaord/         lang(语法/IR/fmt) dialects(阈值YAML) cv build realize lift check io cli
tests/              307 项（unit/property/golden/round-trip/acceptance/bench）
bench/              generate.py + data/（25 案例 × 5 帧 + ground truth）
docs/               reference.md reference manual / tutorials.md / licenses.md
spec/               grammar.ebnf + 9 个示例程序
prototype/          原始 LJ 演示（参考，不再扩展）
reports/            acceptance.json / gate_b.md / nightly_*.md
.github/workflows/  ci.yml（PR + nightly）
```
