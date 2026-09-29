# Chaord 审核指南（供远程评审 · Gate A′ 版）

日期：2026-09-28 · 分支 master · 39 个提交 · 已全部推送
仓库：https://github.com/WhiteCrosstheRiver/Chaord

---

## 一、当前状态（一句话）

**v0.1 原型，诚实核心已建成并独立验证：9/14 验收通过、5 项真实物理/数据缺口如实报告。** 这不是"v1.0 完成"——前一轮外部评审正确地指出了自测验证的循环性，本轮全部修复，数字现在是真实的。

## 二、两轮评审之间的变化

| 上一轮评审指出的问题 | 本轮处置 | 证据 |
|---|---|---|
| P0 "v1.0" 从未在干净环境验证，门被作者自己评 | 状态降级为 v0.1 原型；gate_b 改名为 self_assessment；打标签 v0.1-selftest 冻结 | README、reports/self_assessment_2026-09-28.md |
| P0 无序基准帧不是物理的 | **6 案例 × 5 帧独立 MD 参考数据**（ASE 引擎 + 已发表势：LJ、TIP4P、SPC/E+Joung-Cheatham+Wolf/DSF、FBD-Cu EAM 从 NIST 下载带 sha256） | bench/reference/（生成代码禁 import chaord，有 grep 测试强制） |
| P0 验证循环（自编自测） | 验收器独立实现（自带原子计数、自带 .chaord 文本解析、自带晶体学数据表）；**新 agent 独立验证报告** | tools/acceptance.py、reports/verification_2026-09-28.md |
| P0 验收器弱于 PLAN 定义 | **逐字对齐 PLAN 的 A1–A14**；每条配变异测试（17/17 翻红） | tools/acceptance.py、tests/acceptance/test_mutations.py |
| P1 守恒循环论证 | 三方校验（帧实数 vs 程序隐含计数 vs conserve 声明）；drop-atom 变异翻红 | src/chaord/check/statics.py、tests/test_conservation.py |
| P1 打包/可移植性 | spglib+pymatgen 声明；bash shell；schedule；as_posix；双 OS 命令 | pyproject.toml、.github/workflows/ci.yml |
| P2 阈值逃逸 | 153→134 豁免；真实容差迁入方言键；checker 扩到 cv/realize | tools/check_magic_numbers.py |
| P2 观测太薄 | 偏 g(r)（分子内键剔除）、键角分布、密度剖面；暴力交叉验证 | src/chaord/cv/rich.py、tests/test_rich_observables.py |
| P1 无通用架构 | 设计文档（只文档，待审批后才动代码） | docs/design/lift_build_v2.md（449 行三阶段方案） |

## 三、诚实验收数字（独立验证者实测）

**9/14 通过，5 项真实 FAIL**——每个 FAIL 是物理/数据缺口，不是检查器弱化：

| ID | 结果 | 要点 |
|---|---|---|
| A1 语法与格式 | ✅ | 9 示例全过；10,000 生成程序幂等 |
| A2 不变性 | ❌ | 7/8（fcc_crconi 随机固溶体破坏字节级不变性） |
| A3 精确往返 | ❌ | 7/8（同因；pymatgen StructureMatcher 对 crconi 也不匹配） |
| A4 缺陷恢复 | ❌ | 召回 22/22 全过；精确率 18/22（最差 0.09 @0.8Tm 间隙原子） |
| A5 统计往返 | ❌ | 0/3 有底案例过（距噪声底 7–27×）；4 例 build 失败 |
| A6 守恒 | ✅ | 126/126 三方计数+电荷精确 |
| A7 相分割 | ✅ | 10/10 帧 ≥95%（最差 0.999） |
| A8 反应 census | ✅ | 2 个独立构造案例精确 |
| A9 压缩比 | ❌ | bench 无 ≥1000 原子帧（补充测量 0.74%，不计入） |
| A10 确定性 | ✅ | 重复 lift 字节一致；同 seed build 坐标一致 |
| A11 速度 | ✅ | 10 万原子 lift 3.9 秒（限 120 秒） |
| A12 静态检查 | ✅ | 四类种子错误全捕获（含重叠、电荷失衡） |
| A13 无崩溃 | ✅ | 125/125 帧零异常、零 residual 原子 |
| A14 文档 | ✅ | 35/35 方言键覆盖；26/35 有示例 |

## 四、建议审核路线（按优先级）

### P0 —— 核心诚实性（评审最关心的问题是否真的修了）

1. **独立参考数据**：读 `bench/reference/*/provenance.json`（每案例记录
   引擎/势+引用/协议/seed/帧时刻）；跑 `python bench/reference/check_sanity.py`
   确认 5/6 过（Cu 诚实记为已知限制——FBD-Cu 熔体在完美 fcc 模板上
   1–4ps 内外延再结晶，无法产出平衡两相帧）。
2. **循环验证禁令**：`grep -r "import chaord" bench/reference/` 应为空
   （有测试强制：tests/test_reference_data.py）。
3. **独立验证报告**：读 `reports/verification_2026-09-28.md`——新 agent
   写的（零代码作者权），含逐条 A1–A14 数字 + 独立 g(r) 复核（偏差 0.2%）。
4. **变异测试**：`pytest tests/acceptance -q` 应 17/17——每条验收的
   种子变异必须翻红，证明检查器本身可靠。

### P0 —— 验收数字是否真实

5. 读 `reports/gate_a_prime.md`——Gate A′ 八项清单逐条打勾 + 5 个 FAIL
   的物理解释 + 下一步优先级。
6. 跑 `python tools/acceptance.py`（约 15–20 分钟，含子进程重建）确认
   9/14 与报告一致。

### P1 —— 需要老师裁决的 3 件事

7. **S6 设计审批**：`docs/design/lift_build_v2.md`（449 行）——segment-first
   lift 统一管线 + build 的 region composer；三阶段迁移（每阶段保旧路径
   开关、逐字节等价性测试、最后删旧路径）。**你批准后才动代码。**
8. **S1 物理审核**：`bench/reference/*/provenance.json` 中的势函数选择
   （TIP4P vs SPC/E、JC 离子参数、FBD-Cu EAM）、密度、温度是否合理。
9. **A2/A3 固溶体裁定**：随机固溶体的 SRO 约束行（`constrain sro alpha1
   Ni-Ni +0.18`）是测量值，随 seed 变化——字节级往返是错误判据。建议
   归入统计判据（A5 家族）。需要你裁定。

### P2 —— 抽查（可选）

10. 真守恒：`tests/test_conservation.py` 的 drop-atom 变异（删帧原子 → FAIL）
11. 阈值卫生：`python tools/check_magic_numbers.py` → clean
12. 富观测：`tests/test_rich_observables.py`（偏 g(r) 文献首峰对照 + 暴力实现交叉验证）

## 五、已知限制（诚实声明）

| # | 限制 | 位置 |
|---|---|---|
| 1 | A5 重建帧距噪声底 7–27×（最深的物理缺口：RSA + 短 MD 不足以复现液体/玻璃/界面统计） | gate_a_prime.md |
| 2 | A4 间隙原子 @0.8Tm 精确率 18/22（热抖动伪缺陷对） | 同上 |
| 3 | CI Linux 偶发：CSL 检测的 BLAS 敏感性（Windows 10 种子全过） | tests/test_extended.py 注释 |
| 4 | Cu 固液参考帧无法平衡（FBD-Cu 熔体外延再结晶） | bench/reference/cu_solid_liquid/provenance.json |
| 5 | MACE/LAMMPS/PLUMED 属可选外接 | docs/licenses.md |
| 6 | LLM 50 提示用确定性 writer（非真 LLM 评估） | tools/llm_prompt_suite.py |
| 7 | bench 无 ≥1000 原子原始帧 | acceptance_details.md |

## 六、复现命令

```bash
git clone https://github.com/WhiteCrosstheRiver/Chaord.git && cd Chaord
python -m venv .venv
.venv/bin/python -m pip install -e .[test]        # Linux/macOS
# Windows: .venv\Scripts\python -m pip install -e .[test]

PY=.venv/bin/python        # Windows: .venv\Scripts\python

$PY -m pytest tests -m "not slow"              # 快测（~3 分钟，382 项）
$PY -m pytest tests/acceptance -q              # 变异测试（17 项）
$PY tools/acceptance.py                        # 验收 A1–A14（~20 分钟）
$PY tools/check_magic_numbers.py               # 阈值纪律门禁
$PY tools/sketch_check.py spec/examples/*.chaord
$PY bench/reference/check_sanity.py            # 参考数据物理健全性
```

## 七、给老师的关键文件入口

| 顺序 | 文件 | 内容 |
|---|---|---|
| 1 | `reports/gate_a_prime.md` | **主报告**：八项清单 + 验收数字 + FAIL 解释 + 下一步 |
| 2 | `reports/verification_2026-09-28.md` | 独立验证者报告（逐条数字 + 独立 g(r) 复核） |
| 3 | `docs/design/lift_build_v2.md` | 待审批的 v2 架构设计（449 行） |
| 4 | `bench/reference/` | 独立 MD 参考数据（6 案例的 provenance.json 逐个审物理） |
| 5 | `reports/self_assessment_2026-09-28.md` | 独立验证前的自评（含 Caveat 头） |
| 6 | `reports/noise_floors.json` | 从 MD 帧测量的噪声底 |
| 7 | `AGENTS.md` / `PLAN.md` | 本轮新增的两条规则（参考数据规则 + 主张-证据规则） |
