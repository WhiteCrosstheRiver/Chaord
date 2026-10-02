# PASS 证据说明书（逐判据）— 供严格审阅

日期：2026-10-02 · 树：`7b0ba53`（全部已推送）· 仓库：
https://github.com/WhiteCrosstheRiver/Chaord

本文对 A1–A14 的每一条 PASS 给出：判据的 PLAN 原文要求、得出该判定的
**精确命令**、数字的来源与独立性边界、**必须翻红的 power 变异**及其数字、
独立验证者的复算交叉引用、复现入口、以及该 PASS 附带的已知限制。
凡是偏离 PLAN 字面的地方都在对应条目里明示并列入待批清单，不做静默替代。

## 运行标识与总规则

- **本地运行**（bookkeeping）：Windows 10，`.venv/Scripts/python.exe`，
  提交 `7b0ba53` 的工作树。
- **可引用的干净机器运行**（AGENTS 规则：报告只引干净运行工件）：
  GitHub Actions **workflow_dispatch run #51**（`7b0ba53`，全新 Ubuntu
  runner）：验收 14/14 + 35 分钟慢统计套件，全绿；工件
  `clean-acceptance-report` 由 nightly 作业上传。前一可引运行：#46
  （`d892598`）、#39（`ed85108`）。push CI 双 OS 绿的连续运行：#45/#47/#48/#50。
- **power 变异**：每个判据注入"最可能的真实错误"，其运行必须 FAIL
  （tests/acceptance/test_mutations.py，24 项 + 对抗套件）。
- **对抗注册表**：`tests/adversarial/`，红队 12 项发现全部修复；
  严格 xfail 注册表当前仅 1 项开放（lj overlap 值，待用户批准）。
- **独立验证者**（零代码作者权）：`reports/verification_2026-09-30.md`，
  其复算数字在下文逐条引用。

复现入口总表（Windows；Linux 用 `.venv/bin/python`）：

```bash
PY=.venv/Scripts/python.exe
$PY tools/acceptance.py                 # A1-A14 全量（约 30-40 分钟）
$PY -m pytest tests/acceptance -q       # 24 项变异性金丝雀（约 55 分钟）
$PY -m pytest tests/adversarial -q      # 对抗套件（约 18 分钟）
$PY tools/tutor_demo.py                 # 老师现场演示（约 5 分钟，退出码自证）
$PY -m pytest tests -m "not slow"       # 675 项快测（约 38 分钟）
```

---

## A1 语法与格式 — PASS

**要求（PLAN）**：规范示例全部可解析；parse==IR、fmt 幂等两条性质律成立。

**判定证据**：10/10 spec 示例（spec/examples/*.chaord）解析通过；
10,000 个随机生成程序上 parse→IR→fmt→parse 幂等。

**实现独立性**：性质律由 hypothesis 式生成器驱动（tests/test_fmt_property.py
的字母表含引号/反斜杠/科学计数法边界）；红队 F7（反斜杠转义吞字符）修复后
`"C:\data\x.xyz"` 往返逐字节保持。

**power 变异**：`fmt_idempotence` 变异（格式化器注入一次改写）必须翻红——
金丝雀 `test_a1_fmt_idempotence_and_parse` 红/绿如常。

**验证者复算**：报告 §2 A1 行同数字。

**限制**：无。

---

## A2 规范不变性 — PASS

**要求（PLAN）**：同一帧的刚体变换（旋转/平移/重成像/重排）提升到
**字节一致**的文本。

**判定证据**：9/9 晶体+随机固溶体案例 × 每案例 2 个刚体变换 = 18/18
字节一致。**输入是 bench 存储的热帧**（2026-10-01 起替换了原先的完美重建
帧——验证者观察 §5.3）。

**实现独立性**：变换由验收器自带的 `rigid_transform`（四元数旋转 + 平移 +
重成像 + Fisher-Yates 重排）施加；文本比较为逐字节。

**power 变异**：`scale_lattice`（非均匀 10% 应变）必须翻红——应变超出
lift_symprec 0.25 Å 的设计门，金丝雀断言检测行出现在提升文本中。
**性质测试**（红队 T0 要求）：50 个随机/近阈值标记 × 10 个原子排序 =
500 次提升，**50/50 个标记各自字节唯一**（tests/test_sro_invariance.py）。

**顺带修复的真实 bug**（切换到热帧后暴露）：hcp 位点锚定在旋转帧上锁进
半密度不动点（4 个字典序种子全部 16/32 覆盖，全原子锚定后 32/32）。

**验证者复算**：§2 A2 行 9/9、18/18。

**限制**：无。

---

## A3 精确往返（有序） — PASS

**要求（PLAN）**：lift→build→lift 文本一致；重建结构匹配原帧
（PLAN 字面：pymatgen StructureMatcher ltol 0.2 / stol 0.3 / 5°）。

**判定证据**：9/9 文本字节一致；9/9 结构拟合。**结构拟合的实现是对 PLAN
字面的已披露偏离**：热抖动超胞上 StructureMatcher 结构性失效（其
Niggli/原胞约化无法约化抖动胞——热帧自比都不匹配，已当面测量），换成
验证器侧的**物种感知周期性指派检查**（Hungarian 指派；判据 = 位移超过
0.25 d_NN 的原子占比 ≤ 5%）。实测：干净热帧 p90 ≤ 0.155 d_NN、超 0.25
d_NN 占比 ≤ 0.9%（9 案例全测）；注入 25% 原子位移 0.6 Å 的种子错误后
占比 18.8% → FAIL。随机固溶体按 2026-09-29 的人类批准走物种盲几何 +
Warren-Cowley α 对重标噪声底（现在是对 3 次重建抽签取均值 vs 5 次重标
均值——对称比较）。

**power 变异**（4 个，全部翻红）：`force_occupancy`（有序程序强改为
随机占据——物种感知指派抓）、`displace_rebuilt`（25% 位移 0.6 Å——
质量门抓，18.8% vs 5% 门）、`segregate`（物种 z 排序分块——α 统计抓，
提升文本也发射 SRO 行）、`composition`（5% Cr 改标 Ni）。

**验证者复算**：§3.3 —— fcc_cu 热帧 lift→build→lift 字节一致（335 字符，
sha256 前缀 a851e7f2），cell 7.230 = 2×a 3.615 精确自洽（正是 F1 修复
前断裂的契约）。

**待批**：结构匹配器偏离（decision list #6）。

---

## A4 缺陷恢复 — PASS

**要求（PLAN）**：全部缺陷类型 × 宿主 × 温度（至 0.8 Tm）格，
precision 与 recall 双 ≥ 0.95。

**判定证据**：**26/26 格全部 P=R=1.000**（3 宿主 × 四类 K-V 缺陷 ×
两温度 + 红队 F12 后加入的两个混合格：fcc-Cu V_Cu3+Cu_i3、
L12-NiAl V_Ni3+Al_Ni4+Ni_i3——即红队构造的原始场景）。

**实现独立性**：植入由验收器自己完成（原型位点自建、缺陷物距 ≥2 d_NN
分离、热抖动 0.8 Tm 幅值施加）；P/R 按**植入多重集**对**检出多重集**
逐类型绑定计分（F12 修复：旧计量每格只对一种 token 计分，混合格完美
检出算出 P=0.30）。

**关键实现**（0.8 Tm 全满的物理）：Wigner-Seitz 前先做**受约束能量最小化**
（LJ + 每原子到拟合位点的谐和弹簧，离位原子冻结）——修前最差格 P=0.09
（30 个伪 frenkel 对，根因：位点容差锚在热帧最小近邻距上，已有两个
修前失败的复现测试在 tests/test_defects.py）。

**power 变异**：`false_defect`（植入一个不存在的缺陷）翻红；
红队混合格变异 `test_a4_mixed_cell_mutation` 断言网格必须含混合格。

**验证者复算**（§3.2）：**用自己从头写的植入器 + 新种子**独立重植两个
混合格——检测==植入、P=R=1.000、tp/fp/fn = 6/0/0 与 10/0/0（两种温度），
证明不是对验收种子调过的。

---

## A5 统计往返 — PASS

**要求（PLAN）**：held-out 观测量 ≤ 1.5× 噪声地板；流体/界面 ≥90%、
非晶 ≥80% 逐案例；观测对 ≥5 帧、≥3 种子平均（评审 3 step 3）；
温度来自 provenance；物理关闭与 ±20% 温度变异必须 FAIL。

**判定证据**：6/6 带地板案例全过（fluid 4/4、interface 1/1、glass 1/1）；
参考侧 = ≥5 帧逐帧观测平均，重建侧 = 3 种子（7/13/29）平均；地板取
max(均值, P90)（校准链：重建 vs 参考的距离分布 ≈ 参考自身帧对分布，
经验最大值 ≈ 1.5×均值，均值门按构造误杀 ~20% 平衡重建——全部测量记录
在 gate 报告）。逐案例最差比值：water ×1.1、nacl ×1.1、solid_liquid
×0.9、liquid ×0.6、liquid_large ×0.7、glass ×0.4。

**温度功效（评审核心要求）**：`temp_lo`（0.8×provenance T）→ gr ×2.4 /
cn ×1.9 双超门；`temp_hi`（1.25×）→ ×1.9 / ×2.6 双超门；正确温度
×0.67/×0.58 过。`physics_off` → cn 0.214 vs 门 0.114 翻红。
**诚实余量**：500 原子 lj_liquid 单独无 ±20% 分离度（红队 F3 状态注记
+ tutor_demo 屏上声明）——温度证据由 2048 原子案例承载。

**provenance 温度**：lj_liquid/liquid_large 重建在真实 T*=0.72（此前错误
地用方言默认 0.65 平衡）；缺省 T 包装为 AssumedT 并在程序 provenance 写
"T assumed (dialect default …); not measured from the frame"。

**地板出处（全部可溯源）**：玻璃 = 3 次独立淬火（N=2048/淬火，评审 3
T3 字面）的跨淬火对；液体 = 去相关滞后半区 + 2048 案例双轨迹 21 对；
全部逐对数据在 reports/noise_floors.json。

**验证者复算**（§3.1）：亲自从行注释解析距离/地板重算判定（不信运行器
状态位）——正确 T 过、×0.8 与 ×1.25 双超门，数字与运行器一致。

---

## A6 守恒 — PASS

**要求（PLAN）**：帧实数 == 程序 conserve 行；区域算术三方核对；电荷一致。

**判定证据**：126/126 帧计数精确；验证器**自己的**区域算术（composition/
occupancy/molecules/缺陷净额/residual）推导或交叉核对 81 个程序（不可推导
形状双向检查——F4 修复后 slab 界面程序也进入推导：分子 slab 按公式展开
并用自己的 stated density 审计自身，晶体 slab 报告部分推导）；
126/126 电荷一致。

**power 变异**：`drop_atom`（帧删一原子）翻红；红队 F4 场景（cu_water
删 30 个水）现在 derivation_ok=False（H 212≠272、O 106≠136 且密度自相
矛盾 0.725 vs 0.929 g/cm³）；F5 场景（5×Ca²⁺+5×Cl⁻ 声称中性）修后
报 "expected 0, actual 5"（Ca²⁺ 按 +2 计，价态表出处 Greenwood &
Earnshaw 1997）。

---

## A7 相分割 — PASS

**要求（PLAN）**：界面帧逐原子相标签 ≥95% 正确（界带排除）。

**判定证据**：10/10 帧 ≥95%（最差 0.999）；judged-fraction 逐帧披露
（cu_water 0.854-0.874、lj 0.355-0.395）+ **范围闸门**：判定原子占比
≥90%×界面几何可得率（F9 修复：掩码数字此前"碰巧对"但不披露无闸门）。

**power 变异**：`flip_labels`（腐蚀 1/5 标签）翻红；`widen_band`
（界带×2）触发范围违例 FAIL（0.000/0.66 判定率）。红队的定向腐蚀测试
（原方向是无操作，已改写为正确方向）在 truth 方向下攻击成立且被抓住。

---

## A8 反应 census — PASS

**要求（PLAN）**：独立构造案例上分子计数精确；**吸附位 ≥90% 正确**
（半边自 2026-09-28 起未执行，2026-10-01 补齐）。

**判定证据**：两个 raw-numpy 手植 H2O/OH/H 混合帧（147 与 92 原子，
坐标直接写在 numpy 里，不经 chaord 构建器）census 逐物种精确；
吸附位：raw-numpy 构造的 Pt(111) 4 层板 + 8 个 O 吸附物（fcc(111) 堆垛
按绝对行参数化，与已发表 Pt 几何逐点核对 72/72 距离 0.000）——真值
5 top + 3 bridge，提升程序声明 `{top: 5, bridge: 3}`，**100% 正确**。

**power 变异**：`extra_oh`（多植一个 OH）census 必须不一致——翻红；
`wrong_site`（top 位吸附物侧移 1 Å——位点错配的替身）声明降到 50% →
FAIL（门 ≥0.90 为 PLAN 原文）。

---

## A9 压缩 — PASS

**要求（PLAN）**：≥1000 原子真实体系，程序/坐标比 ≤2%（评审 3：必须
含异质帧）。

**判定证据**：门控测量集 **4 个**（不再只有完美晶体）：l12_ni3al 1372
原子 0.49%（晶体类最差）、lj_liquid_large 2048 原子 0.40%、
nacl_aq 1640 原子 0.51%、**reference/lj_solid_liquid 2304 原子 0.55%
（异质类最差 = 总最差，评审要求的固液界面 MD 帧）**。4 个中 2 个异质。

**诚实披露**：reference/cu_solid_liquid（832 原子）任何模式不可提升
（提升尝试抛 ChaordError），记录为 "not liftable (recorded, not gated)"。

---

## A10 确定性 — PASS

**要求（PLAN）**：重复提升字节一致；同种子构建坐标一致。

**判定证据**：bench 缺陷帧重复 lift 字节一致 True；同种子两次 build
坐标一致 True。跨进程哈希一致亦被 F2 修复流锁定（29ccd9d3… 两次相同）。

---

## A11 速度 — PASS

**要求（PLAN)**：10 万原子提升 ≤120 s。

**判定证据**：100,000 原子流体提升 6.6 s（≤120 s，18 倍裕量）。
**输入如实标注**：fcc 点阵 ρ=0.85 加 0.10×a 抖动（热无序，非平衡 MD）；
程序声明精确原子数：是。

---

## A12 静态检查 — PASS

**要求（PLAN）**：四类种子错误全捕获。

**判定证据**：lattice_mismatch / impossible_density / charge_imbalance /
overlap_0.1σ 四类全部 caught。F6 修复后重叠检查的有效容差 = max(方言值,
sanity 分数×体系尺度)（不再比自家参考数据硬核松）；F5 修复后多价离子
按价态表计（CaCl₂ 正确程序从双 FAIL 变双 PASS，+5 荒谬程序从 PASS 变
FAIL）。两项方言值变更待批（decision list）。

---

## A13 无崩溃 — PASS

**要求（PLAN）**：全 bench 无异常提升。

**判定证据**：125/125 帧零异常、零 residual 原子；61 帧带路由诊断记录
（记录的 lift_mode 被拒或类别名——回退 auto **被记录**而非静默吞掉）。
hcp_mg 5/5 按记录的 mode=crystal 直接提升成功（F2 修复前全部误路由成
伪分子液体）；NaN/非有限坐标在 Frame 入口抛 ChaordError（F10 修复前是
SIGSEGV exit 139）。

---

## A14 文档 — PASS

**要求（PLAN）**：方言键 100% 文档化 + 示例。

**判定证据**：35/35 键结构化覆盖（代码 span/标题/表格行内整词 +
**引用一条真实 spec/examples 行**——散文不算覆盖，F8 修复：散文参考
从 35/35 PASS 变 0/35 FAIL）；26/35 键的示例引用受闸；**9 个键如实上报
"无处引用"**（atom, chirality, … 全库无示例，报告而非豁免）。

**power 变异**：`hide_key`（从文档删一个键）翻红；散文拒绝断言进金丝雀。

---

## 附：本文件未掩盖的事项

1. **五项待用户决策**（gate 报告 "Decisions pending"）：lj overlap 值、
   overlap_sanity_fraction、lj_solid_liquid 密度容差 4%、本阶段全部新
   方言键追认、PR-per-stream 与按流提交等价、A3 匹配器偏离。
2. **诚实余量**：500 原子液体的 ±20% 温度分离度不足（已记录，演示中
   屏上声明）。
3. **玻璃 T3 字面升级已完成（2026-10-02）**：N=2048 × 3 独立淬火
   （生成器步数按 t_mix∝N^(2/3) 从 500 原子校准值放大：melt 3700 /
   quench 7400 / anneal 2500，帧距 1200）；跨淬火地板 gr 0.0525 /
   cn 0.032（有限尺寸平均使地板小于 500 原子时的 0.0999——方向符合
   预期，跨>同的守卫保持：0.0525 > 0.0389）；重建协议按同一 N^(2/3)
   律标度并校准退火深度后，A5 玻璃行 cn ×0.9 / gr ×0.5 全过。
   随之引入的方言基值变更（glass_melt/quench/anneal_steps 1500/3000/
   2850 + glass_protocol_ref_n 500）与 N 感知重建预算已列入待批清单。
