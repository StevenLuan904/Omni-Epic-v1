# 实验手册：peptide sequence/structure codesign + Rosetta ΔG

## 目标

在异步 peptide/receptor flow 通过小数据过拟合后，开启 peptide sequence
生成，使模型同时输出 peptide sequence、peptide structure 和适配后的
receptor pocket，并用 Rosetta 对生成复合物进行独立结合能验证。

## 输入与模型

- 输入：anchor receptor、masked peptide sequence、peptide 初始结构/噪声。
- 保留：PepFlow peptide structure flow、DynamicBind receptor flow、条件
  adapter 和异步 `t_p/t_r` 时间轴。
- 新增：轻量 peptide sequence head（categorical denoising/flow），输出每个
  residue 的 amino-acid logits；sequence embedding 作为 peptide structure
  flow 的条件，同时生成结构再反馈 pocket context。
- 训练参数分阶段打开：先冻结已有两条 flow，仅训练 sequence head；再解冻
  adapter 与 receptor flow，最后才允许 PepFlow structure 层微调。

## 小数据过拟合流程

1. 选择 1 个 case 做 sequence+structure smoke，确认 mask 后仍能反向传播。
2. 使用 4--8 个 case 做 codesign overfit；mask 15%、30%、50% 三个比例。
3. 使用全部 40 对配对做正式小数据过拟合，并保留 random、peptide-cluster、
   receptor-family proxy 的独立评估清单。
4. 每个 peptide 生成多个 sequence/structure samples，按 sequence logits、
   structure score 和 Rosetta 分数排序；不得只报告单个幸运样本。

## 损失

- sequence cross-entropy / categorical flow loss（仅对 masked residues）；
- peptide backbone/frame/torsion flow loss；
- receptor translation/rotation/χ flow loss；
- correct-state ranking 与 minimal-relaxation；
- peptide/receptor shuffle 和 zero-condition controls；
- 可选 interface-contact loss，但不得用 Rosetta 分数直接反传，避免把
  Rosetta 当作训练标签泄漏进模型。

## Rosetta 验证

对每个生成复合物和对应 native 复合物使用完全相同的 Rosetta protocol：
结构清理、约束、侧链 repack/minimization、`InterfaceAnalyzer`。至少保存：

- `dG_separated`（Rosetta interface binding-energy proxy）；
- `dSASA_int`、interface contacts、shape complementarity；
- generated 与 native 的 `ΔΔG = dG_generated - dG_native`；
- peptide backbone RMSD、sequence recovery、pocket RMSD；
- Rosetta 失败原因和未收敛样本数。

## 通过标准

- masked sequence recovery 明显高于随机基线，并且结构没有退化；
- 生成 peptide pose/pocket RMSD 相对 anchor 改善；
- top-ranked samples 中，Rosetta `dG_separated` 不显著差于 native，且
  `ΔΔG`、interface contacts、dSASA 与结构指标方向一致；
- shuffle/zero peptide 后 sequence ranking、结构 ranking 和 Rosetta
  interface 质量同步下降；
- 所有结论同时报告生成样本数、有效结构数、Rosetta 失败率和均值/中位数，
  不用单个最低能量样本宣称成功。

Rosetta 结果是独立验证，不等同于实验亲和力；本手册验证的是小数据
codesign 闭环和排序能力，不是大规模新序列泛化。apo/AlphaFold anchor
需要数据集整理完成后另行加入。
