# 实验手册：peptide sequence/structure codesign + Rosetta ΔG

## 目标

在异步 peptide/receptor flow 通过小数据过拟合后，联合生成 peptide sequence、
peptide structure 和适配后的 receptor pocket。设计任务不存在对应 native complex，
因此 Rosetta 只计算每个生成复合物的绝对界面 ΔG proxy，并以更低（更负）的 ΔG
进行筛选，不计算 ΔΔG。

所有模型、schedule、sampling、loss 和 Rosetta 参数统一读取
`pepflow/configs/joint_async_codesign.yaml`；运行时只指定 YAML 路径和输入/输出路径。

## 模型

- 保留 PepFlow peptide structure flow、DynamicBind receptor flow、condition adapter
  和多时钟 time-token encoder。
- 新增 peptide sequence categorical flow/head，输出 amino-acid logits。
- 当前 codesign 阶段 `peptide_seq` 与 `peptide_struct` 仍共享 `peptide` clock；两个
  peptide 模块都显式注入 peptide/receptor time embeddings。
- receptor module 同样显式注入 receptor/peptide time embeddings。未来需要内部异步
  时，将 `peptide` 拆为 `peptide_seq`、`peptide_struct` 两个命名 token 即可。
- 分阶段解冻：sequence head → adapter/receptor flow → PepFlow structure layers。

## 小数据过拟合

1. 1 个 case 做 sequence+structure backward smoke。
2. 4--8 个 case 使用 15%、30%、50% sequence masking 做 overfit。
3. 全部 40 对配对训练；每个 receptor 生成多个 sequence/structure samples，不只
   报告单个最低能量样本。
4. 训练集可报告 masked sequence recovery；真正 design evaluation 不使用 recovery
   或 native RMSD，因为该任务没有 native design。

## 损失

- masked-residue categorical flow/cross-entropy；
- peptide backbone/frame/torsion flow；
- receptor translation/rotation/χ flow；
- correct-state ranking、minimal-relaxation、shuffle/zero controls；
- 可选 interface-contact loss。Rosetta ΔG 仅用于训练后的独立排序，不直接反传。

## Rosetta ΔG 验证

所有生成复合物使用完全相同的结构清理、side-chain repack/minimization 和
`InterfaceAnalyzer` protocol。将 `dG_separated` 作为 Rosetta 界面 ΔG proxy，
数值越低越好。至少保存：

- 每个有效 sample 的 ΔG（`dG_separated`）；
- ΔG 的均值、中位数、分位数、最低值及 top-k 分布；
- `dSASA_int`、interface contacts、shape complementarity、clash/strain；
- peptide 几何有效性、序列多样性和相对 anchor 的 pocket movement；
- Rosetta 失败原因、有效结构数和失败率。

## 通过标准

- 小数据 masked sequence recovery 明显高于随机基线，且联合结构 flow 能稳定收敛；
- design samples 具有合理 peptide 几何、低 clash、有效界面和受控 pocket movement；
- 多个独立生成样本取得稳定偏低的 Rosetta ΔG，而不是只出现一个偶然低值；
- 模型排序与 Rosetta ΔG、interface contacts、dSASA 的方向具有统计相关性；
- shuffle/zero peptide 条件后，结构质量、模型 ranking 和 Rosetta ΔG 分布恶化。

Rosetta ΔG 是计算能量 proxy，不等同于实验结合自由能；本实验验证小数据 codesign
闭环和候选排序能力，不宣称大规模新序列泛化。
