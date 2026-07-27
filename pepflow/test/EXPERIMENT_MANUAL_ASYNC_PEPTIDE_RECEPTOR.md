# 实验手册：异步 peptide/receptor flow 小数据过拟合

## 目标

在现有 80 个真实 cross-holo 状态上验证：peptide flow 与 receptor flow
可以使用不同的 flow time/积分步长异步推进，并通过条件 adapter 联合优化。
peptide 内部的 sequence embedding、backbone/frame、torsion 暂时仍使用同一
个 peptide clock，不拆成异步子流。

## 固定设置

- 数据：`joint-small-batch/20260727-022909+0800`，先使用 1--4 个 case 做
  smoke，再使用全部 40 对配对过拟合。
- 初始化：官方 PepFlow `model1.pt`、DynamicBind checkpoint；不得随机初始化
  PepFlow 主干。
- 参数：保持当前约 19.3M trainable parameters；sequence masking 关闭。
- 时间轴：每个样本独立采样 `t_p`、`t_r`；建议 `N_p=2N_r`，即 peptide
  每推进两次，receptor 推进一次。每次 receptor update 前交换一次
  peptide embedding/pocket context。
- 训练：每个 state 的 peptide flow loss、receptor translation/rotation/χ
  loss、正负 endpoint ranking、minimal-relaxation、shuffle/zero control。

## 实现约束

1. `peptide_flow(x_p,t_p)` 内部 sequence/structure 模块共享 `t_p`。
2. `receptor_flow(x_r,t_r, adapter(h_p))` 使用独立 `t_r`；adapter 不 detach，
   使 receptor loss 能反向更新 peptide 分支。
3. 只在明确同步点交换条件，不把两个 flow 的状态直接拼接成一个伪状态。
4. 记录每个 flow 的 time、step、loss 和显存；提供同步基线（`t_p=t_r`、
   相同步数）作消融。

## 生成与验收

- 从 anchor receptor + corrupted peptide 开始，按 `N_p=2N_r` 交替积分，输出
  peptide sequence（固定输入序列）和 peptide/receptor 复合物结构。
- 过拟合验收：训练集 peptide pose RMSD、pocket RMSD、correct-state
  accuracy、ranking margin 均优于初始化；shuffle 后 accuracy/margin 下降；
  receptor movement 不出现异常爆炸。
- 异步版本相对同步基线不能显著恶化 peptide RMSD 或 pocket RMSD；若改善，
  必须同时报告 wall time、显存峰值和每条 flow 的收敛曲线。
- 结果目录必须保存命令、commit、GPU、time schedule、checkpoint SHA、
  `training_metrics.csv`、`evaluation_metrics.csv`、结构文件和失败日志。

## 必做消融

| 版本 | peptide clock | receptor clock | 目的 |
| --- | --- | --- | --- |
| sync | 同步 | 同步 | 当前联合 flow 基线 |
| async-2:1 | `N_p=2N_r` | 独立 `t_r` | 主实验 |
| async-1:2 | `N_p=N_r/2` | 独立 `t_r` | 检查 receptor 慢更新是否必要 |
| no-adapter | 独立 | 独立 | 证明条件耦合确实贡献结果 |

本实验只证明异步 peptide/receptor flow 的可训练性，不宣称 sequence
design 或大规模泛化。
