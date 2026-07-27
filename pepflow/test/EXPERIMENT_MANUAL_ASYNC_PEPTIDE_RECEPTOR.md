# 实验手册：异步 peptide/receptor flow 小数据过拟合

## 目标

在现有 80 个真实 cross-holo 状态上验证 peptide flow 与 receptor flow 的异步
联合生成。peptide 内部的 sequence embedding、backbone/frame、torsion 暂时共享
同一个 `peptide` clock，不拆成异步子流。

所有超参数只允许来自 `pepflow/configs/joint_async_codesign.yaml`。运行入口只接收
`--experiment-config`，不得在代码或命令行中散落 `N_p`、`N_r`、时间维度和 loss
权重。YAML 使用命名 preset 和显式数值，首版不支持任意表达式或 `eval`；目前没有
必须依赖表达式才能描述的 schedule，这样更安全且易于复现。

## 多时钟 conditioning

每个 clock 先用高维 Fourier embedding 编码，再加入 learned clock-type embedding，
形成带名字的 time token。当前 token 集为 `{peptide, receptor}`：

- peptide module 显式接收并注入 `E(t_peptide)` 和 `E(t_receptor)`；
- receptor module 显式接收并注入 `E(t_receptor)` 和 `E(t_peptide)`；
- own-time、peer-time 与 state embedding 按固定顺序 concat，再经过小型 projection
  MLP 注入各自模块，不只放进共享 adapter；当前不增加 attention 模块。

建议每个 scalar clock 使用 256 维 Fourier embedding、32 维 clock-type embedding，
concat 后投影到 512 维。每个模块使用三个有序 clock slot，未使用 slot 填零；当前
分别放 own-time、peer-time、一个空 slot。未来拆分 `peptide_seq`/`peptide_struct`
时只需注册新 clock 并填入预留 slot，projection 输入维度和旧 checkpoint 接口不变。
只有 clock 数量超过预留容量后，才另做 attention/gated fusion 消融。

## Flow matching 与 schedule

训练时分别采样 `t_peptide`、`t_receptor` 并监督各自 vector field；推理时的
`N_p/N_r` 只是 ODE 数值积分 schedule，不是 flow matching 本身的假设。
`N_p=2N_r` 只有在 peptide vector field 曲率更大或 receptor 应缓慢松弛时才可能
更好，因此不能直接设为唯一方案。

首轮使用以下 preset：

| preset | 方法 | 用途 |
| --- | --- | --- |
| `sync_heun_1to1` | 同一时间网格，Heun | 同步基线 |
| `async_strang_2to1` | receptor 半步、两次 peptide step、receptor 半步 | 推荐固定异步主实验 |
| `async_heun_1to1` | 独立 clocks、相同步数 | 分离“异步时间”和“步数比例” |
| `adaptive_coupled` | 各 flow 按局部误差调步，在同步点交换条件 | 后续探索 |

推荐先以 `async_strang_2to1` 过拟合，因为对称 splitting 比简单“peptide 两步、
receptor 一步”更稳定。adaptive 不一定更好：局部误差小不代表跨模块条件已经更新，
还会增加非确定性和评估成本；只有固定 schedule 通过后才比较，并限制最大 clock
lag、最小/最大步长和同步间隔。

## 训练与验收

- 初始化：官方 PepFlow `model1.pt` 和 DynamicBind checkpoint；以现有 19.3M
  trainable parameters 为基线，新增 time encoder 参数单独统计，sequence masking 关闭。
- 损失：peptide flow、receptor translation/rotation/χ flow、endpoint ranking、
  minimal-relaxation、shuffle/zero controls。
- adapter 不 detach；两个模块都显式使用两个 time embedding，receptor loss 可反向
  更新 adapter 与允许训练的 PepFlow 层。
- 先 1--4 个 case smoke，再使用全部 40 对配对过拟合；同步和异步版本使用相同
  初始 checkpoint、seed 和 function-evaluation budget。
- 验收：peptide pose、pocket RMSD、correct-state margin/accuracy 优于初始化；
  shuffle 后选择优势下降；无 receptor movement 爆炸。
- 必须同时报告 NFE、wall time、显存峰值、每条 flow 的误差/收敛曲线和实际同步点。

本实验只验证 peptide/receptor 异步 flow 的可训练性，不宣称 sequence design 或
大规模泛化。
