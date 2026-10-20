# 分支 ctx — 慢上下文痕迹（进行中）

## 背景
- 状态线性探针（next-char）：P_V 27.3%、P_d_sum 21.2%、act 18.6%、P_V+衰减上下文 29.0% → 架构上限≈bigram。
- 跨 token 记忆仅 NMDA 延迟驱动（tau 30 步≈0.5 token）+ W 递归；需显式跨 token 记忆。

## 方向
- 慢上下文痕迹 ctx（衰减 tau≈2-4 token 的字符码）并入读出特征；结合预测编码读出（P 脉冲目标 + NLMS）。
- 目标：超过 trigram（PPL<38）；不覆盖旧文件，按版本命名。

## 状态
骨架已建（br/ctx = pred + ctx 特征）；3k 测试 PPL≈2931（近均匀），待稀疏高维展开。
