# 分支 
es — 多时间尺度痕迹 + 训练读出（BP-free）

## 目标
在无 BP、无 t+1 目标注入前提下，用多时间尺度上下文痕迹（tau=1/2/4/8 token）+ 预测编码读出逼近 n-gram/transformer。

## 实现（br/res，保留历史文件）
- 读出：R (V, 4V)，NLMS/delta，目标可选 pspk（P 脉冲）/state（下一状态）/char（下一输入字符 one-hot，数据流本身）。
- 稀疏列索引（>1e-3）+ 阈值化，5x 加速。

## 结果
- 训练集内 next-char acc 21.4%（5k）；val 6.9%（5k）/ 9.5%（30k，data_mini）→ **加法痕迹线性读出无法表达交互**，val 卡在 unigram。
- 诊断结论：线性特征上限≈unigram+；需要非线性上下文混合（储层/SNN 递归动力学）。
