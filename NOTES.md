# 分支 res — 多时间尺度痕迹 + 训练读出（BP-free）

## 目标
在无 BP、无 t+1 目标注入前提下，用多时间尺度（tau=1/2/4/8 token）的痕迹 + 预测编码读出逼近 n-gram/transformer。

## 实现（br/res，保留历史文件）
- 读出：R (V, 4V)，NLMS/delta；目标可选 pspk（P 脉冲）/state（上一状态）/char（下一字符 one-hot）。
- 稀疏列索引（>1e-3）+ 阈值缓存（5x 加速）。

## 结果
- 训练集 next-char acc 21.4%（5k）、val 6.9%（5k）/ 9.5%（30k，data_mini）；**加法痕迹线性读出无法表达交互**（val 低于 unigram）。
- res2（痕迹 + P_d_sum + softmax 读出）：
  - 20k：PPL 1146.3 / acc 16.1%
  - 100k：PPL 719.7 / acc 20.8%（接近 bigram 23.9%）
- res3（+ P_d_sum ⊙ 前文痕迹 绑定特征）：20k 1116.8 / 17.6%（绑定仅 +1.5pt，P 表征弥散）。
- res5 在 data_real 100k：**902.6 / 19.8%**（同数据 mem 原读出 4.8%、ctrl 3.3%；bigram 24.8%）→ 读出修复带来 +15pt。
- res5（+ 前两字符 one-hot）：
  - 20k：1145.2 / 18.8%（res2 16.1%）
  - 100k：715.0 / **22.5%**（res2 20.8%；bigram 23.9%、trigram 26.0%）
- 对照（data_mini）：unigram 423.1/9.2%、KN bigram 98.6/23.9%、KN trigram 80.9/26.0%。

## 结论
- SNN 状态（P_d_sum）并入读出有实质增量（9.5%→20.8%）；绑定特征边际；**瓶颈是状态上下文≈1 token**。
- 代码：br/res（trace）、br/res2（state+softmax）、br/res3（binding）。
