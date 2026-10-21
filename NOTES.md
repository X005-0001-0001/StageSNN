# 分支 pred — 预测编码读出（在线读出训练）

## 关键发现（2026-10-21 诊断）
- 旧读出在 token 自分窗内第 56 步读取（当前字符已注入），target 又是 ids[t] → 实际是"重建当前字符"，与 LM 不可比。
- 学习到的 γ2/γc 读出失效（同快照 P_V 直接解码当前字符 93.7%）：γ2 是乘法（cvec * d_deep）而非加法联想；base 反而略好。
- 真实下一字符基线：unigram 382.5、KN bigram 59.8、KN trigram 38.2、transformer(1ep) 101.4、transformer(8ep) 36.1。

## 方案
- 评估改为 next-char（shift+1）+ 温度校准。
- 读出改为自监督预测编码：状态 t → 下一 token 的 P 脉冲模式（或字符），在线 NLMS/delta（无 BP、无 t+1 目标注入），保留旧版本文件。

## 状态
骨架已建（br/pred：readout=pred、readout_feat=pv、read_target=pspk、eta_read=NLMS）；3k 近均匀（欠训练）。更有效的读出见 res2/res3（软最大化 + 状态特征）。

## 追加：机制在正确指标下的对照（real1 0→100k）
| 协议 | next PPL | next acc | 当前字符 acc |
|---|---|---|---|
| mem（tau_decay_w=36000） | 2497.7 | 4.8% | 11.9% |
| ctrl（tau_decay_w=1e18） | 2676.4 | 3.3% | 14.0% |
| unigram | 382.5 | 11.2% | — |

- mem 在 next-char 上仅小幅领先（PPL -7%、acc +1.5pt）；旧"重建"优势（18.9% vs 14.9%）在正确指标下不成立。
- 读出直接学 next-char（br/res2，100k）到 20.8%，接近 bigram → **瓶颈在状态而非读出**。
- 下一步：慢状态架构（sres 脉冲储层）。
