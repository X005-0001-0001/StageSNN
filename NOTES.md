# 分支 
ef — 参照基线（transformer / n-gram）

## 已得数字（同掩码、同 eval 子集、next-char，n=13777）
| 模型 | PPL | acc |
|---|---|---|
| unigram | 382.5 | 11.2% |
| KN bigram | 59.8 | 24.8% |
| KN trigram | 38.2 | 32.9% |
| transformer 2L d=128 ctx=128 1ep | 44.2 | 39.8% |

## 工具
- snn_lm/eval_next.py：next-char 评估 + 温度校准 + 朴素 n-gram
- snn_lm/baselines_ref.py：KN 2/3/4-gram
- snn_lm/tref.py：字符级 transformer 参照（CPU，1 epoch ~92s）

## 注意
- 旧 mem 长链（PPL 1600-2400）是重建当前字符口径，不可与上述比较。
- 旧 br/mem/train.py 续训未拷贝 W_s/u（每段慢库清零）——新分支已修复。
