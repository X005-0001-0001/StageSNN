# 分支 `opt` — numba 热路径优化（实验完成：等价但收益 ~2%，瓶颈不符预期）

## 假设
`_learn_jit` 中每活跃行做 3 次 V 长度稠密循环：惰性衰减、`wsum`（avail）、LTP 更新。
post 极稀疏（~2 个脉冲/步）→ 稀疏化 LTP；avail 用增量行质量替代逐行求和。

## 实现
- 稀疏 post：每次 learn 只遍历非零 post 索引（LTP / hetero）；
- 增量质量：`mass_w/mass_s`（每次调用前由 W/W_s 行和初始化，之后随 LTP 与惰性衰减增量维护）；
- 数值等价（big3_mem 快照 300tok）：Δlogits **6.2e-6**、argmax **1.0000**、ΔW 9.3e-8、Δγ2 5.1e-7、Δγc 4.7e-8、ltp 4.4e-7。

## 基准（稳态，numba）
| 数据/状态 | mem 基线 | opt | 加速 |
|---|---|---|---|
| real1（V=3000，fresh） | 15.57 ms/token | 15.42 | ~1% |
| big3_mem（V=2706） | 22.02 | 21.16 | ~4% |

## 结论
收益过小 → **瓶颈不在 LTP/avail**，而在每 micro-step 的 LIF/exp 稠密循环
（K×V×15 步 + exp）。真正的加速需重写 `_run_stream` 的向量化 LIF/步进（收益估计 2-3×，
风险高）。本分支保留为"已证等价"的优化原型；未合入主线。
