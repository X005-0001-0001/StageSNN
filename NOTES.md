# 分支 `diag` — 长程退化诊断（30k 末端塌陷定位）

## 问题
data_mini 30k tokens 轨迹中，val 在 25k→30k 之间骤降（ppl 2151→2836，acc 13.89%→9.68%；
旧门控规则同样出现 2204→2714 / 13.97%→9.86%）。

## 方法
- 逐 5k 保存快照（`br/diag/curve.py` 改造，numba 后端）；
- `diag_analysis.py`：逐快照评估 + 逐 token NLL 差分 + 前后半 val 分组；
- `diag2.py`：按 target token 的 NLL 差分 + W 状态余弦/行变化 + 段分布统计。

## 结论
1. **段效应**：25–30k 训练窗为代码段（`{`=106、`\`=278；其它段 0–20/≤88），val 以空格/星号/换行为主。
2. **token 级**：中位 Δ=0；7.6% 位置 Δ>1 nat，全部集中于 id=4（空格，+2.19 nats）；前后半 val 同时劣化。
3. **W 状态**：5k 窗口间 cos≈0.30（W 近乎全换）→ 与 τ_W≈2400 tokens 一致。
4. 判定：**数据非平稳（段切换）× W 近因记忆**，非内核错配（numpy/numba 逐位一致）、非局部数据异常。
5. 由此引出 `mem`（双时间尺度巩固）与 `decay`（选择性遗忘）两个机制分支；`mem` 显著抑制塌陷。

## 工具与产物
- `snn_lm/br/diag/`（curve.py 逐块快照、diag_analysis.py、diag2.py）
- 快照：`br/diag/runs/curve_diag/snap_{0,5k,...,30k}.npz`（含 params sidecar）
- 输出：`diag_analysis.json`、`diag2.json`
