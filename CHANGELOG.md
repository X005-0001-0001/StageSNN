# Changelog

本项目所有值得注意的变更均记录于此文件。版本号使用纯数字三段式（如 `1.0.0`），不使用 `v` 前缀。

---

## 1.0.0 — 2026-10-19

> 定位：**首版交付** —— elite 外壳（菜单 / 日志 / 数据约定）+ StageSNN 阶段链脉冲内核（纯前向、局部可塑性；无反向传播、无 PyTorch、CPU 可训）。

### 内容
- **外壳**：菜单 / 日志 / 数据约定（v5 messages 渲染 + 训练区间 spans；`select_training_file` / `render_messages` 风格沿用 312*elite 系列）。
- **内核**：K=5 延迟链 M（固定链权重 + SFA，中继每窗 ~1 脉冲）→ P 读出群（LIF、AMPA/NMDA 慢核）；M(i−k)→P 权重 `W[k]` 为唯一被学习的突触：局部 Hebbian `ΔW = η·dt·pre·avail·outer(coef, post)` + ATP 疲劳 + 物质约束 `avail = relu(1 − rowsum/B)`。
- **先验门控（默认开启）**：γ2 二阶突触（explaining-away）；γcopy 对角回指路由。
- **协议**：`logits[i]` 预测 `ids[i]`（上下文 `w_{i-1..i-K}` 驱动，无 t+1 目标注入）；掩码口径（仅 assistant 段计分）；温度仅训练切片自校准；ACC（温度无关）为第一指标。
- **数据管线**：v5 messages jsonl → 字符级 ids/mask；词表仅训练切分构建（防验证集泄露）；验证 = 100 条对话（seed 0 打乱）。
- **移除**：elite 原内核（并行复波 SSM + AdamW BP）与 RL/GRPO 功能（依赖反向传播，红线不允许）。

### 边界与注意
- 纯 NumPy 路径（逐事件全阵衰减）；V=3000 训练约 30~40 ms/token（CPU）。
- 唯一必需依赖：numpy。
- 文件字节原样收录（无 BOM、LF）。

---
