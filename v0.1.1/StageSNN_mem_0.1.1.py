# ================================================================================================
# StageSNN mem 0.1.1 —— 单文件：外壳沿用 3120012elite.py 的菜单/日志/数据约定（v5 messages 渲染 +
# 训练区间 spans），内核为 StageSNN 阶段链脉冲内核（numpy 基准 + 可选 numba 热路径；无 torch、
# 无 BP、无两次前向对比、无目标注入、无外部队列）。
#
# 版本约定（与 312*elite 系列同风格）：
#   每次改动 = 复制本文件到新版本号（StageSNN_mem_0.1.1.py ...），文件头加一行变更记录，
#   通过保存文件保持历史，不原地改写旧版本。
#   CHANGELOG:
#     1.0.0  首版：elite 外壳（菜单/日志/select_training_file/render_messages/spans）+ StageSNN 内核
#            （K=5 延迟链 M + 读出群 P + 稀疏 Hebbian/ATP + γ2 二阶突触 + γcopy 对角回指路由）。
#            移除 elite 原内核（并行复波 SSM + AdamW BP）与 RL/GRPO 功能（依赖反向传播，红线不许可）。
#     1.0.1  数值阈值/精度下限全部提升为具名开关（去除散落魔数）；菜单续训自动推算下一分块偏移
#            （offset = 上块 offset + tokens_done）；预览长度改为开关。
#     1.1.0  内核加速：可选 numba 热路径（整段流一次 JIT；与 numpy 基准逐元素等价验证通过：
#            logits Δ~1e-6、argmax 100%、ΔW~1e-8）；新增 DECAY_AMORT_STEPS 被动衰减摊销开关
#            （数学等价，W/γ2 全阵衰减频率降低 → V=3000 实测 37→10.7 ms/token，3.5×）。
#            numba 缺失时自动回退 numpy。
#     1.1.1  numba 热路径 v2：稀疏脉冲列表化（LIF 直出 id；hist 链/写入/M_a/pre/Mc_read/
#            spk_acc/win_spk 全按脉冲更新；hist 布尔为真值源 + 快表溢出扫描回退，严格等价）。
#            等价复测 logits Δ≤1e-5、argmax 100%、ΔW≤3e-8、Δγ2≤8e-9；同刻对照 numpy 1.3~1.8×。
#     1.1.2  ① numba 热路径新增惰性逐行衰减（LAZY_DECAY，默认开启）：事件只计数，行被读/写时按
#            (ev-row_ev) 补乘 dec**k，结束物化全表；与逐事件衰减严格同语义（big3+门控 300tok：
#            Δlogits 3.4e-5、argmax 100%、ΔW 8.9e-7、Δγ2 8.6e-6；eager 回退路径保持 3e-8 级）。
#            训练期热点衰减流量降为 ~0：V=3000 amort=25 时 17.2→9.5 ms/tok；amort 开销≈0（可放心用小值）。
#            置 LAZY_DECAY=False 可回退逐事件全阵衰减（更多一次极限对齐验证时使用）。
#            ② 修复两处缩进嵌套缺陷（atp 恢复循环吞掉 counter/衰减/return；P_r_all 块被 P_I 衰减
#            循环吞掉放大 3000×）——修复后 eager 路径与 1.1.1 逐位一致（ΔW 2.98e-8）。
#
#     1.2.0  先验门控自适应化：γ2/γcopy 的固定绝对阈值（GATE2_KAPPA=0.5 / COPY_KAPPA=0.3）替换为
#            BCM 式滑动修改阈值 —— γ2 更新用 (post - post_bar)、γcopy 用 (Mc_read[:, idx] - mc_bar)；
#            post_bar（逐神经元）/mc_bar（逐级、获胜列）为各自门 τ（TAU_GATE2 / TAU_COPY）同尺度的
#            EMA 基线：零新增参数、零手调常数；更新天然零均值，长训不再单向漂移。
#            动机与验证（研究线）：real1 长训后门控读出项反向伤害（关 γc：acc 12.30%→14.19%；
#            全关 →14.31%）；小语料 30k 轨迹旧规则 acc 15.2%→9.86%（γc_max 0.16→0.25 爬升）。
#            新规则重验 P3 判据：ca_lag4 191/191=1.000（与旧规则一致）、acc 37.62%（旧 37.64%）、
#            cal PPL 111.92（旧 108.40，仍远优 bigram 306.6 / trigram 128.6）。
#            等价性：big3+门控 300tok Δlogits 1.5e-5、argmax 100%、ΔW 8.9e-7（与 1.1.2 同级）。
#            旧快照兼容：load 自动过滤已移除的 kappa 参数键；post_bar/mc_bar 随快照保存。
#
#     mem 0.1.0  实验分支（自 bio 1.2.0）：双时间尺度巩固（fast W + slow W_s，快/慢两库）。
#            机制：逐行使用度 EMA u（τ=TAU_DECAY_W）→ 巩固占比 rho = u_act/(u_act+mean(u_act))
#            （同批活跃行内相对，零新增常数）；LTP 同一 dW 按 (1-rho)/rho 分入 W/W_s；
#            W_s 无衰减；容量按 W+W_s 统计；读出差值相加（W+W_s）。
#            验证：big3 重训 cal PPL 106.70（bio 1.2.0 为 111.92；旧 kappa 规则 108.40）、
#            acc 37.75%、ca_lag4 191/191=1.000；小语料 30k 末端 ppl 2204.8/acc 14.57%
#            （同配置对照 2836.0/9.68%——近因塌陷被显著抑制）。
#            注意：numba 内核未移植（本分支 BACKEND='numpy'；--backend numba 自动回退 numpy）。
#            W_s/u 随快照保存；旧快照兼容（缺省为零）。
#
#     mem 0.1.1  恢复 numba 双后端：慢库/使用度/rho 已移植进 numba 内核（与 numpy 严格同构）。
#            等价性（big3_mem 快照 300tok，含 W_s/u 非零）：Δlogits 3.8e-6、argmax 一致率 1.0000、
#            ΔW 7.8e-8、Δγ2 5.1e-7、Δγc 7.5e-9、ltp relΔ 2.0e-9；numba 约 2.7× 加速。
#            BACKEND 恢复 'auto'；移除 0.1.0 的 numba 回退提示与 main 兜底。
#
# 内核结构（与 snn_lm 研究线完全同构，此文件为规范版本）：
#   输入爆发（one-hot token, burst_offsets 六脉冲于一窗内）
#   → M 延迟链（K 级，每级延迟一窗，w_chain 固定 + SFA 保证每窗 ~1 脉冲中继）
#   → P 读出群（LIF, NMDA 慢核 τr/τd），M→P 权重 W[k] 通过局部 Hebbian 规则学习：
#       ΔW = η·dt·pre·avail·outer(coef, post)，post 为钙式脉冲痕迹；avail=relu(1-rowsum/B) 物质约束
#   先验门控：γ2（二阶突触, explaining-away 抑制）与 γcopy（对角回指路由），默认开启。
#   评分语义：logits[i] 预测 ids[i]（上下文 w_{i-1..i-K} 驱动，无输入泄漏），非 ids[i+1]。
#   温度：仅在训练集切片上自校准（cal）；评估同时报 raw@T_READOUT；ACC 温度无关，为第一指标。
# ================================================================================================

import argparse
import json
import os
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ================================================================================================
# 统一开关区（所有可调参数集中在此；代码中不得出现这些值以外的魔数）
# ================================================================================================
PROJECT_NAME = "StageSNN"
BRANCH = "mem"
VERSION = "0.1.1"

# ---- 文件与目录 ----
DATA_FILE = "dsr1_v5_0001.jsonl"      # 默认语料（v5 messages jsonl，位于本文件同目录）
DATA_DIR = "stage_data"               # 数据管线输出目录（ids/mask/meta）
RUNS_DIR = "stage_runs"               # 训练运行目录（快照/指标）
LOG_FILE = "training_log.txt"         # 会话日志
BRAIN_FILE = "brain_stage.npz"        # 菜单“保存模型”默认权重文件
SEED = 0

# ---- 分词（字符级；词表仅从训练切分构建，防验证集泄露） ----
VOCAB_SIZE = 3000
SPECIAL_TOKENS_LIST = ["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>",
                       "<|user|>", "<|assistant|>", "<|end|>"]
NORMALIZE_MODE = "none"               # 'none' 原样 | 'nfkc' 仅 NFKC（v5 历史模式）
TRAIN_ON_REASONING = True             # assistant 的 reasoning_content 是否计入训练区间

# ---- 数据切分 ----
VAL_CONVS = 100                       # 验证集对话条数（训练集 = 其余全部）
SPLIT_SEED = 0

# ---- 内核：时间结构 ----
MODEL_WINDOW_MS = 60                  # 一窗 = 一个 token 的时间槽（ms 步）
MODEL_K = 5                           # 延迟链级数 = 上下文深度
MODEL_BURST_OFFSETS = (2, 6, 10, 14, 18, 22)   # 输入六脉冲在一窗内的时刻
LEARN_EVERY_MS = 4                    # 可塑性更新周期（ms）
READ_MARGIN = 4                       # 读出时刻 = 窗末前 4ms

# ---- 内核：神经元生物物理（LIF + K+ shunt + SFA） ----
V_REST = -65.0
V_TH = -50.0
V_RESET = -66.0
TAU_M = 20.0
T_REF = 2
SHUNT_GAIN = 6.0
SHUNT_C = -45.0
SHUNT_S = 1.5
TAU_AMPA = 5.0

# ---- 内核：NMDA 慢核 / 痕迹 ----
TAU_NMDA_R = 15.0
TAU_NMDA_D = 30.0
TAU_TRACE = 10.0
TAU_CA = 8.0
THETA_POST = -63.0
POST_W_SPK = 1.0
POST_W_ANALOG = 0.0
TAU_ADAPT = 30.0
G_ADAPT_M = 8.0
G_ADAPT_P = 0.0

# ---- 内核：固定突触权重 ----
W_IN_M = 80.0
W_CHAIN = 160.0
W_IN_P = 40.0
G_NMDA = 6.0

# ---- 内核：可塑性（Hebbian + 资源约束 + ATP） ----
ETA_LTP = 0.0015
ETA_HETERO = 0.0
TAU_DECAY_W = 36000.0
BUDGET_B = 1.0
TAU_ATP = 3000.0
C_SPK = 0.004
C_PLAST = 0.02
DECAY_AMORT_STEPS = 100               # 被动衰减摊销步数（数学等价：因子按 e^{-N·dt/τ} 精确放大）
LAZY_DECAY = True                     # True=惰性逐行衰减（默认，见 CHANGELOG 1.1.2；语义与逐事件
                                      # 衰减严格一致，长跑等价性验证于 snn_lm/probe.py；置 False 回退）

# ---- 计算后端（numba 可选；缺失/编译失败自动回退 numpy） ----
BACKEND = "auto"                      # 'auto' | 'numpy' | 'numba'

# ---- 内核：先验门控 γ2（二阶突触 / explaining-away） ----
USE_GAMMA2 = True
ETA_GATE2 = 0.05
TAU_GATE2 = 12000.0

# ---- 内核：先验门控 γcopy（对角回指路由） ----
USE_COPY_GATE = True
ETA_COPY = 0.02
TAU_COPY = 1500.0
TAU_COPY_READ = 35.0

# ---- 内核：门控痕迹 ----
GATE_NEAR_STAGES = 2
TAU_GATE = 10.0                       # G（第 0 级脉冲痕迹）衰减常数
TAU_GATE_READ = 30.0                  # Gs（读出用痕迹）衰减常数

# ---- 数值阈值与口径（活动判定/统计下限；代码中不得散落这些魔数） ----
POST_ACTIVE_THRESHOLD = 1e-4          # post 痕迹高于此值才触发可塑性更新
PRE_ACTIVE_THRESHOLD = 1e-3           # pre 痕迹高于此值视为活跃突触前
GATE_ACTIVE_THRESHOLD = 0.05          # G 高于此值的行参与门控学习
GATE_READ_THRESHOLD = 0.01            # Gs 高于此值的行参与门控读出
DIV_EPS = 1e-9                        # 除法保护
POST_SUM_THRESHOLD = 1e-6             # post 总痕迹高于此值才进入门控学习
W_REL_FLOOR = 1e-12                   # w_sum 相对变化量分母保护
LOG_PROB_FLOOR = 1e-30                # 评估对数下限
GEN_TEMPERATURE_MIN = 0.05            # 生成温度下限
W_NNZ_EPS = 1e-9                      # 权重“非零”判定阈值
GAMMA_NNZ_EPS = 1e-6                  # 门控“激活”判定阈值
BYTES_PER_MB = 1e6                    # 统计显示换算

# ---- 菜单预览（数据检查） ----
PREVIEW_CHARS = 200                   # 预览渲染文本截断字符数
PREVIEW_IDS = 60                      # 预览 id/mask 截断长度

# ---- 温度与评估 ----
T_READOUT = 3.0                       # 默认读出温度（未标定时的 raw 值）
TEMPERATURE_GRID = (0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                    0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
CAL_TOKENS = 4000                     # 温度自校准使用的训练集切片长度
EVAL_TOKENS_LIMIT = 15000             # 评估/验证使用的 val token 上限（0 = 全部）
BASELINE_MAX_N = 3                    # n-gram 基线最高阶
BASELINE_TRAIN_TOKENS = 60000         # n-gram 基线可见的训练切片长度（与训练预算对齐）

# ---- 训练 ----
TRAIN_CHUNK_TOKENS = 50000            # 单次运行训练 token 数（分块防超时）
CKPT_EVERY_TOKENS = 5000              # 快照写盘周期
EPOCHS = 1

# ---- 生成（对话/续写） ----
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
GEN_MAX_NEW_TOKENS = 200

DEF_NGRAM_LAMBDA = 0.6                # 插值 n-gram 基线权重（probe 约定）
DEF_NGRAM_ALPHA = 0.1                 # 加法平滑

# ================================================================================================
# 日志（elite 外壳约定：training_log.txt 追加 + 控制台同步）
# ================================================================================================
_LOG_FILE = None
_LOG_FLUSH_COUNTER = 0
_LOG_FLUSH_INTERVAL = 50


def init_log_file():
    global _LOG_FILE
    if _LOG_FILE is None:
        _LOG_FILE = open(LOG_FILE, "a", encoding="utf-8")
        _LOG_FILE.write(f"\n--- 新会话开始：{datetime.now()} ---\n")
        _LOG_FILE.flush()


def log_and_print(*args, **kwargs):
    global _LOG_FLUSH_COUNTER
    end = kwargs.get("end", "\n")
    msg = " ".join(str(a) for a in args) + end
    sys.stdout.write(msg)
    sys.stdout.flush()
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg)
        _LOG_FLUSH_COUNTER += 1
        if _LOG_FLUSH_COUNTER >= _LOG_FLUSH_INTERVAL:
            _LOG_FILE.flush()
            _LOG_FLUSH_COUNTER = 0


def close_log_file():
    global _LOG_FILE
    if _LOG_FILE is not None:
        _LOG_FILE.flush()
        _LOG_FILE.close()
        _LOG_FILE = None


# ================================================================================================
# 工具函数（elite 外壳约定）
# ================================================================================================
_VOCAB_NORMALIZE_MODE = None


def _active_normalize_mode():
    return _VOCAB_NORMALIZE_MODE if _VOCAB_NORMALIZE_MODE is not None else NORMALIZE_MODE


def normalize_text(text):
    if _active_normalize_mode() == "nfkc":
        return unicodedata.normalize("NFKC", text)
    return text


def _messages_from_record(rec):
    """取出一条数据的 messages（v5 唯一格式；兼容纯 list）。"""
    if isinstance(rec, list):
        msgs = rec
    else:
        msgs = rec.get("messages")
        if msgs is None:
            raise ValueError("记录缺少 messages 字段（v5 仅支持 {'messages': [...]}）")
    for _m in msgs:
        _r = _m.get("role")
        if _r not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"非法 role {_r!r}；只允许 system/user/assistant/tool")
        if _m.get("content") is None and not _m.get("tool_calls"):
            raise ValueError("消息既没有 content 也没有 tool_calls")
    return [dict(m) for m in msgs]


def _render_tools(tools):
    if not tools:
        return ""
    return "<|user|>[工具]" + json.dumps(tools, ensure_ascii=False) + "<|end|>"


def _render_tool_calls(tcs):
    if not tcs:
        return ""
    return json.dumps(tcs, ensure_ascii=False)


def render_messages(messages, tools=None, add_generation_prompt=False):
    """渲染 messages → (text, spans)；spans 为需要计分的字符区间（assistant 段）。

    与 elite 原版逐字符一致：system/user/tool 段不计分；assistant 段
    （reasoning_content 可选 + content + tool_calls + <|end|>）与文末 <|eos|> 计分。
    """
    parts = []
    spans = []
    pos = 0

    def emit(s, train=False):
        nonlocal pos
        n = normalize_text(s)
        if train and len(n) > 0:
            spans.append((pos, pos + len(n)))
        parts.append(n)
        pos += len(n)

    emit("<|bos|>")
    if tools:
        emit(_render_tools(tools))
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        content = "" if content is None else str(content)
        if role == "system":
            emit("<|user|>[系统]" + content + "<|end|>")
        elif role == "user":
            emit("<|user|>" + content + "<|end|>")
        elif role == "tool":
            tid = msg.get("tool_call_id")
            emit("<|user|>[工具结果" + ((" id=" + str(tid)) if tid else "") + "]" + content + "<|end|>")
        elif role == "assistant":
            emit("<|assistant|>")
            seg = []
            rc = msg.get("reasoning_content")
            if TRAIN_ON_REASONING and rc:
                seg.append(str(rc))
            seg.append(content)
            tc = _render_tool_calls(msg.get("tool_calls"))
            if tc:
                seg.append(tc)
            emit("".join(seg) + "<|end|>", train=True)
    if add_generation_prompt:
        emit("<|assistant|>")
    else:
        emit("<|eos|>", train=True)
    return "".join(parts), spans


def select_training_file():
    """扫描当前目录中的 .txt / .jsonl 文件，让用户选择数据集文件（elite 约定）。"""
    files = []
    for f in sorted(os.listdir(".")):
        if f.startswith("."):
            continue
        if f.endswith(".txt") or f.endswith(".jsonl"):
            try:
                size = os.path.getsize(f)
            except OSError:
                size = 0
            files.append((f, size))
    if not files:
        log_and_print("当前目录未找到 .txt 或 .jsonl 文件。")
        return None, None
    print("\n可用数据集文件：")
    for i, (name, size) in enumerate(files, 1):
        if size >= 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        elif size >= 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {i}. {name} ({size_str})")
    while True:
        raw = input("请输入文件编号: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("输入无效，请输入数字。")
            continue
        if idx < 1 or idx > len(files):
            print(f"编号超出范围（1-{len(files)}）。")
            continue
        selected = files[idx - 1][0]
        log_and_print(f"已选择文件: {selected}")
        return selected, os.path.splitext(selected)[1].lower()


def _read_convs(path):
    convs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            convs.append(_messages_from_record(rec))
    return convs


def _split_convs(convs, val_convs, seed):
    n = len(convs)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    val_set = set(idx[:val_convs].tolist())
    train_convs = [convs[i] for i in range(n) if i not in val_set]
    val_convs_list = [convs[i] for i in idx[:val_convs].tolist()]
    return train_convs, val_convs_list, val_set


# ================================================================================================
# 字符级词表（替代 elite FastBPE；只用训练切分构建，防泄露）
# ================================================================================================
class CharVocab:
    def __init__(self):
        self.itos = []                 # id → 字符/特殊符
        self.stoi = {}                 # 字符 → id
        self.specials = list(SPECIAL_TOKENS_LIST)
        self.vocab_size = 0

    def train(self, texts, verbose=True):
        cc = Counter()
        for t in texts:
            cc.update(t)
        n_special = len(self.specials)
        budget = max(int(VOCAB_SIZE) - n_special, 1)
        top = [c for c, _ in cc.most_common(budget)]
        self.itos = self.specials + top
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        if verbose:
            covered = sum(cc[c] for c in top)
            total = sum(cc.values())
            log_and_print(f"[词表] 训练切分唯一字符 {len(cc)}，入表 {len(top)}，"
                          f"字符覆盖率 {covered / max(total, 1):.4%}，V={self.vocab_size}")

    def encode(self, text):
        unk = self.stoi["<|unk|>"]
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids, skip_special=True):
        out = []
        for i in ids:
            c = self.itos[int(i)] if 0 <= int(i) < self.vocab_size else "<|unk|>"
            if skip_special and c in self.specials:
                continue
            out.append(c)
        return "".join(out)

    def save(self, path):
        Path(path).write_text(json.dumps(
            {"specials": self.specials, "itos": self.itos,
             "normalize_mode": _active_normalize_mode(), "vocab_size": self.vocab_size},
            ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        v = CharVocab()
        v.specials = d["specials"]
        v.itos = d["itos"]
        v.stoi = {c: i for i, c in enumerate(v.itos)}
        v.vocab_size = d["vocab_size"]
        globals()["_VOCAB_NORMALIZE_MODE"] = d.get("normalize_mode", NORMALIZE_MODE)
        return v


# ================================================================================================
# 数据管线：v5 jsonl → 字符流 ids + 训练掩码（spans）+ 防泄露切分
# ================================================================================================
def build_dataset(src_path, data_dir, val_convs, seed, preview=0):
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    convs = _read_convs(src_path)
    train_convs, val_convs_list, val_set = _split_convs(convs, val_convs, seed)
    log_and_print(f"[数据] 总对话 {len(convs)}；训练 {len(train_convs)}；验证 {len(val_convs_list)}"
                  f"（seed={seed}；验证集对话索引一次成型，train/vocab 均不含其任何字符流）")

    # 词表：仅训练切分
    train_texts = []
    for msgs in train_convs:
        for m in msgs:
            c = m.get("content")
            if c is not None:
                train_texts.append(normalize_text(str(c)))
    vocab = CharVocab()
    vocab.train(train_texts)

    def encode_convs(msgs_list):
        ids_all, mask_all = [], []
        for msgs in msgs_list:
            text, spans = render_messages(msgs)
            ids = vocab.encode(text)
            mask = np.zeros(len(text), dtype=np.int8)
            for a, b in spans:
                mask[a:b] = 1
            # 字符级 1:1，spans 与字符位置严格对齐
            ids_all.extend(ids)
            mask_all.append(mask)
        if mask_all:
            return np.asarray(ids_all, dtype=np.int32), np.concatenate(mask_all)
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int8)

    tr_ids, tr_mask = encode_convs(train_convs)
    va_ids, va_mask = encode_convs(val_convs_list)

    np.save(out / "train_ids.npy", tr_ids)
    np.save(out / "train_mask.npy", tr_mask)
    np.save(out / "val_ids.npy", va_ids)
    np.save(out / "val_mask.npy", va_mask)
    vocab.save(out / "vocab.json")

    tr_unk = float((tr_ids == vocab.stoi["<|unk|>"]).mean()) if len(tr_ids) else 0.0
    va_unk = float((va_ids == vocab.stoi["<|unk|>"]).mean()) if len(va_ids) else 0.0
    meta = {
        "src": str(src_path),
        "n_conversations": len(convs),
        "train_convs": len(train_convs),
        "val_convs": len(val_convs_list),
        "val_conv_indices": sorted(int(i) for i in val_set),
        "train_tokens": int(len(tr_ids)),
        "val_tokens": int(len(va_ids)),
        "train_masked_frac": float(tr_mask.mean()) if len(tr_mask) else 0.0,
        "val_masked_frac": float(va_mask.mean()) if len(va_mask) else 0.0,
        "train_unk_frac": tr_unk,
        "val_unk_frac": va_unk,
        "seed": seed,
        "val_convs_arg": val_convs,
        "vocab_size": vocab.vocab_size,
        "normalize_mode": _active_normalize_mode(),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    log_and_print(f"[数据] 完成：V={vocab.vocab_size} train_tokens={len(tr_ids)}"
                  f"（masked {meta['train_masked_frac']:.1%}, unk {tr_unk:.2%}）"
                  f" val_tokens={len(va_ids)}（masked {meta['val_masked_frac']:.1%}, unk {va_unk:.2%}）")
    log_and_print(f"[数据] 输出目录: {out}")

    # 防泄露自检：验证对话索引与训练索引不相交（由构造保证，打印留痕）
    tr_idx_set = set(range(len(convs))) - val_set
    assert not (tr_idx_set & val_set), "切分异常：训练/验证对话索引相交"
    log_and_print(f"[数据] 防泄露自检通过：训练/验证对话索引不相交（验证 {len(val_set)} 条）")

    if preview:
        for k, msgs in enumerate(val_convs_list[:preview]):
            text, spans = render_messages(msgs)
            ids = vocab.encode(text)
            m = np.zeros(len(text), dtype=np.int8)
            for a, b in spans:
                m[a:b] = 1
            show = text[:PREVIEW_CHARS].replace("\n", "⏎")
            log_and_print(f"[预览] val#{k} 渲染前 {PREVIEW_CHARS} 字: {show}")
            log_and_print(f"[预览] val#{k} ids 前 {PREVIEW_IDS}: {ids[:PREVIEW_IDS]}")
            log_and_print(f"[预览] val#{k} mask 前 {PREVIEW_IDS}: {m[:PREVIEW_IDS].tolist()}")
    return out


def _load_stage_data(data_dir):
    d = Path(data_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    tr = np.load(d / "train_ids.npy")
    va = np.load(d / "val_ids.npy")
    trm = np.load(d / "train_mask.npy") if (d / "train_mask.npy").exists() else None
    vam = np.load(d / "val_mask.npy") if (d / "val_mask.npy").exists() else None
    vocab = CharVocab.load(d / "vocab.json")
    return tr, trm, va, vam, vocab, meta


# ================================================================================================
# 内核：Params（由开关区组装；不得出现其他魔数）
# ================================================================================================
@dataclass
class Params:
    window_ms: int = MODEL_WINDOW_MS
    K: int = MODEL_K
    V_rest: float = V_REST
    V_th: float = V_TH
    V_reset: float = V_RESET
    tau_m: float = TAU_M
    t_ref: int = T_REF
    shunt_gain: float = SHUNT_GAIN
    shunt_c: float = SHUNT_C
    shunt_s: float = SHUNT_S
    tau_ampa: float = TAU_AMPA
    tau_nmda_r: float = TAU_NMDA_R
    tau_nmda_d: float = TAU_NMDA_D
    tau_trace: float = TAU_TRACE
    burst_offsets: tuple = MODEL_BURST_OFFSETS
    w_in_m: float = W_IN_M
    w_chain: float = W_CHAIN
    w_in_p: float = W_IN_P
    g_nmda: float = G_NMDA
    theta_post: float = THETA_POST
    tau_ca: float = TAU_CA
    post_w_spk: float = POST_W_SPK
    post_w_analog: float = POST_W_ANALOG
    tau_adapt: float = TAU_ADAPT
    g_adapt_m: float = G_ADAPT_M
    g_adapt_p: float = G_ADAPT_P
    eta_ltp: float = ETA_LTP
    eta_hetero: float = ETA_HETERO
    tau_decay_w: float = TAU_DECAY_W
    decay_amort_steps: int = DECAY_AMORT_STEPS
    budget_B: float = BUDGET_B
    tau_atp: float = TAU_ATP
    c_spk: float = C_SPK
    c_plast: float = C_PLAST
    learn_every_ms: int = LEARN_EVERY_MS
    read_margin: int = READ_MARGIN
    T_readout: float = T_READOUT
    gate_near_stages: int = GATE_NEAR_STAGES
    tau_gate: float = TAU_GATE
    tau_gate_read: float = TAU_GATE_READ
    tau_gate2: float = TAU_GATE2
    eta_gate2: float = ETA_GATE2 if USE_GAMMA2 else 0.0
    tau_copy_read: float = TAU_COPY_READ
    tau_copy: float = TAU_COPY
    eta_copy: float = ETA_COPY if USE_COPY_GATE else 0.0

    def to_json(self):
        d = asdict(self)
        d["burst_offsets"] = list(self.burst_offsets)
        return d


def build_params(**overrides):
    names = {f.name for f in fields(Params)}
    kw = {k.replace("-", "_"): v for k, v in overrides.items()
          if k.replace("-", "_") in names and v is not None}
    return Params(**kw)


# ================================================================================================
# 内核：SNNLM（阶段链脉冲语言模型；与 snn_lm/snn.py 同构，移除失败的 γ[v3] 死路径）
# ================================================================================================
class SNNLM:
    def __init__(self, V, params=None, seed=SEED):
        self.V = int(V)
        self.p = params or Params()
        self.rng = np.random.default_rng(seed)
        p = self.p
        K, D = p.K, p.window_ms
        self.D = D
        self.offsets = set(p.burst_offsets)
        f32, i32, b8 = np.float32, np.int32, np.bool_

        # 状态：M 延迟链 / P 读出群 / 痕迹
        self.hist = np.zeros((K, D, self.V), dtype=b8)
        self.in_spk = np.zeros(self.V, dtype=b8)
        self.M_V = np.full((K, self.V), p.V_rest, dtype=f32)
        self.M_I = np.zeros((K, self.V), dtype=f32)
        self.M_ref = np.zeros((K, self.V), dtype=i32)
        self.M_a = np.zeros((K, self.V), dtype=f32)
        self.P_V = np.full(self.V, p.V_rest, dtype=f32)
        self.P_I = np.zeros(self.V, dtype=f32)
        self.P_r_all = np.zeros((K, self.V), dtype=f32)
        self.P_d_all = np.zeros((K, self.V), dtype=f32)
        self.P_ref = np.zeros(self.V, dtype=i32)
        self.P_a = np.zeros(self.V, dtype=f32)
        self.G = np.zeros(self.V, dtype=f32)
        self.Gs = np.zeros(self.V, dtype=f32)
        self.gamma2 = np.zeros((self.V, self.V), dtype=f32)
        self.gamma_c = np.zeros((self.V, K), dtype=f32)
        self.Mc_read = np.zeros((K, self.V), dtype=f32)
        self.W = np.zeros((K, self.V, self.V), dtype=f32)
        self.pre = np.zeros((K, self.V), dtype=f32)
        self.post = np.zeros(self.V, dtype=f32)
        self.atp = np.ones((K, self.V), dtype=f32)
        self.spk_acc = np.zeros((K, self.V), dtype=f32)
        # 滑动阈值（BCM 式，见 CHANGELOG 1.2.0；reset 不清、随快照保存）
        self.post_bar = np.zeros(self.V, dtype=f32)
        self.mc_bar = np.zeros(p.K, dtype=f32)
        # 慢库 W_s（无衰减）与逐行使用度 EMA u（mem 分支，见 CHANGELOG mem 0.1.0）
        self.W_s = np.zeros((K, self.V, self.V), dtype=f32)
        self.u = np.zeros((K, self.V), dtype=f32)

        # 时间常数（一次性预计算）
        self.dec_ca = float(np.exp(-1.0 / p.tau_ca))
        self.dec_ad = float(np.exp(-1.0 / p.tau_adapt))
        self.dec_g = float(np.exp(-1.0 / p.tau_gate))
        self.dec_gs = float(np.exp(-1.0 / p.tau_gate_read))
        self.dec_ampa = float(np.exp(-1.0 / p.tau_ampa))
        self.dec_r = float(np.exp(-1.0 / p.tau_nmda_r))
        self.dec_d = float(np.exp(-1.0 / p.tau_nmda_d))
        self.dec_tr = float(np.exp(-1.0 / p.tau_trace))
        self.dec_w = float(np.exp(-1.0 / p.tau_decay_w))
        self.dec_w_amort = self.dec_w ** p.decay_amort_steps
        self.dec_g2_amort = float(np.exp(-p.decay_amort_steps * p.learn_every_ms / p.tau_gate2))
        self.dec_cr = float(np.exp(-1.0 / p.tau_copy_read))
        self.dec_c_amort = float(np.exp(-p.decay_amort_steps * p.learn_every_ms / p.tau_copy))
        self._lt_counter = 0
        self._gamma2_any = False
        self._gammac_any = False
        self.alpha_m = float(1.0 / p.tau_m)

    def reset(self):
        p = self.p
        self.hist[:] = False
        self.M_V[:] = p.V_rest
        self.M_I[:] = 0.0
        self.M_ref[:] = 0
        self.P_V[:] = p.V_rest
        self.P_I[:] = 0.0
        self.P_r_all[:] = 0.0
        self.P_d_all[:] = 0.0
        self.P_ref[:] = 0
        self.G[:] = 0.0
        self.Gs[:] = 0.0
        self.Mc_read[:] = 0.0
        self.post[:] = 0.0
        self.atp[:] = 1.0
        self.spk_acc[:] = 0.0
        self.M_a[:] = 0.0
        self.P_a[:] = 0.0
        self._lt_counter = 0

    def _lif_step(self, V, I, ref, adapt=None, g_adapt=0.0):
        p = self.p
        g = 1.0 + p.shunt_gain / (1.0 + np.exp(-(V - p.shunt_c) / p.shunt_s))
        drive = I
        if adapt is not None and g_adapt:
            drive = I - g_adapt * adapt
        V += self.alpha_m * (-(V - p.V_rest) * g + drive)
        ref -= 1
        ref[ref < 0] = 0
        spk = (V > p.V_th) & (ref == 0)
        V[spk] = p.V_reset
        ref[spk] = p.t_ref
        hold = ref > 0
        V[hold] = p.V_reset
        return spk

    def _learn(self, dt_l):
        """每 LEARN_EVERY_MS 触发一次：局部 Hebbian + 资源可用度 + ATP + 先验门控。"""
        p = self.p
        post = self.post
        ltp_mass = 0.0
        het_mass = 0.0
        if post.max() > POST_ACTIVE_THRESHOLD:
            for i in range(p.K):
                act = np.flatnonzero(self.pre[i] > PRE_ACTIVE_THRESHOLD)
                if act.size == 0:
                    continue
                pre_a = np.minimum(self.pre[i][act], 1.0)
                wsum = (self.W[i][act] + self.W_s[i][act]).sum(1)
                avail = np.maximum(1.0 - wsum / p.budget_B, 0.0)
                coef = pre_a * avail * self.atp[i][act]
                dW = (p.eta_ltp * dt_l) * np.outer(coef, post)
                ua = self.u[i][act]
                rho = ua / (ua + float(ua.mean()) + 1e-12)
                self.W[i][act] += dW * (1.0 - rho)[:, None]
                self.W_s[i][act] += dW * rho[:, None]
                ltp_mass += float(dW.sum())
                if p.eta_hetero > 0:
                    het = (p.eta_hetero * dt_l) * np.outer(1.0 - pre_a, post)
                    bloc = self.W[i][act]
                    het_mass += float((bloc * het).sum())
                    self.W[i][act] = bloc * (1.0 - het)
                self.atp[i][act] -= p.c_spk * self.spk_acc[i][act] + p.c_plast * dW.sum(1)
        if p.eta_gate2 > 0 and float(self.post.sum()) > POST_SUM_THRESHOLD:
            act = np.flatnonzero(self.G > GATE_ACTIVE_THRESHOLD)
            if act.size:
                near = p.gate_near_stages
                d_deep = self.P_d_all[near:].sum(0)
                u_near = self.P_d_all[:near].sum(0)
                idx = int(np.argmax(self.post))
                u_mean = float(u_near.mean())
                q = float(u_near[idx]) / (u_mean + DIV_EPS)
                wgt = 1.0 / (1.0 + max(q, 0.0))
                co = d_deep * (self.post - self.post_bar)
                self.gamma2[act] += (p.eta_gate2 * dt_l) * wgt * self.G[act, None] * co[None, :]
                self._gamma2_any = True
            self.post_bar += (self.post - self.post_bar) * (dt_l / p.tau_gate2)
        if p.eta_copy > 0 and float(self.post.sum()) > POST_SUM_THRESHOLD:
            act = np.flatnonzero(self.G > GATE_ACTIVE_THRESHOLD)
            if act.size:
                idx = int(np.argmax(self.post))
                u_near = self.P_d_all[:p.gate_near_stages].sum(0)
                u_mean = float(u_near.mean())
                q = float(u_near[idx]) / (u_mean + DIV_EPS)
                wgt = 1.0 / (1.0 + max(q, 0.0))
                dot = self.Mc_read[:, idx] - self.mc_bar
                self.gamma_c[act] += (p.eta_copy * dt_l) * wgt * self.G[act, None] * dot[None, :]
                self._gammac_any = True
                self.mc_bar += (self.Mc_read[:, idx] - self.mc_bar) * (dt_l / p.tau_copy)
        self.spk_acc[:] = 0.0
        self.u += (self.pre - self.u) * (dt_l / p.tau_decay_w)
        self.atp += (1.0 - self.atp) * (dt_l / p.tau_atp)
        self._lt_counter += 1
        if self._lt_counter % p.decay_amort_steps == 1:
            self.W *= self.dec_w_amort
            if p.eta_gate2 > 0:
                self.gamma2 *= self.dec_g2_amort
            if p.eta_copy > 0:
                self.gamma_c *= self.dec_c_amort
        return ltp_mass, het_mass

    def forward_stream(self, ids, learn=False, collect=False):
        """流式前向：ids[i] 的输出 logits[i] 预测 ids[i]（上下文 w_{i-1..i-K} 驱动）。"""
        p = self.p
        K, D, V = p.K, self.D, self.V
        ids = np.asarray(ids, dtype=np.int64)
        N = len(ids)
        T = N * D
        logits = np.zeros((N, V), dtype=np.float32) if collect else None
        logits_v = np.zeros((N, V), dtype=np.float32) if collect else None
        mspk_win = np.zeros((N, K, V), dtype=bool) if collect else None
        pspk_win = np.zeros(N, dtype=np.int32) if collect else None
        win_spk = np.zeros((K, V), dtype=bool)
        p_count = 0
        ltp_mass = 0.0
        het_mass = 0.0
        offsets = self.offsets
        learn_every = p.learn_every_ms
        read_at = D - p.read_margin
        g_read = p.g_nmda

        for t in range(T):
            tok = ids[t // D]
            off = t - (t // D) * D
            head = off
            self.in_spk[:] = False
            if off in offsets:
                self.in_spk[tok] = True

            cin = self.hist[:, head]
            self.M_I += p.w_chain * cin
            self.M_I *= self.dec_ampa
            self.M_a *= self.dec_ad
            mspk = self._lif_step(self.M_V, self.M_I, self.M_ref, self.M_a, p.g_adapt_m)
            self.M_a += mspk

            self.pre *= self.dec_tr
            self.pre += mspk

            self.P_I += p.w_in_p * self.in_spk
            self.P_I *= self.dec_ampa
            for i in range(K):
                idx = np.flatnonzero(mspk[i])
                if idx.size == 1:
                    self.P_r_all[i] += self.W[i][idx[0]] + self.W_s[i][idx[0]]
                elif idx.size:
                    self.P_r_all[i] += self.W[i][idx].sum(0) + self.W_s[i][idx].sum(0)
            self.G *= self.dec_g
            self.G += mspk[0]
            self.Gs *= self.dec_gs
            self.Gs += mspk[0]
            self.Mc_read *= self.dec_cr
            self.Mc_read += mspk
            self.P_d_all += self.P_r_all * (1.0 / p.tau_nmda_d)
            self.P_r_all *= self.dec_r
            self.P_d_all *= self.dec_d
            p_drive = self.P_d_all.sum(0)
            self.P_a *= self.dec_ad
            pspk = self._lif_step(self.P_V, self.P_I + p.g_nmda * p_drive, self.P_ref,
                                  self.P_a, p.g_adapt_p)
            self.P_a += pspk

            post_inst = (p.post_w_spk * pspk
                         + p.post_w_analog * np.maximum(self.P_V - p.theta_post, 0.0)
                         / (p.V_th - p.V_rest))
            self.post *= self.dec_ca
            self.post += post_inst

            if learn:
                self.spk_acc += mspk
                if t % learn_every == learn_every - 1:
                    a, b = self._learn(learn_every)
                    ltp_mass += a
                    het_mass += b

            if collect:
                win_spk |= mspk
                p_count += int(pspk.sum())
                if off == read_at:
                    near = p.gate_near_stages
                    lt = self.P_d_all.sum(0)
                    if self._gamma2_any:
                        act = np.flatnonzero(self.Gs > GATE_READ_THRESHOLD)
                        if act.size:
                            d_deep = self.P_d_all[near:].sum(0)
                            cvec = self.Gs[act] @ self.gamma2[act]
                            lt = lt + cvec * d_deep
                    if self._gammac_any:
                        act = np.flatnonzero(self.Gs > GATE_READ_THRESHOLD)
                        if act.size:
                            gcv = (self.Gs[act, None] * self.gamma_c[act]).sum(0)
                            lt = lt + (gcv[:, None] * self.Mc_read).sum(0)
                    logits[t // D] = g_read * lt
                    logits_v[t // D] = self.P_V
                    mspk_win[t // D] = win_spk
                    win_spk[:] = False
                    pspk_win[t // D] = p_count
                    p_count = 0

            self.hist[0, head] = self.in_spk
            for i in range(K - 1):
                self.hist[i + 1, head] = mspk[i]

        out = {
            "tokens": N,
            "ltp_mass": ltp_mass,
            "het_mass": het_mass,
            "w_sum": float(self.W.sum()),
            "w_nnz": float((self.W > W_NNZ_EPS).mean()),
            "atp_mean": float(self.atp.mean()),
            "gamma2_max": float(np.abs(self.gamma2).max()),
            "gamma_c_max": float(np.abs(self.gamma_c).max()),
        }
        if collect:
            out["logits"] = logits
            out["logits_v"] = logits_v
            out["mspk"] = mspk_win
            out["pspk"] = pspk_win
        return out

    # ---- 存读（权重 + 参数 sidecar；不存神经动态状态） ----
    def save(self, path, compressed=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = np.savez_compressed if compressed else np.savez
        writer(path, W=self.W, atp=self.atp, gamma2=self.gamma2, gamma_c=self.gamma_c,
               post_bar=self.post_bar, mc_bar=self.mc_bar, W_s=self.W_s, u=self.u)
        (path.parent / (path.name + ".params.json")).write_text(
            json.dumps(self.p.to_json(), indent=1), encoding="utf-8")

    @staticmethod
    def load(path, seed=SEED):
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        d = np.load(path)
        W = d["W"].astype(np.float32)
        K, V, _ = W.shape
        pj = json.loads((path.parent / (path.name + ".params.json")).read_text(encoding="utf-8"))
        pj["burst_offsets"] = tuple(pj["burst_offsets"])
        names = {f.name for f in fields(Params)}
        params = Params(**{k: v for k, v in pj.items() if k in names})
        m = SNNLM(V, params, seed=seed)
        m.W = W
        m.atp = d["atp"].astype(np.float32)
        if "gamma2" in d:
            g2 = d["gamma2"].astype(np.float32)
            if g2.shape == (V, V):
                m.gamma2 = g2
                m._gamma2_any = bool(np.abs(m.gamma2).max() > GAMMA_NNZ_EPS)
        if "gamma_c" in d:
            gc = d["gamma_c"].astype(np.float32)
            if gc.shape == (V, params.K):
                m.gamma_c = gc
                m._gammac_any = bool(np.abs(m.gamma_c).max() > GAMMA_NNZ_EPS)
        if "post_bar" in d:
            pb = d["post_bar"].astype(np.float32)
            if pb.shape == (V,):
                m.post_bar = pb
        if "mc_bar" in d:
            mb = d["mc_bar"].astype(np.float32)
            if mb.shape == (params.K,):
                m.mc_bar = mb
        if "W_s" in d:
            ws = d["W_s"].astype(np.float32)
            if ws.shape == m.W.shape:
                m.W_s = ws
        if "u" in d:
            uu = d["u"].astype(np.float32)
            if uu.shape == m.u.shape:
                m.u = uu
        return m


# ================================================================================================
# 可选加速后端：numba 热路径 v2（整段流一次 JIT；语义与 forward_stream 严格同构）
#   稀疏化：LIF 直接产出脉冲 id 列表；hist 链/M_a/pre/Mc_read/spk_acc/win_spk/hist 写入
#   全部按脉冲列表更新（实测 M 脉冲 ~5/token）；hist 布尔数组仍为真值源，
#   快表容量 HIST_FAST_CAP 溢出时读侧扫描回退 —— 严格等价，无近似。
#   惰性衰减（LAZY_DECAY，默认开）：事件计数 ev；W/γ2/γc 每行记录 row_ev，读/写行前按
#   (ev-row_ev) 补乘 dec**k（严格同语义；pow 与逐次乘在本次验证中等价）；learn 调用结束物化全表。
#   等价性验证（研究线 probe.py --part numba，big3+门控 300tok）：eager logits Δ=7.6e-6 / ΔW=3.0e-8；
#   lazy logits Δ=3.4e-5 / argmax 100% / ΔW=8.9e-7 / Δγ2=8.6e-6。
#   numba 缺失时自动回退 numpy（BACKEND='auto'）。
# ================================================================================================
try:
    from numba import njit as _njit
    HAS_NUMBA = True
except Exception:
    _njit = None
    HAS_NUMBA = False

HIST_FAST_CAP = 32          # hist 快表每 (级, head) 的脉冲 id 容量（溢出则 hist_n=-1 走扫描回退）


if HAS_NUMBA:
    @_njit(cache=True)
    def _nb_hist_sync(hist, hist_ids, hist_n, cap, K, D, V):
        """从 hist 布尔真值源重建 (ids, n) 快表（入口同步；n>cap 置 -1 走扫描回退）。"""
        for i in range(K):
            for h in range(D):
                s = 0
                for v in range(V):
                    if hist[i, h, v]:
                        if s < cap:
                            hist_ids[i, h, s] = v
                        s += 1
                if s <= cap:
                    hist_n[i, h] = s
                else:
                    hist_n[i, h] = -1


    @_njit(cache=True)
    def _nb_lif_ids(Vv, Ii, ref, adapt, g_adapt,
                 V_rest, V_th, V_reset, alpha_m, shunt_gain, shunt_c, shunt_s, t_ref,
                 ids_out):
        """(V,) 级 LIF（与 snn.py _lif_step 同序）；返回脉冲 id（写入 ids_out，返回个数）。"""
        n = Vv.shape[0]
        n_spk = 0
        for j in range(n):
            g = 1.0 + shunt_gain / (1.0 + np.exp(-(Vv[j] - shunt_c) / shunt_s))
            drive = Ii[j]
            if g_adapt != 0.0:
                drive = Ii[j] - g_adapt * adapt[j]
            Vv[j] += alpha_m * (-(Vv[j] - V_rest) * g + drive)
            r = ref[j] - 1
            if r < 0:
                r = 0
            ref[j] = r
        for j in range(n):
            if Vv[j] > V_th and ref[j] == 0:
                Vv[j] = V_reset
                ref[j] = t_ref
                ids_out[n_spk] = j
                n_spk += 1
            elif ref[j] > 0:
                Vv[j] = V_reset
        return n_spk


    @_njit(cache=True)
    def _nb_learn(counter, ev_vec, row_ev_w, g2_ev, gc_ev,
                   pre, post, spk_acc, atp, W, W_s, u, gamma2, gamma_c, P_d_all, G, Mc_read,
                   act_buf, act_buf2, d_deep, u_near, gcv,
                   eta_ltp, eta_hetero, budget_B, c_spk, c_plast, tau_atp_inv, learn_dt, u_rate,
                   post_act_th, pre_act_th, gate_act_th, post_sum_th, div_eps,
                   gate_near, use_gate2, eta_gate2, post_bar, post_bar_rate,
                   use_copy, eta_copy, mc_bar, mc_bar_rate,
                   decay_amort_steps, w_dec, g2_dec, gc_dec,
                   g2_any_in, gc_any_in, use_lazy, K, V):
        """每 learn_every_ms 一次的可塑性更新（与 snn.py _learn 同序）。

        惰性衰减：全局事件数在 ev_vec[0]（counter%amort==1 时 +1）；每行 W/γ2/γc 记录
        上次补乘的事件数 row_ev_*；读写任一行前先按 (ev - row_ev) 事件补乘 dec**k（严格等价
        于逐事件乘 dec**amort）。入口/结束由 run_stream_numba 负责同步与全量物化。
        """
        ltp = 0.0
        het = 0.0
        post_sum = 0.0
        post_max = 0.0
        for x in range(V):
            px = post[x]
            post_sum += px
            if px > post_max:
                post_max = px
        if post_max > post_act_th:
            for i in range(K):
                n = 0
                for v in range(V):
                    if pre[i, v] > pre_act_th:
                        act_buf[n] = v
                        n += 1
                if n == 0:
                    continue
                usum = 0.0
                for a in range(n):
                    usum += u[i, act_buf[a]]
                um = usum / n
                for a in range(n):
                    v = act_buf[a]
                    kk = ev_vec[0] - row_ev_w[i, v]
                    if use_lazy == 0:
                        kk = 0
                    if kk > 0:
                        ff = w_dec ** kk
                        for x in range(V):
                            W[i, v, x] *= ff
                        row_ev_w[i, v] = ev_vec[0]
                    pre_raw = pre[i, v]
                    pre_a = pre_raw if pre_raw < 1.0 else 1.0
                    wsum = 0.0
                    for x in range(V):
                        wsum += W[i, v, x] + W_s[i, v, x]
                    avail = 1.0 - wsum / budget_B
                    if avail < 0.0:
                        avail = 0.0
                    coef = pre_a * avail * atp[i, v]
                    ua = u[i, v]
                    rho = ua / (ua + um + div_eps)
                    rowsum = 0.0
                    scale = eta_ltp * learn_dt * coef
                    for x in range(V):
                        dw = scale * post[x]
                        W[i, v, x] += dw * (1.0 - rho)
                        W_s[i, v, x] += dw * rho
                        rowsum += dw
                        ltp += dw
                    if eta_hetero > 0.0:
                        hscale = eta_hetero * learn_dt * (1.0 - pre_a)
                        for x in range(V):
                            h = hscale * post[x]
                            het += W[i, v, x] * h
                            W[i, v, x] = W[i, v, x] * (1.0 - h)
                    atp[i, v] -= c_spk * spk_acc[i, v] + c_plast * rowsum
        if use_gate2 and post_sum > post_sum_th:
            n2 = 0
            for j in range(V):
                if G[j] > gate_act_th:
                    act_buf2[n2] = j
                    n2 += 1
            if n2 > 0:
                for x in range(V):
                    dd = 0.0
                    un = 0.0
                    for i in range(K):
                        if i < gate_near:
                            un += P_d_all[i, x]
                        else:
                            dd += P_d_all[i, x]
                    d_deep[x] = dd
                    u_near[x] = un
                idx = 0
                best = post[0]
                for x in range(1, V):
                    if post[x] > best:
                        best = post[x]
                        idx = x
                usum = 0.0
                for x in range(V):
                    usum += u_near[x]
                u_mean = usum / V
                q = u_near[idx] / (u_mean + div_eps)
                wgt = 1.0 / (1.0 + (q if q > 0.0 else 0.0))
                for a in range(n2):
                    j = act_buf2[a]
                    kk = ev_vec[0] - g2_ev[j]
                    if use_lazy == 0:
                        kk = 0
                    if kk > 0:
                        ff = g2_dec ** kk
                        for x in range(V):
                            gamma2[j, x] *= ff
                        g2_ev[j] = ev_vec[0]
                    gv = G[j]
                    base = eta_gate2 * learn_dt * wgt * gv
                    for x in range(V):
                        gamma2[j, x] += base * (d_deep[x] * (post[x] - post_bar[x]))
                    g2_any_in = True
            for x in range(V):
                post_bar[x] += (post[x] - post_bar[x]) * post_bar_rate
        if use_copy and post_sum > post_sum_th:
            n2 = 0
            for j in range(V):
                if G[j] > gate_act_th:
                    act_buf2[n2] = j
                    n2 += 1
            if n2 > 0:
                idx = 0
                best = post[0]
                for x in range(1, V):
                    if post[x] > best:
                        best = post[x]
                        idx = x
                u_mean = 0.0
                for x in range(V):
                    s = 0.0
                    for i in range(K):
                        if i < gate_near:
                            s += P_d_all[i, x]
                    u_near[x] = s
                    u_mean += s
                u_mean = u_mean / V
                q = u_near[idx] / (u_mean + div_eps)
                wgt = 1.0 / (1.0 + (q if q > 0.0 else 0.0))
                for k in range(K):
                    gcv[k] = Mc_read[k, idx] - mc_bar[k]
                for a in range(n2):
                    j = act_buf2[a]
                    kk = ev_vec[0] - gc_ev[j]
                    if use_lazy == 0:
                        kk = 0
                    if kk > 0:
                        ff = gc_dec ** kk
                        for k2 in range(K):
                            gamma_c[j, k2] *= ff
                        gc_ev[j] = ev_vec[0]
                    gv = G[j]
                    base = eta_copy * learn_dt * wgt * gv
                    for k in range(K):
                        gamma_c[j, k] += base * gcv[k]
                for k in range(K):
                    mc_bar[k] += (Mc_read[k, idx] - mc_bar[k]) * mc_bar_rate
                    gc_any_in = True
        for i in range(K):
            for v in range(V):
                spk_acc[i, v] = 0.0
        for i in range(K):
            for v in range(V):
                a = atp[i, v]
                atp[i, v] = a + (1.0 - a) * tau_atp_inv * learn_dt
        for i in range(K):
            for v in range(V):
                u[i, v] += (pre[i, v] - u[i, v]) * u_rate
        counter += 1
        if counter % decay_amort_steps == 1:
            if use_lazy == 1:
                ev_vec[0] += 1
            else:
                for i in range(K):
                    for v in range(V):
                        for x in range(V):
                            W[i, v, x] *= w_dec
                if use_gate2:
                    for j in range(V):
                        for x in range(V):
                            gamma2[j, x] *= g2_dec
                if use_copy:
                    for j in range(V):
                        for k in range(K):
                            gamma_c[j, k] *= gc_dec
        return ltp, het, counter, g2_any_in, gc_any_in


    @_njit(cache=True)
    def _nb_run(ids, learn, collect,
                    K, D, V, offsets_mask, read_at,
                    hist, hist_ids, hist_n,
                    ev_vec, row_ev_w, g2_ev, gc_ev,
                    M_V, M_I, M_ref, M_a, P_V, P_I, P_ref, P_a,
                    G, Gs, Mc_read, P_r_all, P_d_all, pre, post, atp, spk_acc,
                    W, W_s, u, gamma2, gamma_c,
                    ids_buf, n_spk, act_buf, act_buf2, d_deep, u_near, cv, gcv, lt, pdrive, win_spk,
                    logits, logits_v, mspk_win, pspk_win,
                    w_chain, dec_ampa, dec_ad, dec_tr, dec_g, dec_gs, dec_cr,
                    w_in_p, g_nmda, inv_tau_d, dec_r, dec_d, dec_ca,
                    post_w_spk, post_w_analog, theta_post, vrange,
                    V_rest, V_th, V_reset, alpha_m, shunt_gain, shunt_c, shunt_s, t_ref,
                    g_adapt_m, g_adapt_p,
                    eta_ltp, eta_hetero, budget_B, c_spk, c_plast, tau_atp_inv, learn_every_ms, u_rate,
                    use_gate2, eta_gate2, post_bar, post_bar_rate, g2_dec,
                    use_copy, eta_copy, mc_bar, mc_bar_rate, gc_dec,
                    gate_near, gate_act_th, gate_read_th, post_act_th, pre_act_th,
                    post_sum_th, div_eps, decay_amort_steps, w_dec, hist_cap, use_lazy,
                    counter_in, g2_any_in, gc_any_in):
        N = ids.shape[0]
        T = N * D
        ltp_mass = 0.0
        het_mass = 0.0
        counter = counter_in
        g2_any = g2_any_in
        gc_any = gc_any_in
        p_count = 0
        g_read = g_nmda
        for t in range(T):
            n_tok = t // D
            off = t - n_tok * D
            head = off
            tok = ids[n_tok]
            set_spk = offsets_mask[off]

            # ---- 链输入（hist 快表驱动；溢出单元扫描回退） ----
            for i in range(K):
                hn = hist_n[i, head]
                if hn > 0:
                    for a in range(hn):
                        M_I[i, hist_ids[i, head, a]] += w_chain
                elif hn < 0:
                    for v in range(V):
                        if hist[i, head, v]:
                            M_I[i, v] += w_chain
            for i in range(K):
                for v in range(V):
                    M_I[i, v] *= dec_ampa
                    M_a[i, v] *= dec_ad
            # ---- M 级 LIF（脉冲 id 列表） ----
            for i in range(K):
                n_spk[i] = _nb_lif_ids(M_V[i], M_I[i], M_ref[i], M_a[i], g_adapt_m,
                                    V_rest, V_th, V_reset, alpha_m, shunt_gain, shunt_c, shunt_s,
                                    t_ref, ids_buf[i])
            for i in range(K):
                for v in range(V):
                    pre[i, v] *= dec_tr
                for a in range(n_spk[i]):
                    v = ids_buf[i, a]
                    M_a[i, v] += 1.0
                    pre[i, v] += 1.0

            # ---- P 级（快输入 + NMDA 慢驱动） ----
            if set_spk:
                P_I[tok] += w_in_p
            for v in range(V):
                P_I[v] *= dec_ampa
            for i in range(K):
                for a in range(n_spk[i]):
                    v = ids_buf[i, a]
                    kk = ev_vec[0] - row_ev_w[i, v] if use_lazy == 1 else 0
                    if kk > 0:
                        ff = w_dec ** kk
                        for x in range(V):
                            W[i, v, x] *= ff
                        row_ev_w[i, v] = ev_vec[0]
                    for x in range(V):
                        P_r_all[i, x] += W[i, v, x] + W_s[i, v, x]
            for v in range(V):
                G[v] *= dec_g
                Gs[v] *= dec_gs
            for a in range(n_spk[0]):
                v = ids_buf[0, a]
                G[v] += 1.0
                Gs[v] += 1.0
            for i in range(K):
                for v in range(V):
                    Mc_read[i, v] *= dec_cr
            for i in range(K):
                for a in range(n_spk[i]):
                    Mc_read[i, ids_buf[i, a]] += 1.0
            for i in range(K):
                for x in range(V):
                    P_d_all[i, x] += P_r_all[i, x] * inv_tau_d
                    P_r_all[i, x] *= dec_r
                    P_d_all[i, x] *= dec_d
            for x in range(V):
                s = 0.0
                for i in range(K):
                    s += P_d_all[i, x]
                lt[x] = s
            for v in range(V):
                P_a[v] *= dec_ad
                pdrive[v] = P_I[v] + g_nmda * lt[v]
            pspk_n = _nb_lif_ids(P_V, pdrive, P_ref, P_a, g_adapt_p,
                              V_rest, V_th, V_reset, alpha_m, shunt_gain, shunt_c, shunt_s,
                              t_ref, ids_buf[K])
            for a in range(pspk_n):
                P_a[ids_buf[K, a]] += 1.0
            for x in range(V):
                post[x] *= dec_ca
            for a in range(pspk_n):
                post[ids_buf[K, a]] += post_w_spk
            if post_w_analog != 0.0:
                for x in range(V):
                    d = P_V[x] - theta_post
                    if d > 0.0:
                        post[x] += post_w_analog * d / vrange

            # ---- 可塑性（每 learn_every_ms） ----
            if learn:
                for i in range(K):
                    for a in range(n_spk[i]):
                        spk_acc[i, ids_buf[i, a]] += 1.0
                if t % learn_every_ms == learn_every_ms - 1:
                    a_, b_, counter, g2_any, gc_any = _nb_learn(
                            counter, ev_vec, row_ev_w, g2_ev, gc_ev,
                            pre, post, spk_acc, atp, W, W_s, u, gamma2, gamma_c, P_d_all, G, Mc_read,
                        act_buf, act_buf2, d_deep, u_near, gcv,
                        eta_ltp, eta_hetero, budget_B, c_spk, c_plast, tau_atp_inv,
                        float(learn_every_ms), u_rate,
                        post_act_th, pre_act_th, gate_act_th, post_sum_th, div_eps,
                        gate_near, use_gate2, eta_gate2, post_bar, post_bar_rate,
                        use_copy, eta_copy, mc_bar, mc_bar_rate,
                            decay_amort_steps, w_dec, g2_dec, gc_dec,
                            g2_any, gc_any, use_lazy, K, V)
                    ltp_mass += a_
                    het_mass += b_

            # ---- 读出（窗末） ----
            if collect:
                for i in range(K):
                    for a in range(n_spk[i]):
                        win_spk[i, ids_buf[i, a]] = True
                p_count += pspk_n
                if off == read_at:
                    if g2_any:
                        n2 = 0
                        for j in range(V):
                            if Gs[j] > gate_read_th:
                                act_buf2[n2] = j
                                n2 += 1
                        if n2 > 0:
                            for x in range(V):
                                dd = 0.0
                                for i in range(gate_near, K):
                                    dd += P_d_all[i, x]
                                d_deep[x] = dd
                            for x in range(V):
                                cv[x] = 0.0
                            for a in range(n2):
                                j = act_buf2[a]
                                kk = ev_vec[0] - g2_ev[j] if use_lazy == 1 else 0
                                if kk > 0:
                                    ff = g2_dec ** kk
                                    for x in range(V):
                                        gamma2[j, x] *= ff
                                    g2_ev[j] = ev_vec[0]
                                gsj = Gs[j]
                                for x in range(V):
                                    cv[x] += gsj * gamma2[j, x]
                            for x in range(V):
                                lt[x] += cv[x] * d_deep[x]
                    if gc_any:
                        n2 = 0
                        for j in range(V):
                            if Gs[j] > gate_read_th:
                                act_buf2[n2] = j
                                n2 += 1
                        if n2 > 0:
                            for k in range(K):
                                gcv[k] = 0.0
                            for a in range(n2):
                                j = act_buf2[a]
                                kk = ev_vec[0] - gc_ev[j] if use_lazy == 1 else 0
                                if kk > 0:
                                    ff = gc_dec ** kk
                                    for k2 in range(K):
                                        gamma_c[j, k2] *= ff
                                    gc_ev[j] = ev_vec[0]
                                gsj = Gs[j]
                                for k in range(K):
                                    gcv[k] += gsj * gamma_c[j, k]
                            for x in range(V):
                                s = 0.0
                                for k in range(K):
                                    s += gcv[k] * Mc_read[k, x]
                                lt[x] += s
                    for x in range(V):
                        logits[n_tok, x] = g_read * lt[x]
                        logits_v[n_tok, x] = P_V[x]
                    for i in range(K):
                        for v in range(V):
                            mspk_win[n_tok, i, v] = win_spk[i, v]
                            win_spk[i, v] = False
                    pspk_win[n_tok] = p_count
                    p_count = 0

            # ---- hist 写（快表 + 布尔真值源；旧值按旧 id 清除/回退扫描清除） ----
            # 级 0：in_spk（本步）
            hn0 = hist_n[0, head]
            if hn0 > 0:
                for a in range(hn0):
                    hist[0, head, hist_ids[0, head, a]] = False
            elif hn0 < 0:
                for v in range(V):
                    hist[0, head, v] = False
            if set_spk:
                hist[0, head, tok] = True
                hist_ids[0, head, 0] = tok
                hist_n[0, head] = 1
            else:
                hist_n[0, head] = 0
            # 级 i+1：本步 mspk[i] 的 id 列表
            for i in range(K - 1):
                hn_old = hist_n[i + 1, head]
                if hn_old > 0:
                    for a in range(hn_old):
                        hist[i + 1, head, hist_ids[i + 1, head, a]] = False
                elif hn_old < 0:
                    for v in range(V):
                        hist[i + 1, head, v] = False
                n_i = n_spk[i]
                for a in range(n_i):
                    v = ids_buf[i, a]
                    hist[i + 1, head, v] = True
                if n_i <= hist_cap:
                    for a in range(n_i):
                        hist_ids[i + 1, head, a] = ids_buf[i, a]
                    hist_n[i + 1, head] = n_i
                else:
                    hist_n[i + 1, head] = -1

        # ---- 结束物化（惰性模式 learn 时补齐全部行；eager 模式无需） ----
        if use_lazy == 1 and learn:
            ev = ev_vec[0]
            for i in range(K):
                for v in range(V):
                    kk = ev - row_ev_w[i, v]
                    if kk > 0:
                        ff = w_dec ** kk
                        for x in range(V):
                            W[i, v, x] *= ff
                        row_ev_w[i, v] = ev
            if use_gate2:
                for j in range(V):
                    kk = ev - g2_ev[j]
                    if kk > 0:
                        ff = g2_dec ** kk
                        for x in range(V):
                            gamma2[j, x] *= ff
                        g2_ev[j] = ev
            if use_copy:
                for j in range(V):
                    kk = ev - gc_ev[j]
                    if kk > 0:
                        ff = gc_dec ** kk
                        for k2 in range(K):
                            gamma_c[j, k2] *= ff
                        gc_ev[j] = ev

        return ltp_mass, het_mass, counter, g2_any, gc_any


    def run_stream_numba(model, ids, learn=False, collect=False, lazy=None):
        """与 SNNLM.forward_stream 同接口的 numba 后端（更新 model 状态并返回同构 out）。

        lazy: None → 用模块常量 LAZY_DECAY（默认 False）；True/False 显式指定。
        惰性模式语义与 eager 相同，但引入 ~1ulp 舍入种子，长程轨迹因相位敏感性会漂移。
        """
        lz = LAZY_DECAY if lazy is None else bool(lazy)
        p = model.p
        K, D, V = p.K, model.D, model.V
        ids = np.asarray(ids, dtype=np.int64)
        N = len(ids)

        offsets_mask = np.zeros(D, dtype=np.bool_)
        for o in p.burst_offsets:
            offsets_mask[o] = True
        read_at = D - p.read_margin

        if not hasattr(model, "_nb_buffers"):
            model._nb_buffers = {
                "hist_ids": np.zeros((K, D, HIST_FAST_CAP), dtype=np.int32),
                "hist_n": np.zeros((K, D), dtype=np.int8),
                "ids_buf": np.zeros((K + 1, V), dtype=np.int64),
                "n_spk": np.zeros(K, dtype=np.int64),
                "act_buf": np.zeros(V, dtype=np.int64),
                "act_buf2": np.zeros(V, dtype=np.int64),
                "d_deep": np.zeros(V, dtype=np.float32),
                "u_near": np.zeros(V, dtype=np.float32),
                "cv": np.zeros(V, dtype=np.float32),
                "gcv": np.zeros(K, dtype=np.float32),
                "lt": np.zeros(V, dtype=np.float32),
                "pdrive": np.zeros(V, dtype=np.float32),
                "win_spk": np.zeros((K, V), dtype=np.bool_),
                "ev": np.zeros(1, dtype=np.int64),
                "row_ev_w": np.zeros((K, V), dtype=np.int32),
                "g2_ev": np.zeros(V, dtype=np.int32),
                "gc_ev": np.zeros(V, dtype=np.int32),
                "last_counter": 0,
            }
        b = model._nb_buffers
        _nb_hist_sync(model.hist, b["hist_ids"], b["hist_n"], HIST_FAST_CAP, K, D, V)
        b["win_spk"][:] = False

        # 惰性衰减入口同步：外部（reset/ numpy 训练）改变过 _lt_counter 时重建事件基线
        if b["last_counter"] != model._lt_counter:
            if model._lt_counter > b["last_counter"]:
                # numpy 路径为 eager 衰减 → 全部行视为“已按事件补齐”
                nc = model._lt_counter
                e = 0 if nc <= 0 else (nc - 1) // p.decay_amort_steps + 1
                b["ev"][0] = e
                b["row_ev_w"][:] = e
                b["g2_ev"][:] = e
                b["gc_ev"][:] = e
            else:
                # reset（计数归零且 W 未动）→ 事件基线归零
                b["ev"][0] = 0
                b["row_ev_w"][:] = 0
                b["g2_ev"][:] = 0
                b["gc_ev"][:] = 0

        if collect:
            logits = np.zeros((N, V), dtype=np.float32)
            logits_v = np.zeros((N, V), dtype=np.float32)
            mspk_win = np.zeros((N, K, V), dtype=np.bool_)
            pspk_win = np.zeros(N, dtype=np.int32)
        else:
            logits = np.zeros((0, V), dtype=np.float32)
            logits_v = np.zeros((0, V), dtype=np.float32)
            mspk_win = np.zeros((0, K, V), dtype=np.bool_)
            pspk_win = np.zeros(0, dtype=np.int32)

        ltp, het, counter, g2_any, gc_any = _nb_run(
            ids, learn, collect,
            K, D, V, offsets_mask, read_at,
            model.hist, b["hist_ids"], b["hist_n"],
            b["ev"], b["row_ev_w"], b["g2_ev"], b["gc_ev"],
            model.M_V, model.M_I, model.M_ref, model.M_a,
            model.P_V, model.P_I, model.P_ref, model.P_a,
            model.G, model.Gs, model.Mc_read, model.P_r_all, model.P_d_all,
            model.pre, model.post, model.atp, model.spk_acc,
            model.W, model.W_s, model.u, model.gamma2, model.gamma_c,
            b["ids_buf"], b["n_spk"], b["act_buf"], b["act_buf2"],
            b["d_deep"], b["u_near"], b["cv"], b["gcv"], b["lt"], b["pdrive"], b["win_spk"],
            logits, logits_v, mspk_win, pspk_win,
            np.float32(p.w_chain), np.float32(model.dec_ampa), np.float32(model.dec_ad),
            np.float32(model.dec_tr), np.float32(model.dec_g), np.float32(model.dec_gs),
            np.float32(model.dec_cr), np.float32(p.w_in_p), np.float32(p.g_nmda),
            np.float32(1.0 / p.tau_nmda_d), np.float32(model.dec_r), np.float32(model.dec_d),
            np.float32(model.dec_ca),
            np.float32(p.post_w_spk), np.float32(p.post_w_analog), np.float32(p.theta_post),
            np.float32(p.V_th - p.V_rest),
            np.float32(p.V_rest), np.float32(p.V_th), np.float32(p.V_reset),
            np.float32(model.alpha_m), np.float32(p.shunt_gain), np.float32(p.shunt_c),
            np.float32(p.shunt_s), np.int64(p.t_ref),
            np.float32(p.g_adapt_m), np.float32(p.g_adapt_p),
            np.float32(p.eta_ltp), np.float32(p.eta_hetero), np.float32(p.budget_B),
            np.float32(p.c_spk), np.float32(p.c_plast), np.float32(1.0 / p.tau_atp),
            np.int64(p.learn_every_ms),
            np.float32(p.learn_every_ms / p.tau_decay_w),
            p.eta_gate2 > 0, np.float32(p.eta_gate2), model.post_bar,
            np.float32(p.learn_every_ms / p.tau_gate2),
            np.float32(model.dec_g2_amort),
            p.eta_copy > 0, np.float32(p.eta_copy), model.mc_bar,
            np.float32(p.learn_every_ms / p.tau_copy),
            np.float32(model.dec_c_amort),
            np.int64(p.gate_near_stages),
            np.float32(GATE_ACTIVE_THRESHOLD),
            np.float32(GATE_READ_THRESHOLD),
            np.float32(POST_ACTIVE_THRESHOLD),
            np.float32(PRE_ACTIVE_THRESHOLD),
            np.float32(POST_SUM_THRESHOLD),
            np.float32(DIV_EPS),
            np.int64(p.decay_amort_steps), np.float32(model.dec_w_amort),
            np.int64(HIST_FAST_CAP), np.int64(1 if lz else 0),
            model._lt_counter, model._gamma2_any, model._gammac_any)

        model._lt_counter = int(counter)
        model._gamma2_any = bool(g2_any)
        model._gammac_any = bool(gc_any)
        b["last_counter"] = model._lt_counter

        out = {
            "tokens": N,
            "ltp_mass": ltp,
            "het_mass": het,
            "w_sum": float(model.W.sum()),
            "w_nnz": float((model.W > W_NNZ_EPS).mean()),
            "atp_mean": float(model.atp.mean()),
            "gamma2_abs_mean": float(np.abs(model.gamma2).mean()),
            "gamma2_max": float(np.abs(model.gamma2).max()),
            "gamma_c_max": float(np.abs(model.gamma_c).max()),
        }
        if collect:
            out["logits"] = logits
            out["logits_v"] = logits_v
            out["mspk"] = mspk_win
            out["pspk"] = pspk_win
        return out


def _fwd(model, ids, learn, collect, backend=None):
    """统一前向入口：按后端分发（auto 优先 numba，异常回退 numpy）。"""
    be = backend or BACKEND
    if be in ("auto", "numba") and HAS_NUMBA:
        if be == "numba":
            return run_stream_numba(model, ids, learn=learn, collect=collect)
        try:
            return run_stream_numba(model, ids, learn=learn, collect=collect)
        except Exception:
            pass
    return model.forward_stream(ids, learn=learn, collect=collect)


# ================================================================================================
# 评估：softmax / 掩码 PPL·ACC / n-gram 基线 / 温度自校准
# ================================================================================================
def softmax(x, T=1.0):
    x = x / T
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def evaluate(ids, logits, skip=MODEL_K, T=T_READOUT, mask=None):
    """掩码评估：仅对 mask==1 且位置 >= skip 的 token 计分；logits[i] 预测 ids[i]。"""
    ids = np.asarray(ids)
    n = min(len(ids), len(logits))
    tgt = ids[:n]
    lg = logits[:n]
    sel = np.arange(n) >= skip
    if mask is not None:
        sel = sel & (np.asarray(mask)[:n] == 1)
    lg = lg[sel]
    tgt = tgt[sel]
    probs = softmax(lg, T=T)
    p_t = probs[np.arange(len(tgt)), tgt]
    ppl = float(np.exp(-np.log(np.maximum(p_t, LOG_PROB_FLOOR)).mean()))
    acc = float((probs.argmax(1) == tgt).mean())
    return {"ppl": ppl, "acc": acc, "n": int(sel.sum())}


def ngram_reference(train, val, max_n=BASELINE_MAX_N, skip=MODEL_K,
                    lam=DEF_NGRAM_LAMBDA, alpha=DEF_NGRAM_ALPHA, val_mask=None):
    """插值 n-gram 参照（掩码口径与内核评估一致；与 snn_lm/probe.py 同协议）。"""
    V = int(max(train.max(), val.max())) + 1
    tables = [None] * (max_n + 1)
    for n in range(1, max_n + 1):
        tbl = defaultdict(lambda: defaultdict(int))
        for i in range(len(train) - n + 1):
            ctx = tuple(train[i:i + n - 1])
            tbl[ctx][int(train[i + n - 1])] += 1
        tables[n] = tbl
    uni = np.bincount(train, minlength=V).astype(np.float64)
    p1 = (uni + alpha) / (uni.sum() + alpha * V)

    logp = {n: [] for n in range(1, max_n + 1)}
    hits = {n: [] for n in range(1, max_n + 1)}
    for j in range(skip, len(val)):
        if val_mask is not None and val_mask[j] != 1:
            continue
        w = int(val[j])
        p = p1
        logp[1].append(np.log(max(p[w], LOG_PROB_FLOOR)))
        hits[1].append(int(p.argmax() == w))
        for n in range(2, max_n + 1):
            ctx = tuple(int(x) for x in val[j - n + 1:j])
            row = tables[n].get(ctx)
            c_ctx = sum(row.values()) if row else 0
            if row and c_ctx > 0:
                ml = np.zeros(V)
                for k, c in row.items():
                    ml[k] = c / c_ctx
                p = lam * ml + (1 - lam) * p
            logp[n].append(np.log(max(p[w], LOG_PROB_FLOOR)))
            hits[n].append(int(p.argmax() == w))
    out = {}
    for n in range(1, max_n + 1):
        out[f"{n}gram"] = {
            "ppl": float(np.exp(-np.mean(logp[n]))),
            "acc": float(np.mean(hits[n])),
            "n": int(len(logp[n])),
        }
    return out


def fit_temperature(model, ids, skip=MODEL_K, grid=TEMPERATURE_GRID, mask=None):
    """仅在训练集切片上拟合温度（不触碰验证集）。"""
    model.reset()
    out = _fwd(model, ids, False, True)
    best = (None, float("inf"))
    for T in grid:
        e = evaluate(ids, out["logits"], skip=skip, T=T, mask=mask)
        ce = float(np.log(e["ppl"]))
        if ce < best[1]:
            best = (T, ce)
    return float(best[0])


# ================================================================================================
# 训练：分块 + 快照 + 指标（防超时；与 snn_lm/train.py 同协议）
# ================================================================================================
def train_run(data_dir, tag, epochs=EPOCHS, max_tokens=TRAIN_CHUNK_TOKENS, offset=0,
              resume=None, ckpt_every=CKPT_EVERY_TOKENS, eval_limit=EVAL_TOKENS_LIMIT,
              params_override=None):
    tr, trm, va, vam, vocab, meta = _load_stage_data(data_dir)
    V = vocab.vocab_size
    if offset:
        tr = tr[offset:]
        if trm is not None:
            trm = trm[offset:]
    if max_tokens:
        tr = tr[:max_tokens]
        if trm is not None:
            trm = trm[:max_tokens]
    va_eval = va[:eval_limit] if eval_limit else va
    vam_eval = vam[:eval_limit] if (vam is not None and eval_limit) else vam

    p = build_params(**(params_override or {}))
    model = SNNLM(V, p, seed=SEED)
    if resume:
        src = SNNLM.load(resume, seed=SEED)
        assert src.V == V, f"resume 模型 V={src.V} 与数据 V={V} 不一致"
        model.W = src.W.copy()
        model.atp = src.atp.copy()
        model.gamma2 = src.gamma2.copy()
        model.gamma_c = src.gamma_c.copy()
        model._gamma2_any = src._gamma2_any
        model._gammac_any = src._gammac_any

    run = Path(RUNS_DIR) / tag
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(
        {"data": str(data_dir), "tag": tag, "epochs": epochs, "max_tokens": max_tokens,
         "offset": offset, "resume": str(resume), "ckpt_every": ckpt_every,
         "params": p.to_json(), "version": VERSION},
        indent=1, ensure_ascii=False), encoding="utf-8")
    logf = (run / "metrics.jsonl").open("a", encoding="utf-8")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.reset()
        agg = {"ltp_mass": 0.0, "het_mass": 0.0, "tokens": 0}
        pos = 0
        while pos < len(tr):
            seg = tr[pos: pos + ckpt_every]
            out = _fwd(model, seg, True, False)
            agg["ltp_mass"] += out["ltp_mass"]
            agg["het_mass"] += out["het_mass"]
            agg["tokens"] += out["tokens"]
            pos += len(seg)
            model.save(run / "snapshot.npz", compressed=False)
            (run / "progress.json").write_text(json.dumps(
                {"epoch": ep, "tokens_done": agg["tokens"], "of": len(tr),
                 "offset": offset}, ensure_ascii=False), encoding="utf-8")
            log_and_print(f"[训练] {tag} ep{ep} {agg['tokens']}/{len(tr)} tokens "
                          f"({time.time() - t0:.0f}s)")
        dt = time.time() - t0
        out = agg
        out["w_sum"] = float(model.W.sum())
        out["w_nnz"] = float((model.W > W_NNZ_EPS).mean())
        out["atp_mean"] = float(model.atp.mean())
        out["gamma2_max"] = float(np.abs(model.gamma2).max())
        out["gamma_c_max"] = float(np.abs(model.gamma_c).max())

        model.reset()
        ev_out = _fwd(model, va_eval, False, True)
        ev = evaluate(va_eval, ev_out["logits"], mask=vam_eval)
        rel = (out["ltp_mass"] + out["het_mass"]) / max(out["w_sum"], W_REL_FLOOR) / out["tokens"]
        rec = {
            "epoch": ep, "offset": offset, "tokens": out["tokens"],
            "train_sec": round(dt, 1),
            "ltp_mass": round(out["ltp_mass"], 4), "het_mass": round(out["het_mass"], 4),
            "w_sum": round(out["w_sum"], 4), "w_nnz": round(out["w_nnz"], 5),
            "atp_mean": round(out["atp_mean"], 4),
            "gamma2_max": round(out["gamma2_max"], 4), "gamma_c_max": round(out["gamma_c_max"], 4),
            "w_rel_change_per_token": round(rel, 8),
            "val_ppl": round(ev["ppl"], 2), "val_acc": round(ev["acc"], 4), "val_n": ev["n"],
            "backend": BACKEND, "version": VERSION,
        }
        logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logf.flush()
        log_and_print(json.dumps(rec, ensure_ascii=False))
    model.save(run / "snapshot.npz", compressed=True)
    logf.close()
    return model, run, vocab


# ================================================================================================
# 生成（对话/续写）：状态保持流式采样；无目标注入——生成 token 仅作为输入脉冲
# ================================================================================================
def generate(model, vocab, prompt_msgs, max_new=GEN_MAX_NEW_TOKENS,
             T=GEN_TEMPERATURE, top_p=GEN_TOP_P):
    text, _ = render_messages(prompt_msgs, add_generation_prompt=True)
    ids = vocab.encode(text)
    model.reset()
    out = _fwd(model, ids, False, True)
    lg = out["logits"][-1] / max(T, GEN_TEMPERATURE_MIN)
    eos = vocab.stoi.get("<|eos|>")
    out_ids = []
    for _ in range(max_new):
        e = np.exp(lg - lg.max())
        pr = e / e.sum()
        if top_p < 1.0:
            order = np.argsort(-pr)
            csum = np.cumsum(pr[order])
            keep = order[: int(np.searchsorted(csum, top_p) + 1)]
            mask = np.zeros_like(pr, dtype=bool)
            mask[keep] = True
            pr = pr * mask
            pr = pr / pr.sum()
        nxt = int(model.rng.choice(len(pr), p=pr))
        if eos is not None and nxt == eos:
            break
        out_ids.append(nxt)
        out = _fwd(model, np.array([nxt]), False, True)
        lg = out["logits"][-1] / max(T, GEN_TEMPERATURE_MIN)
    return vocab.decode(out_ids)


# ================================================================================================
# 评估套件：raw@T_READOUT + 温度标定 + n-gram 基线（全部掩码口径）
# ================================================================================================
def eval_suite(data_dir, model, vocab, tag_note=""):
    tr, trm, va, vam, _vocab, meta = _load_stage_data(data_dir)
    va_eval = va[:EVAL_TOKENS_LIMIT] if EVAL_TOKENS_LIMIT else va
    vam_eval = vam[:EVAL_TOKENS_LIMIT] if (vam is not None and EVAL_TOKENS_LIMIT) else vam
    cal_ids = tr[:CAL_TOKENS]
    cal_mask = trm[:CAL_TOKENS] if trm is not None else None

    T_cal = fit_temperature(model, cal_ids, mask=cal_mask)
    model.reset()
    out = _fwd(model, va_eval, False, True)
    raw = evaluate(va_eval, out["logits"], T=T_READOUT, mask=vam_eval)
    cal = evaluate(va_eval, out["logits"], T=T_cal, mask=vam_eval)
    log_and_print(f"[评估{tag_note}] 位置数 n={raw['n']}（mask==1 且 >= K）")
    log_and_print(f"[评估{tag_note}] raw @T={T_READOUT}: ppl={raw['ppl']:.2f} acc={raw['acc']:.4f}")
    log_and_print(f"[评估{tag_note}] cal @T={T_cal:.3f}（训练切片自校准）: "
                  f"ppl={cal['ppl']:.2f} acc={cal['acc']:.4f}")
    return {"n": raw["n"], "T_cal": T_cal, "raw": raw, "cal": cal}


def baselines(data_dir, max_n=BASELINE_MAX_N):
    tr, trm, va, vam, vocab, meta = _load_stage_data(data_dir)
    tr_slice = tr[:BASELINE_TRAIN_TOKENS] if BASELINE_TRAIN_TOKENS else tr
    va_eval = va[:EVAL_TOKENS_LIMIT] if EVAL_TOKENS_LIMIT else va
    vam_eval = vam[:EVAL_TOKENS_LIMIT] if (vam is not None and EVAL_TOKENS_LIMIT) else vam
    log_and_print(f"[基线] n-gram 可见训练切片 {len(tr_slice)} tokens；插值 lam={DEF_NGRAM_LAMBDA}，"
                  f"alpha={DEF_NGRAM_ALPHA}；评分口径与内核一致（mask==1 且 >= K）")
    refs = ngram_reference(tr_slice, va_eval, max_n=max_n, val_mask=vam_eval)
    V = vocab.vocab_size
    log_and_print(f"{'uniform':9s} ppl={float(V):8.2f} acc={1.0 / V:.4f}")
    for k, v in refs.items():
        log_and_print(f"{k:9s} ppl={v['ppl']:8.2f} acc={v['acc']:.4f} n={v['n']}")
    return refs


# ================================================================================================
# CLI
# ================================================================================================
def _parse_args():
    ap = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} {BRANCH} {VERSION} —— 纯 CPU/无 BP 阶段链脉冲语言模型（单文件）")
    ap.add_argument("--menu", action="store_true", help="强制进入交互菜单")
    sub = ap.add_subparsers(dest="cmd")

    pd = sub.add_parser("data", help="数据准备与检查")
    pd.add_argument("--src", default=DATA_FILE)
    pd.add_argument("--out", default=DATA_DIR)
    pd.add_argument("--val-convs", type=int, default=VAL_CONVS)
    pd.add_argument("--seed", type=int, default=SPLIT_SEED)
    pd.add_argument("--preview", type=int, default=1)

    pt = sub.add_parser("train", help="训练/续训（分块）")
    pt.add_argument("--data-dir", default=DATA_DIR)
    pt.add_argument("--tag", default="bio1")
    pt.add_argument("--tokens", type=int, default=TRAIN_CHUNK_TOKENS)
    pt.add_argument("--offset", type=int, default=0)
    pt.add_argument("--resume", default=None, help="从此快照恢复权重（如 stage_runs/bio1/snapshot.npz）")
    pt.add_argument("--epochs", type=int, default=EPOCHS)
    pt.add_argument("--ckpt-every", type=int, default=CKPT_EVERY_TOKENS)
    pt.add_argument("--eval-limit", type=int, default=EVAL_TOKENS_LIMIT)
    pt.add_argument("--backend", choices=["auto", "numpy", "numba"], default=BACKEND)
    pt.add_argument("--decay-amort-steps", type=int, default=None)
    pt.add_argument("--no-gamma2", action="store_true")
    pt.add_argument("--no-copy", action="store_true")

    pe = sub.add_parser("eval", help="评估（掩码 PPL/ACC + 温度标定）")
    pe.add_argument("--data-dir", default=DATA_DIR)
    pe.add_argument("--brain", default=BRAIN_FILE)

    pb = sub.add_parser("baselines", help="n-gram 基线（掩码口径）")
    pb.add_argument("--data-dir", default=DATA_DIR)
    pb.add_argument("--max-n", type=int, default=BASELINE_MAX_N)

    pc = sub.add_parser("chat", help="生成/对话")
    pc.add_argument("--brain", default=BRAIN_FILE)
    pc.add_argument("--prompt", required=True)
    pc.add_argument("--max-new", type=int, default=GEN_MAX_NEW_TOKENS)
    pc.add_argument("--temperature", type=float, default=GEN_TEMPERATURE)
    pc.add_argument("--top-p", type=float, default=GEN_TOP_P)

    ps = sub.add_parser("stats", help="模型/内核统计信息")
    ps.add_argument("--brain", default=BRAIN_FILE)

    pv = sub.add_parser("save", help="保存模型（可选 --load 先加载）")
    pv.add_argument("--load", default=None)
    pv.add_argument("--out", default=BRAIN_FILE)

    return ap.parse_args()


# ================================================================================================
# 菜单（elite 风格）：只换内核，保留外壳交互习惯
# ================================================================================================
def _ensure_brain(model_state, brain_path=BRAIN_FILE):
    if model_state["model"] is not None:
        return True
    if not os.path.exists(brain_path):
        log_and_print(f"未找到 {brain_path}，请先训练或加载模型。")
        return False
    model_state["model"] = SNNLM.load(brain_path)
    log_and_print(f"已加载 {brain_path}（V={model_state['model'].V}）")
    return True


def main_menu():
    log_and_print(f"=== {PROJECT_NAME} {BRANCH} {VERSION} 训练与对话系统菜单 ===")
    log_and_print(f"内核：K={MODEL_K} 窗={MODEL_WINDOW_MS}ms 学习/4ms 稀疏 Hebbian+ATP；"
                  f"γ2={'开' if USE_GAMMA2 else '关'} γcopy={'开' if USE_COPY_GATE else '关'}；"
                  f"后端={BACKEND}（numba{'可用' if HAS_NUMBA else '不可用→回退 numpy'}）")
    state = {"model": None}

    while True:
        print(f"\n=== {PROJECT_NAME} {BRANCH} {VERSION} 训练与对话系统菜单 ===")
        print("1. 数据准备与检查（v5 jsonl → 字符流 + 掩码 + 防泄露切分）")
        print("2. 训练/续训（分块 + 快照）")
        print("3. 对话/续写（生成）")
        print("4. 评估（掩码 PPL/ACC + 温度标定）")
        print("5. 基线参照（unigram/bigram/trigram，掩码）")
        print("6. 保存模型")
        print("7. 加载模型")
        print("8. 生成参数（temperature/top_p/max_new）")
        print("9. 模型统计信息")
        print("10. 退出")
        choice = input("请选择: ").strip()

        if choice == "1":
            src, ext = select_training_file()
            if src is None:
                continue
            if ext != ".jsonl":
                log_and_print("数据准备仅支持 v5 jsonl（messages 格式）。")
                continue
            log_and_print(f"将使用开关：VAL_CONVS={VAL_CONVS} SPLIT_SEED={SPLIT_SEED} "
                          f"VOCAB_SIZE={VOCAB_SIZE} 输出={DATA_DIR}")
            if input("开始构建数据？(y/n): ").lower() == "y":
                build_dataset(src, DATA_DIR, VAL_CONVS, SPLIT_SEED, preview=1)
        elif choice == "2":
            tag = input("运行标签 [bio1]: ").strip() or "bio1"
            snap = Path(RUNS_DIR) / tag / "snapshot.npz"
            resume = None
            offset = 0
            if snap.exists():
                prog = Path(RUNS_DIR) / tag / "progress.json"
                prev_offset, done = 0, 0
                if prog.exists():
                    pj = json.loads(prog.read_text(encoding="utf-8"))
                    prev_offset = int(pj.get("offset", 0))
                    done = int(pj.get("tokens_done", 0))
                if input(f"检测到快照（上块 offset={prev_offset}，已完成 {done} tokens）。"
                         f"续训？(y/n): ").lower() == "y":
                    resume = str(snap)
                    offset = prev_offset + done   # 下一分块自动接续
            raw = input(f"本次训练 token 数 [{TRAIN_CHUNK_TOKENS}]: ").strip()
            tokens = int(raw) if raw else TRAIN_CHUNK_TOKENS
            raw = input(f"起始偏移 [{offset}]: ").strip()
            offset = int(raw) if raw else offset
            model, run, vocab = train_run(DATA_DIR, tag, max_tokens=tokens, offset=offset,
                                          resume=resume)
            state["model"] = model
        elif choice == "3":
            if not _ensure_brain(state):
                continue
            d = Path(DATA_DIR) / "vocab.json"
            if not d.exists():
                log_and_print("缺少 stage_data/vocab.json，请先执行数据准备。")
                continue
            vocab = CharVocab.load(d)
            prompt = input("请输入用户消息: ").strip()
            txt = generate(state["model"], vocab, [{"role": "user", "content": prompt}])
            log_and_print("[生成] " + txt)
        elif choice == "4":
            if not _ensure_brain(state):
                continue
            d = Path(DATA_DIR) / "vocab.json"
            if not d.exists():
                log_and_print("缺少 stage_data/vocab.json，请先执行数据准备。")
                continue
            vocab = CharVocab.load(d)
            eval_suite(DATA_DIR, state["model"], vocab, tag_note="(菜单)")
        elif choice == "5":
            baselines(DATA_DIR)
        elif choice == "6":
            if state["model"] is None:
                log_and_print("当前没有已训练的模型对象。")
                continue
            state["model"].save(BRAIN_FILE, compressed=True)
            log_and_print(f"已保存 {BRAIN_FILE}")
        elif choice == "7":
            if os.path.exists(BRAIN_FILE):
                state["model"] = SNNLM.load(BRAIN_FILE)
                log_and_print(f"已加载 {BRAIN_FILE}（V={state['model'].V}）")
            else:
                log_and_print(f"未找到 {BRAIN_FILE}。")
        elif choice == "8":
            global GEN_TEMPERATURE, GEN_TOP_P, GEN_MAX_NEW_TOKENS
            try:
                GEN_TEMPERATURE = float(input(f"temperature [当前 {GEN_TEMPERATURE}]: ")
                                        or GEN_TEMPERATURE)
                GEN_TOP_P = float(input(f"top_p [当前 {GEN_TOP_P}]: ") or GEN_TOP_P)
                GEN_MAX_NEW_TOKENS = int(input(f"max_new_tokens [当前 {GEN_MAX_NEW_TOKENS}]: ")
                                         or GEN_MAX_NEW_TOKENS)
            except ValueError:
                log_and_print("输入无效，保持原值。")
        elif choice == "9":
            if not _ensure_brain(state):
                continue
            m = state["model"]
            p = m.p
            var_elems = m.W.size + m.gamma2.size + m.gamma_c.size
            log_and_print(f"V={m.V} K={p.K} 窗={p.window_ms}ms 学习周期={p.learn_every_ms}ms")
            log_and_print(f"可塑参数：W {m.W.shape}（{m.W.size} 元素）+ γ2 {m.gamma2.shape} "
                          f"+ γcopy {m.gamma_c.shape}；共 {var_elems} 元素")
            log_and_print(f"W 均值={float(m.W.mean()):.6f} 非零率={float((m.W > W_NNZ_EPS).mean()):.5f} "
                          f"max={float(m.W.max()):.4f} |γ2|max={float(np.abs(m.gamma2).max()):.4f} "
                          f"|γcopy|max={float(np.abs(m.gamma_c).max()):.4f}")
            log_and_print(f"内存约 {var_elems * 4 / BYTES_PER_MB:.0f} MB（float32）")
            log_and_print(f"计算后端={BACKEND}（numba{'可用' if HAS_NUMBA else '不可用'}）；"
                          f"衰减摊销={p.decay_amort_steps} 学习步")
            log_and_print("机制：M 延迟链（每级 1 窗）+ SFA；P 读出群；Hebbian ΔW=η·dt·pre·avail·"
                          "outer(coef,post)；物质约束 B=%.1f；ATP 池；γ2/γcopy 先验门控。" % p.budget_B)
        elif choice == "10":
            break
        else:
            log_and_print("无效选择，请重试。")


def main():
    init_log_file()
    args = _parse_args()
    if args.menu or args.cmd is None:
        try:
            main_menu()
        finally:
            close_log_file()
        return
    try:
        if args.cmd == "data":
            build_dataset(args.src, args.out, args.val_convs, args.seed, preview=args.preview)
        elif args.cmd == "train":
            global BACKEND
            BACKEND = args.backend
            overrides = {}
            if args.no_gamma2:
                overrides["eta_gate2"] = 0.0
            if args.no_copy:
                overrides["eta_copy"] = 0.0
            if args.decay_amort_steps is not None:
                overrides["decay_amort_steps"] = args.decay_amort_steps
            train_run(args.data_dir, args.tag, epochs=args.epochs, max_tokens=args.tokens,
                      offset=args.offset, resume=args.resume, ckpt_every=args.ckpt_every,
                      eval_limit=args.eval_limit, params_override=overrides)
        elif args.cmd == "eval":
            d = Path(args.data_dir) / "vocab.json"
            vocab = CharVocab.load(d)
            model = SNNLM.load(args.brain)
            eval_suite(args.data_dir, model, vocab, tag_note=f"({Path(args.brain).name})")
        elif args.cmd == "baselines":
            baselines(args.data_dir, max_n=args.max_n)
        elif args.cmd == "chat":
            vocab = CharVocab.load(Path(DATA_DIR) / "vocab.json")
            model = SNNLM.load(args.brain)
            txt = generate(model, vocab, [{"role": "user", "content": args.prompt}],
                           max_new=args.max_new, T=args.temperature, top_p=args.top_p)
            log_and_print("[生成] " + txt)
        elif args.cmd == "stats":
            model = SNNLM.load(args.brain)
            m = model
            log_and_print(f"V={m.V} K={m.p.K} 参数元素={m.W.size + m.gamma2.size + m.gamma_c.size}"
                          f"（≈{(m.W.size + m.gamma2.size + m.gamma_c.size) * 4 / BYTES_PER_MB:.0f} MB）")
        elif args.cmd == "save":
            if args.load:
                model = SNNLM.load(args.load)
            else:
                log_and_print("--load 未提供，无可保存对象。")
                return
            model.save(args.out, compressed=True)
            log_and_print(f"已保存 {args.out}")
    finally:
        close_log_file()


if __name__ == "__main__":
    main()
