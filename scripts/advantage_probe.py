#!/usr/bin/env python3
"""策略改进算子的离线审计：只用已有的 v2 数据批，零算力（不训练、不跑谱）。

要回答的问题
------------
纯自蒸馏（策略目标 = 教师的根访问分布）的不动点就是教师，学生不可能在期望上超过它，
ADR 0030 因此量到"策略指标逐轮变好、搜索强度逐轮变差"。策略改进算子想把目标从
访问分布推到

    π̃_a ∝ π̂_a · exp(Â_a / β̂),   Â_a = Q(s,a) − V,   V = Σ_a π̂_a Q(s,a)

这一步改进的保证（E_{π̃}[Q] ≥ E_{π̂}[Q]）**是在同一个 Q 下**成立的，用生成 Â 的那个 Q
去度量它等于什么都没量（本脚本把它作为自检行保留，恒 ≥ 0，不是证据）。

为什么不用"另一批数据当裁判"
------------------------------
最自然的独立裁判是"另一批搜索在同一局面上的 Q"。但两条线只要有一处选招不同就各下各的棋，
局面重合只发生在开局前两步：实测把 64 模拟的 20000 个根与 2048 模拟的 199737 个局面按棋盘
对齐，只匹配上 40 个根（步号中位 1）。**所以离线无法用跨批裁判判"改进后是不是真的更好"，
这件事只能由训练轮 + 评测臂回答**（ADR 0030 §6.3 的纯价值贪心臂是便宜且与搜索臂同向的判据）。

本脚本因此只做它能可靠做的三件事：
1. **机制锚点**：复现 ADR 0030 §3/§6 的同根 std、自洽度、Q̄ 均值——用来验证解码口径没跑偏，
   并把不同教师批的标签信息量摆在同一张表上（数字单位是原始分数，与 ADR 一致）。
   同时给出 r 的同根 std 与两套自洽度（A 口径 / r+A 口径）以及"r 翻转率"：引擎的选择值是
   r + A 而价值头学的只是 A，这两者的差距就是排序被立即得分改写的比例（见 ADR 0034）。
2. **算子审计**：每个 β 下 π̃ 的 KL、归一化熵、决策改变率——回答"算子动了多少"。
   这一条决定 β 的取值：目标该有多尖、有多少决策被改写，必须在跑训练之前就看得见。
3. **自检行**：同一个 Q 下的配对差必须 ≥ 0（构造保证），用来抓实现错误。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python" / "src"))
from r2048.data.shards import read_manifest, record_dtype

FIELDS = (
    "decision",
    "legal_mask",
    "episode",
    "step",
    "candidate_value",
    "candidate_reward",
    "candidate_visits",
    "candidate_count",
)
EPS = 1e-9


def load_batch(path: Path, fields: tuple[str, ...] = FIELDS) -> dict[str, np.ndarray]:
    shards = sorted((path / "shards").glob("*.bin"))
    if not shards:
        raise SystemExit(f"{path} 下没有分片")
    # 记录布局由清单声明的 tree_nodes 决定（v3）：探针只用根候选块，但**字节数**必须按 K 算，
    # 否则读出来会错位（看起来像棋盘）。
    dtype = record_dtype(read_manifest(path).tree_nodes)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in fields}
    for shard in shards:
        view = np.memmap(shard, dtype=dtype, mode="r")
        for name in fields:
            parts[name].append(np.array(view[name]))
    return {name: np.concatenate(chunks) for name, chunks in parts.items()}


class Roots:
    """一批数据里每个决策节点的候选集合（只保留有搜索、且合法候选 ≥ 2 的根）。

    全部量保持**原始分数单位**（记录里的整数），不除以 score_scale：ADR 0030 的
    同根 std / Q̄ 均值都是这个单位，用它才能直接对齐锚点。
    """

    def __init__(self, path: Path) -> None:
        raw = load_batch(path)
        self.path = path
        legal = ((raw["legal_mask"].astype(np.uint8)[:, None] >> np.arange(4, dtype=np.uint8)) & 1) > 0
        has_candidates = raw["candidate_count"] > 0
        visits = raw["candidate_visits"].astype(np.float64) * legal
        total = visits.sum(axis=1)
        n_legal = legal.sum(axis=1)
        keep = (total > 0) & (n_legal >= 2) & has_candidates

        self.legal = legal[keep]
        self.visits = visits[keep]
        self.total = total[keep]
        self.n_legal = n_legal[keep].astype(np.float64)
        self.episode = raw["episode"][keep]
        self.step = raw["step"][keep]
        # Q(s,a) = r(s,a) + A(u_a)：候选块里存的正是这两项，所以每个根的动作值向量是**重建**的，
        # 不是新信息——这就是"训练侧就能实现策略改进、不必重新跑谱"的全部原因。
        self.reward = raw["candidate_reward"][keep].astype(np.float64) * self.legal
        self.value = raw["candidate_value"][keep].astype(np.float64) * self.legal
        self.q = (self.reward + self.value) * self.legal

    def __len__(self) -> int:
        return int(self.legal.shape[0])

    def policy_hat(self, floor: float = 0.0) -> np.ndarray:
        visits = self.visits + floor
        return visits / np.maximum(visits.sum(axis=1, keepdims=True), EPS)

    def advantages(self) -> tuple[np.ndarray, np.ndarray]:
        """Â_a = Q_a − V，V = Σ_a π̂_a Q_a（on-policy 基线）；尺度 = 同根无权重 std。

        尺度用无权重 std（与 ADR 0030 的"同根标签 std"同定义，便于对齐锚点）：
        π̂ 加权 std 会把"访问数怎么分配"的判断混进尺度，而尺度只该描述候选之间差多少。
        """
        pi = self.policy_hat()
        v = (pi * self.q).sum(axis=1, keepdims=True)
        adv = (self.q - v) * self.legal
        mean = (adv * self.legal).sum(axis=1, keepdims=True) / self.n_legal[:, None]
        var = (((adv - mean) ** 2) * self.legal).sum(axis=1) / self.n_legal
        return adv, np.sqrt(np.maximum(var, 0.0))

    def improved_policy(
        self, adv: np.ndarray, scale: np.ndarray, beta: float, floor: float, rows: np.ndarray | None = None
    ) -> np.ndarray:
        """π̃ ∝ π̂ · exp(Â / (β·scale))。

        floor 是每个合法候选的伪访问数：纯乘法重加权只能把质量在**搜索访问过的**边之间
        重分配，未访问边永远拿不到质量（ADR 0030 §6.6：NN 自举批有 0.8%~7.7% 的合法边
        访问数为 0），所以这个选择必须显式做，不能靠默认值。
        """
        pi = self.policy_hat(floor) if rows is None else self.policy_hat(floor)[rows]
        adv = adv if rows is None else adv[rows]
        scale = scale if rows is None else scale[rows]
        z = adv / np.maximum(beta * np.maximum(scale, EPS)[:, None], EPS)
        # 先减去同根最大权重再取指数：溢出防护，且不影响归一化后的分布。
        masked = np.where(self.legal if rows is None else self.legal[rows], z, -np.inf)
        z = z - masked.max(axis=1, keepdims=True)
        weighted = pi * np.exp(np.where(masked > -np.inf, z, -np.inf))
        return weighted / np.maximum(weighted.sum(axis=1, keepdims=True), EPS)


def entropy(normalized: np.ndarray, n_legal: np.ndarray) -> float:
    """归一化熵（1 = 均匀）：策略目标有多尖，是探索塌缩的直接判据（ADR 0030 §6.6）。"""
    p = np.maximum(normalized, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    h = -terms.sum(axis=1)
    return float(np.mean(h / np.log(np.maximum(n_legal, 2.0))))


def kl(p: np.ndarray, q: np.ndarray, legal: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where((p > 0) & legal, p * np.log(p / np.maximum(q, EPS)), 0.0)
    return float(terms.sum(axis=1).mean())


def argmax_masked(values: np.ndarray, legal: np.ndarray) -> np.ndarray:
    return np.where(legal, values, -np.inf).argmax(axis=1)


def same_root_std(values: np.ndarray, legal: np.ndarray, n_legal: np.ndarray) -> float:
    """同根离散度的中位数（无权重 std，与 ADR 0030 的"同根标签 std"同定义）。"""
    mean = (values * legal).sum(axis=1, keepdims=True) / n_legal[:, None]
    var = (((values - mean) ** 2) * legal).sum(axis=1) / n_legal
    return float(np.median(np.sqrt(np.maximum(var, 0.0))))


def report_anchors(roots: Roots, name: str) -> None:
    _, scale = roots.advantages()
    a, r, q = roots.value, roots.reward, roots.q
    pick = argmax_masked(roots.visits, roots.legal)
    # 与 ADR 0030 §3/§6 的同名量对齐（同根 std 取中位、自洽度 = argmax(访问数) == argmax(标签)），
    # 并**同时给出两套口径**：引擎的选择值是 r + A，而价值头学的只是 A，所以"会不会选对"必须看
    # r+A 那一列（ADR 0034）。r 的同根 std 与"翻转率"是解释两列差距的两个量——注意 r 的**全局**
    # std 是几十到八十（含大合并的尾部），**同根** std 只有几分，用错口径会把"价值项主导排序"
    # 误读成"立即得分主导排序"（ADR 0030 §3 的参照量就是全局口径，已由 0034 订正）。
    print(
        f"{name:24s} 根 {len(roots):>8d}  同根 Â std 中位 {np.median(scale):7.1f}  "
        f"r std 中位 {same_root_std(r, roots.legal, roots.n_legal):5.1f}  "
        f"自洽度 A {np.mean(pick == argmax_masked(a, roots.legal)):.3f} / "
        f"r+A {np.mean(pick == argmax_masked(q, roots.legal)):.3f}  "
        f"r 翻转 {np.mean(argmax_masked(a, roots.legal) != argmax_masked(q, roots.legal)):5.1%}  "
        f"Q̄ 均值 {q[roots.legal].mean():7.1f}  步号中位 {int(np.median(roots.step)):>4d}  "
        f"π̂ 熵 {entropy(roots.policy_hat(), roots.n_legal):.3f}"
    )


def report_sweep(roots: Roots, betas: list[float], floor: float) -> None:
    adv, scale = roots.advantages()
    pi = roots.policy_hat(floor)
    legal = roots.legal
    visit_pick = argmax_masked(roots.visits, legal)
    print(f"  π̂（访问分布，β=∞）           熵 {entropy(pi, roots.n_legal):.3f}")
    for beta in betas:
        tilde = roots.improved_policy(adv, scale, beta, floor)
        moved = np.mean(argmax_masked(tilde, legal) != visit_pick)
        gain = ((tilde - pi) * roots.q).sum(axis=1)
        print(
            f"  β={beta:<5g} KL(π̃‖π̂) {kl(tilde, pi, legal):6.3f}  "
            f"熵 {entropy(tilde, roots.n_legal):.3f}  改变决策 {moved * 100:5.1f}%  "
            f"（自检：同 Q 下配对差 {gain.mean():+8.1f} 分，必须 ≥ 0）"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--batch", action="append", required=True, help="数据批目录（可多次给出，只报锚点）")
    parser.add_argument("--sweep", default="", help="要扫描 β 的那一批（给出目录才做算子审计）")
    parser.add_argument("--betas", default="0.25,0.5,1.0,2.0,4.0,8.0")
    parser.add_argument("--floor", type=float, default=0.0, help="π̂ 的均匀底（伪访问数，0 = 不加）")
    parser.add_argument("--max-roots", type=int, default=0, help="只用前 N 个根（0 = 全部）")
    args = parser.parse_args()

    betas = [float(item) for item in args.betas.split(",")]
    print("机制锚点（原始分数单位，与 ADR 0030 同名量对齐）：")
    for path in args.batch:
        report_anchors(Roots(Path(path)), Path(path).name)
    if args.sweep:
        roots = Roots(Path(args.sweep))
        if args.max_roots > 0 and len(roots) > args.max_roots:
            for name in ("legal", "visits", "total", "n_legal", "episode", "step", "reward", "value", "q"):
                setattr(roots, name, getattr(roots, name)[: args.max_roots])
        print(f"\n算子审计 {Path(args.sweep).name}（floor={args.floor}，{len(roots)} 个根）：")
        report_sweep(roots, betas, args.floor)


if __name__ == "__main__":
    main()
