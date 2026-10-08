"""会话：一个盘面 + 一条棋谱 + 一个引擎，一步一步走。

它是界面（以及任何脚本）与引擎之间的那一层：界面不做棋规、不选招，只把命令交给会话，
再把会话回报的状态画出来。这样"选招口径只有引擎一处"（ADR 0006）在界面这里也成立。

**棋谱是唯一事实源。** 盘面、得分、步数、合法方向都由重放棋谱算出来，而不是在别处增量维护。
代价是每走一步要重放一遍整条棋谱（几百条命令、微秒级），换来的是不可能出现"记录的棋"与
"屏幕上的棋"不一致——而那种不一致在复盘时正是最难发现、也最致命的错误。`undo` 也因此在
同一处收口：把棋谱截到上一个 `move` 之前再重放即可，不需要保存随机数状态（ADR 0036）。

生成块由**引擎**生成并回报（`commit_step` 的 `spawn_index/spawn_exponent`），会话把它写成
一条显式的 `set` 进棋谱。所以本模块一行规则都不实现。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import r2048_core
import torch

from r2048 import records
from r2048.fileio import write_text_atomic
from r2048.nn.model import load_checkpoint
from r2048.runtime.inference import BatchedEvaluator

# 会话可选的决策口径。名字与界面上的三档一致；引擎侧的枚举在 batched_engine.h。
STRATEGIES = {
    "search": r2048_core.DECISION_SEARCH,
    "policy": r2048_core.DECISION_POLICY_GREEDY,
    "value": r2048_core.DECISION_VALUE_GREEDY,
}
# 前两档不做树搜索：它们的"分析"就是把根局面求值一次（拿先验或 r+A），所以预算是 1。
GREEDY_STRATEGIES = ("policy", "value")


class SessionNotReady(RuntimeError):
    """没有加载模型时试图分析或落子。编辑与复盘不需要模型，分析与对局需要。"""


@dataclass
class SessionConfig:
    checkpoint: str = ""
    # 交互分析默认 CPU：旧版实测 batch 很小（1~5 行）时 CUDA 反而更慢（0.77ms vs 0.42ms），
    # 而且这样不会和跑谱/训练抢那张唯一的 GPU。
    device: str = "cpu"
    strategy: str = "search"
    simulations: int = 256
    time_ms: int = 0  # 0 = 不限时；与 simulations 先到者为准
    in_flight: int = 4
    cache_entries: int = 1 << 16
    max_batch: int = 4096
    c_puct: float = 1.5
    seed: int = 0


@dataclass
class AnalysisReport:
    """一次分析的结果。字段与 `BatchedEngine::RootReport` 一一对应，另加协议快照。"""

    board: np.ndarray
    legal_mask: int
    strategy: str
    simulations: int
    visits: list[int] = field(default_factory=list)
    q_mean: list[float] = field(default_factory=list)
    one_ply: list[float] = field(default_factory=list)
    prior: list[float] = field(default_factory=list)
    root_value: float = 0.0
    best_action: int = -1
    seconds: float = 0.0
    protocol: dict = field(default_factory=dict)
    # 流式帧用：true 表示"搜索还在进行，这是到目前为止的结果"，false 是终帧。
    during_search: bool = False

    def payload(self) -> dict:
        return {
            "is_during_search": self.during_search,
            "strategy": self.strategy,
            "simulations": self.simulations,
            "legal_mask": self.legal_mask,
            "best_action": self.best_action,
            "root_value": round(self.root_value, 3),
            "seconds": round(self.seconds, 3),
            "protocol": self.protocol,
            "directions": [
                {
                    "name": records.DIRECTIONS[action],
                    "action": action,
                    "legal": bool((self.legal_mask >> action) & 1),
                    "visits": self.visits[action],
                    "q_mean": round(self.q_mean[action], 1),
                    "one_ply": round(self.one_ply[action], 1),
                    "prior": round(self.prior[action], 4),
                }
                for action in range(len(records.DIRECTIONS))
            ],
        }


class Session:
    """一个盘面 + 一条棋谱的会话。线程不安全：调用方负责串行化（界面里由单一持有者保证）。"""

    def __init__(self, config: SessionConfig | None = None) -> None:
        self.config = config or SessionConfig()
        if self.config.strategy not in STRATEGIES:
            raise ValueError(f"未知策略 {self.config.strategy!r}，可选 {'/'.join(STRATEGIES)}")
        self.record: list[records.Command] = []
        self.board = np.zeros(records.BOARD_CELLS, dtype=np.uint8)
        self.score = 0
        self.moves = 0
        self.max_exponent = 0
        self.legal_mask = 0
        # 这一局的开局种子（`new_game` 记下来）；0 表示不是用 new_game 起的局。
        self.seed = 0

        self._model: torch.nn.Module | None = None
        self._metadata: dict = {}
        self._evaluator: BatchedEvaluator | None = None
        self._device = torch.device(self.config.device)
        self._engine: r2048_core.BatchedEngine | None = None
        # 当前搜索树是否与盘面同步且已完成一次分析。用来避免"自动模式每走一步都重搜一遍"：
        # analyze 之后紧接着 play 可以直接用那棵树（引擎的选招读的就是它）。
        self._tree_fresh = False
        # 最近一次分析的报告。界面在 genmove 之后仍要显示"为什么走了这一步"，而那时树已经
        # 被提交动作用掉了——所以报告要留下来。
        self.last_report: AnalysisReport | None = None
        if self.config.checkpoint:
            self._load_model(self.config.checkpoint)
        self._recompute()
        if self._model is not None:
            self._new_engine()

    # ---- 模型与引擎 -------------------------------------------------------------

    def _load_model(self, checkpoint: str) -> None:
        if self._device.type == "cuda" and not torch.cuda.is_available():
            self._device = torch.device("cpu")
        self._model, self._metadata = load_checkpoint(checkpoint, self._device)
        self._evaluator = BatchedEvaluator(self._model, self._device)

    def _new_engine(self) -> None:
        """建（或重建）会话引擎。

        策略改变要重建，因为 `decision_mode` 在引擎构造时固定；而模拟次数不需要——它在
        每次 `begin_analysis` 里作为预算给出。重建只重新分配叶子缓存，模型与前向器不动。
        """
        assert self._model is not None
        self._engine = r2048_core.BatchedEngine(
            {
                "lanes": 1,  # 会话只对一个盘面工作（C++ 侧会拒绝别的配置，见 ADR 0037）
                "simulations": max(1, self.config.simulations),
                "in_flight_per_lane": self.config.in_flight,
                "c_puct": self.config.c_puct,
                # 会话要确定性：不做开局按访问分布采样，选招一律取最大访问数。
                "temperature": 0.0,
                "temperature_moves": 0,
                "cache_entries": self.config.cache_entries,
                "score_scale": float(r2048_core.SCORE_SCALE),
                "seed": self.config.seed,
                "decision_mode": STRATEGIES[self.config.strategy],
                "collect_samples": 0,
            },
            "",  # 会话不写数据批
            "",
            str(self._metadata.get("model_id", "")) or "session",
            0,
            0,
        )
        self._engine.set_root(self.board)
        self._tree_fresh = False

    def require_engine(self) -> r2048_core.BatchedEngine:
        if self._engine is None:
            raise SessionNotReady(
                "这个会话没有模型：编辑与复盘不需要模型，分析（analyze）与落子（play）需要"
                "启动时给出 --checkpoint"
            )
        return self._engine

    def set_strategy(self, strategy: str) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"未知策略 {strategy!r}，可选 {'/'.join(STRATEGIES)}")
        if strategy == self.config.strategy and self._engine is not None:
            return
        self.config.strategy = strategy
        if self._model is not None:
            self._new_engine()

    # ---- 盘面与棋谱 -------------------------------------------------------------

    def _recompute(self) -> None:
        """按棋谱重放，得出全部派生状态。唯一的取数处——盘面与棋谱不可能不一致。"""
        result = records.apply(self.record)
        self.board = result.board
        self.score = result.total_score
        self.moves = result.move_count
        self.max_exponent = result.max_exponent
        self.legal_mask = result.legal_mask
        self._tree_fresh = False  # 盘面变了，上一步的搜索结果作废
        self.last_report = None
        if self._engine is not None:
            self._engine.set_root(self.board)

    def terminal(self) -> bool:
        return self.legal_mask == 0

    def _require_playable(self) -> None:
        """空局面与终局都是"没有合法方向"，但要对用户说清是哪一种——两者的下一步动作不同。"""
        if self.legal_mask:
            return
        if self.max_exponent == 0:
            raise RuntimeError("当前盘面是空的：先摆几个块（set）再分析或落子")
        raise RuntimeError("当前盘面已经没有合法方向（终局）")

    def reset(self) -> None:
        """清空盘面与棋谱（`clear`）。"""
        self.record = [records.Clear()]
        self.seed = 0
        self._recompute()

    def new_game(self, seed: int | None = None) -> int:
        """开一局标准 2048：两枚随机块（90% 出 2、10% 出 4）。

        开局走 C++ 的 `opening_board(seed, episode)` —— **与跑谱用的是同一个函数**，所以界面
        上开的局与训练数据里的开局是同一分布、同一条随机流推导，而不是 Python 另写一遍生成规则
        （规则只实现一次，ADR 0004/0005）。

        棋谱写下来的仍然是显式的 `set`，于是这一步又把随机性固定进了文件（ADR 0036 决策 3）：
        同一个 seed 开的局能被逐字节重放，换个 seed 就是另一局。
        """
        value = int(seed) if seed else int(time.time() * 1000) & 0x7FFFFFFF
        board = np.asarray(r2048_core.opening_board(value, 0), dtype=np.uint8)
        self.record = records.commands_for_board(board)
        self.seed = value
        self._recompute()
        return value

    def set_tile(self, index: int, exponent: int | None) -> None:
        """摆一块或清一块（棋谱里的 `set`）。合法性由 records.apply 判定。"""
        self.record.append(records.SetTile(index=index, exponent=exponent))
        try:
            self._recompute()
        except records.RecordApplyError:
            self.record.pop()  # 应用不了：棋谱维持原样，错误照样往上抛
            raise

    # ---- 落子 -------------------------------------------------------------------

    def play(self, action: int | None = None) -> dict:
        """走一步。`action=None` 表示让引擎按当前策略自己选。

        生成块由引擎生成并回报，会话把它写成一条显式的 `set`——于是这盘棋的随机性落在
        棋谱里，与随机数流无关（ADR 0036）。
        """
        engine = self.require_engine()
        self._require_playable()
        if action is None and not self._tree_fresh:
            # 让引擎自己选招，前提是有一棵刚跑满的树。否则根上全是 0，引擎只能走
            # "编号最小的合法方向"那种退化选择——那不是"AI 的选择"，而是没算过。
            self.analyze()
        before = list(self.record)
        step = engine.commit_step(-1 if action is None else int(action))
        if int(step["action"]) < 0 or bool(step["terminal"]):
            return {"action": -1, "terminal": True, "board": self.board.tolist(), "score": self.score}
        self.record.append(records.Move(action=int(step["action"])))
        if int(step["spawn_index"]) >= 0:
            self.record.append(
                records.SetTile(index=int(step["spawn_index"]), exponent=int(step["spawn_exponent"]))
            )
        try:
            self._recompute()
        except records.RecordApplyError:
            # 这一步会把局面带出键表（两个 2^17 合并出 2^18）：这种局面没有任何棋谱能表示它，
            # 所以在写进棋谱之前就拒绝，并把引擎也退回原局面（见 ADR 0036 决策 6）。
            self.record[:] = before
            self._recompute()
            raise
        return {
            "action": int(step["action"]),
            "terminal": bool(step["terminal"]),
            "spawn_index": int(step["spawn_index"]),
            "spawn_exponent": int(step["spawn_exponent"]),
            "board": self.board.tolist(),
            "score": self.score,
        }

    def undo(self) -> bool:
        """退回上一个 `move` 之前（连同那一步的生成块与之后的手工改动）。

        因为棋谱是唯一事实源，这里只需要截断再重放——不需要保存随机数状态，
        也不需要引擎提供"反走一步"。
        """
        last_move = next(
            (
                index
                for index in range(len(self.record) - 1, -1, -1)
                if isinstance(self.record[index], records.Move)
            ),
            None,
        )
        if last_move is None:
            return False
        del self.record[last_move:]
        self._recompute()
        return True

    # ---- 分析 -------------------------------------------------------------------

    def analyze(
        self,
        on_frame: Callable[[AnalysisReport], None] | None = None,
        progress_every: float = 0.15,
    ) -> AnalysisReport:
        """从当前盘面分析一次（不落子）。

        `on_frame` 是流式用的：搜索进行中每 `progress_every` 秒回调一次"到目前为止"的报告，
        于是界面可以在整次搜索结束之前就把访问数的增长画出来（旧版 GTP 扩展里
        `is_during_search` 那套帧就是这个形状）。返回的是终帧。
        """
        engine = self.require_engine()
        self._require_playable()
        # 贪心档不做搜索：它的"分析"就是把根求值一次（策略档拿先验、价值档拿 r+A）。
        budget = 1 if self.config.strategy in GREEDY_STRATEGIES else max(1, self.config.simulations)
        engine.begin_analysis(budget)
        deadline = time.monotonic() + self.config.time_ms / 1000.0 if self.config.time_ms > 0 else None
        started = time.monotonic()
        last_frame = started
        while True:
            advance = engine.advance(self.config.max_batch)
            ids = advance["request_ids"]
            if ids.size:
                values, logits = self._evaluator.evaluate(advance["request_kinds"], advance["boards"])
                engine.submit(ids, values, logits)
            if advance["done"]:
                break
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break  # 时间到：手上的根统计就是"到目前为止"的结果（simulations 如实反映）
            if on_frame is not None and now - last_frame >= progress_every:
                last_frame = now
                on_frame(self._report(engine, now - started, during_search=True))
            if not ids.size:
                raise RuntimeError("分析卡住：既无待求值请求，也未跑满预算")
        self._tree_fresh = True  # 这棵树刚跑满：紧接着的 play 可以直接用它选招
        self.last_report = self._report(engine, time.monotonic() - started, during_search=False)
        return self.last_report

    def _report(self, engine, seconds: float, during_search: bool) -> AnalysisReport:
        raw = engine.report_root()
        return AnalysisReport(
            board=self.board,
            legal_mask=int(raw["legal_mask"]),
            strategy=self.config.strategy,
            simulations=int(raw["simulations"]),
            visits=[int(value) for value in raw["visits"]],
            q_mean=[float(value) for value in raw["q_mean"]],
            one_ply=[float(value) for value in raw["one_ply"]],
            prior=[float(value) for value in raw["prior"]],
            root_value=float(raw["root_value"]),
            best_action=int(raw["best_action"]),
            seconds=seconds,
            protocol=self.protocol(),
            during_search=during_search,
        )

    def protocol(self) -> dict:
        """这次分析实际生效的口径。

        必须跟着结果一起回给界面：会话的批组成（lanes=1）与评测不同，而批组成会改变结果
        （ADR 0017），所以"分析面板说 AI 会走 X"与"评测里 nn_mcts 走 X"**可能不一致**。
        这不是 bug，但要让看的人知道——否则他会以为是同一个数。
        """
        return {
            "kind": "session-analysis",
            "strategy": self.config.strategy,
            "simulations": self.config.simulations,
            "time_ms": self.config.time_ms,
            "lanes": 1,
            "in_flight": self.config.in_flight,
            "cache_entries": self.config.cache_entries,
            "c_puct": self.config.c_puct,
            "device": self._device.type,
            "model_id": str(self._metadata.get("model_id", "")),
            "rules_version": int(r2048_core.RULES_VERSION),
            "note": "交互分析口径：批组成与评测不同，选招可能与评测不一致（ADR 0017）",
        }

    # ---- 状态与文件 -------------------------------------------------------------

    def render(self) -> str:
        return records.render(self.record)

    def state(self) -> dict:
        return {
            "board": [int(cell) for cell in self.board],
            "score": self.score,
            "moves": self.moves,
            "seed": self.seed,
            "max_exponent": self.max_exponent,
            "max_face": (1 << self.max_exponent) if self.max_exponent else 0,
            "legal_mask": self.legal_mask,
            "terminal": self.terminal(),
            "strategy": self.config.strategy,
            "model_id": str(self._metadata.get("model_id", "")),
            "device": self._device.type,
            "record": self.render(),
        }

    def save(self, path: str | Path) -> Path:
        return write_text_atomic(path, self.render())

    def load(self, path: str | Path) -> None:
        """载入一条棋谱。与 save 对称；载入后棋谱就是唯一事实源。"""
        self.record = records.load(path)
        if not self.record:
            self.record = [records.Clear()]
        self._recompute()
