"""局面探针：对一个局面打印网络的两个头在做什么。

只做前向与整理，**不做选招**——选招口径只有 C++ 引擎一处（ADR 0006）。
探针与引擎给出不同答案，比没有探针更糟，所以这里连 argmax 都不做：
它把两个头的原始输出摆出来，让人自己看。

抽成库函数（而不是留在 CLI 里）是为了让界面与脚本复用同一条口径：
`r2048 probe` 的子命令现在只负责把这里的结果打印成表格。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import r2048_core
import torch

from r2048.nn.encode import encode_boards
from r2048.nn.model import load_checkpoint

DIRECTION_NAMES = ("上", "右", "下", "左")  # 顺序 = C++ 的动作顺序契约 kUp/kRight/kDown/kLeft


@dataclass
class ProbeResult:
    board: np.ndarray  # (16,) 指数
    legal_mask: int
    model_id: str
    architecture: str
    merge_scores: list[int] = field(default_factory=list)  # 每个方向的即时合并得分
    afterstate_values: list[float] = field(default_factory=list)  # A(u)，单位是原始分数
    policy: list[float] = field(default_factory=list)  # 掩掉非法方向后的策略分布

    def directional_scores(self) -> list[float]:
        """r + A(u)：网络对每个方向的直接估计，与价值头的训练目标同量纲。"""
        return [score + value for score, value in zip(self.merge_scores, self.afterstate_values)]


def opening_board(seed: int) -> np.ndarray:
    """随机开一个局（两枚随机块，90% 出 2、10% 出 4）。

    注意：这里用的是 Python 的 `random`，**不是引擎的开局序列**
    （引擎按 `derive_seed(seed, kSaltEnvironment, 局号, 步号, 0)` 派生）。
    也就是说同一个 seed 并不给出引擎第 0 局的开局——探针只需要一个合法的随机开局，
    想复现某局的具体局面应当直接给 `--board`。
    """
    rng = random.Random(seed)
    board = np.zeros(16, dtype=np.uint8)
    for _ in range(2):
        empty = [index for index in range(16) if board[index] == 0]
        board[rng.choice(empty)] = 2 if rng.random() < 0.1 else 1
    return board


def parse_board(text: str) -> np.ndarray:
    values = [int(part) for part in text.replace(",", " ").split()]
    if len(values) != 16:
        raise ValueError(f"--board 需要 16 个指数，收到 {len(values)} 个")
    return np.array(values, dtype=np.uint8)


def probe_position(
    checkpoint: str,
    board: np.ndarray | None = None,
    seed: int = 1,
    device: str = "cuda",
    model: torch.nn.Module | None = None,
    metadata: dict | None = None,
) -> ProbeResult:
    """对单个局面做一次前向。`model`/`metadata` 已给定时不再从 checkpoint 载入。"""
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        target = torch.device("cpu")
    if model is None:
        model, loaded = load_checkpoint(checkpoint, target)
        metadata = loaded
    metadata = metadata or {}
    if board is None:
        board = opening_board(seed)

    scale = float(r2048_core.SCORE_SCALE)
    legal = int(r2048_core.legal_mask(board))

    # 第 0 行是决策局面（喂策略头），后面四行是各方向的 afterstate（喂价值头）。
    # 一次前向同时算两种请求，这正是两阶段批推理协议的形状（ADR 0003）。
    rows = [board]
    scores: list[int] = []
    for action in range(4):
        outcome = r2048_core.apply_move(board, action)
        rows.append(np.asarray(outcome["board"], dtype=np.uint8))
        scores.append(int(outcome["score"]))

    with torch.no_grad():
        value, logits = model(encode_boards(torch.from_numpy(np.stack(rows)).to(torch.long).to(target)))
    afterstate_values = (value[1:] * scale).tolist()
    logits = logits[0].float().cpu()
    masked = torch.where(
        torch.tensor([(legal >> action) & 1 == 1 for action in range(4)], dtype=torch.bool),
        logits,
        torch.tensor(float("-inf")),
    )
    return ProbeResult(
        board=board,
        legal_mask=legal,
        model_id=str(metadata.get("model_id", "")),
        architecture=str(metadata.get("architecture", "")),
        merge_scores=scores,
        afterstate_values=[float(item) for item in afterstate_values],
        policy=[float(item) for item in torch.softmax(masked, dim=-1).tolist()],
    )
