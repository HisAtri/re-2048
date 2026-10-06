"""2048 规则语义的 Python 参考实现（跨语言一致性测试的判据）。

语义与旧版实现保持一致，避免历史数据在规则上错位：

* 棋盘是 16 个“指数”（0 = 空，1 = 2，2 = 4，…），行优先。
* 一次移动把每条线上的非零块压缩后从左到右贪心合并，每个块每回合至多合并一次。
* 合并得分是新块数值（即 2^(指数)），本次移动的得分是所有合并之和。
* 无效移动不改变棋盘、不得分；有效移动后由环境在有空格中均匀生成 2（90%）或 4（10%）。
* 无合法移动即终局；达到 2048 后继续游戏（本项目研究 4096 以上的大块）。

动作顺序冻结为 UP=0、RIGHT=1、DOWN=2、LEFT=3，与 C++ 侧 `r2048::Action` 一致。
"""

from __future__ import annotations

from typing import NamedTuple

UP, RIGHT, DOWN, LEFT = range(4)
ACTIONS = (UP, RIGHT, DOWN, LEFT)

BOARD_CELLS = 16
BOARD_SIDE = 4

Board = tuple[int, ...]


class MoveResult(NamedTuple):
    board: Board
    score: int
    valid: bool


def _lines(action: int) -> tuple[tuple[int, ...], ...]:
    """按移动方向给出每一条线取格顺序：线的首元素是移动的“目标端”。"""
    if action == LEFT:
        return tuple(tuple(row * 4 + col for col in range(4)) for row in range(4))
    if action == RIGHT:
        return tuple(tuple(row * 4 + col for col in range(3, -1, -1)) for row in range(4))
    if action == UP:
        return tuple(tuple(row * 4 + col for row in range(4)) for col in range(4))
    if action == DOWN:
        return tuple(tuple(row * 4 + col for row in range(3, -1, -1)) for col in range(4))
    raise ValueError(f"未知方向: {action}")


_ACTION_LINES = tuple(_lines(action) for action in ACTIONS)


def move(board: Board, action: int) -> MoveResult:
    if len(board) != BOARD_CELLS:
        raise ValueError("棋盘必须包含 16 个指数")
    result = list(board)
    score = 0
    for line in _ACTION_LINES[action]:
        tiles = [board[index] for index in line if board[index] != 0]
        merged: list[int] = []
        cursor = 0
        while cursor < len(tiles):
            if cursor + 1 < len(tiles) and tiles[cursor] == tiles[cursor + 1]:
                exponent = tiles[cursor] + 1
                merged.append(exponent)
                score += 1 << exponent
                cursor += 2
            else:
                merged.append(tiles[cursor])
                cursor += 1
        merged.extend([0] * (4 - len(merged)))
        for index, tile in zip(line, merged):
            result[index] = tile
    afterstate = tuple(result)
    return MoveResult(afterstate, score, afterstate != board)


def legal_moves(board: Board) -> tuple[MoveResult, ...]:
    return tuple(result for action in ACTIONS if (result := move(board, action)).valid)


def is_terminal(board: Board) -> bool:
    return not legal_moves(board)


def legal_mask(board: Board) -> int:
    """与 C++ 侧 legal_mask 同构：bit a = 1 表示方向 a 合法。"""
    mask = 0
    for action in ACTIONS:
        if move(board, action).valid:
            mask |= 1 << action
    return mask


def max_exponent(board: Board) -> int:
    return max(board, default=0)
