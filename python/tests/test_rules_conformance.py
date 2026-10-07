"""跨语言规则一致性：Python 参考实现生成期望向量，C++ 二进制逐条比对。

这是唯一一处“另一门语言算出来的期望值”。规则错误会污染全部训练数据，
而且很难在事后发现，所以这条检查放在最高优先级：随机棋盘 + 边界局面，
四个方向的结果与得分都必须逐格一致。
"""

from __future__ import annotations

import os
import random
import struct
import subprocess
from pathlib import Path

import pytest

from r2048.reference import rules_reference as reference

HEADER = struct.Struct("<8sII")  # magic, version, count
RECORD_PREFIX = struct.Struct("<16B")
OUTCOME = struct.Struct("<16BQB")  # afterstate, score, valid
VECTOR_VERSION = 1

# 指数上界的约定：u8 表示在规则上不设上限，但 4×4 的理论上界是 2^17，
# 测试向量覆盖到 30 已经远超任何可达局面。
MAX_TEST_EXPONENT = 30


def _binary_path() -> Path:
    raw = os.environ.get("R2048_TEST_BINARY")
    if not raw:
        pytest.skip("未设置 R2048_TEST_BINARY：通过 make test 运行，或先 make build")
    path = Path(raw)
    if not path.exists():
        pytest.skip(f"测试二进制不存在（{path}）：先跑 make build")
    return path


def _random_boards(rng: random.Random, count: int) -> list[tuple[int, ...]]:
    boards: list[tuple[int, ...]] = []
    for _ in range(count):
        cells = [0] * 16
        fill = rng.randrange(0, 17)
        for _ in range(fill):
            cells[rng.randrange(16)] = rng.choice(
                [1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 12, 17, MAX_TEST_EXPONENT]
            )
        boards.append(tuple(cells))
    return boards


def _edge_boards() -> list[tuple[int, ...]]:
    boards = [
        tuple([0] * 16),  # 空盘
        tuple([1] * 16),  # 满盘同值：一次移动连锁合并
        tuple([1, 2] * 8),  # 满盘交错：完全无法移动
        tuple([MAX_TEST_EXPONENT] * 16),  # 极大指数满盘
        (1, 1, 0, 0) + (0,) * 12,  # 跨半区合并（左端）
        (0, 0, 1, 1) + (0,) * 12,  # 跨半区合并（右端）
        (0,) + (1, 1, 1) + (0,) * 12,  # 跨半区合并优先级的关键用例
        (2, 2, 2, 0) + (0,) * 12,  # 合并后不再参与本回合合并
        (2, 2, 2, 2) + (0,) * 12,
        (0,) * 12 + (1, 0, 1, 0),  # 间隔块压缩
        (5, 0, 5, 0) + (0,) * 12,
    ]
    # 每条线（行与列）的全部组合：指数取自 0..4，覆盖半区内部与跨半区的所有形态。
    for a in range(5):
        for b in range(5):
            for c in range(5):
                for d in range(5):
                    row = (a, b, c, d)
                    for index in range(4):
                        board = [0] * 16
                        for offset in range(4):
                            board[index * 4 + offset] = row[offset]
                        boards.append(tuple(board))
                        column = [0] * 16
                        for offset in range(4):
                            column[offset * 4 + index] = row[offset]
                        boards.append(tuple(column))
    return boards


def _write_vectors(path: Path, boards: list[tuple[int, ...]]) -> None:
    with path.open("wb") as handle:
        handle.write(HEADER.pack(b"R2048VEC", VECTOR_VERSION, len(boards)))
        for board in boards:
            handle.write(RECORD_PREFIX.pack(*board))
            for action in reference.ACTIONS:
                outcome = reference.move(board, action)
                handle.write(OUTCOME.pack(*outcome.board, outcome.score, 1 if outcome.valid else 0))


def test_cpp_rules_match_reference(tmp_path: Path) -> None:
    rng = random.Random(20261006)
    boards = _edge_boards() + _random_boards(rng, 20000)
    vector_path = tmp_path / "rules_vectors.bin"
    _write_vectors(vector_path, boards)

    completed = subprocess.run(
        [str(_binary_path()), "rules-vectors", str(vector_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"C++ 侧规则与参考实现不一致（{len(boards)} 条向量）：\n{completed.stdout}\n{completed.stderr}"
    )
    assert "通过" in completed.stdout
