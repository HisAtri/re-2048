"""棋谱：对局记录的文本格式与编解码（规格见 ADR 0036）。

格式只有三条命令，一行一条：

    record_format 1
    rules_version 1
    clear
    set 2 a1
    set 4 b3
    move left
    set 2 c2

三条设计要点，都是为了让"记录能被独立重放"这件事成立：

* **生成块也是显式的 `set`**：2048 的随机性只在生成块这一处。把它写下来之后，重放不需要
  种子、不需要 PRNG，`undo` 也只需要"截断棋谱再重放"而不必保存随机数状态。
* **解析与应用分两层**：`parse` 只认语法与值域键表（不持棋盘、不判规则）；`apply` 把命令
  作用到棋盘上（`move` 走 C++ 的 `apply_move`，规则权威仍在那一处）。应用不了就失败退出，
  **不静默跳过**——静默跳过的记录，读起来和走出来的会是两盘不同的棋。
* **值域是键表而非计算式**：`null` + 2^1…2^17。值空间有限且可枚举，表天然可扩展
  （追加键并抬 `record_format`）。上限 2^17 是 4×4 的理论上界（ADR 0005），于是
  "记录里的值只可能是键表成员"是一条结构性质，不需要事后检查维持。

注意：**本格式里的值是面值**（`set 2048 a1`），而本项目内部棋盘与 `probe --board` 用的是
**指数**（同一个块在那里写作 `11`）。两套写法的转换只在本模块发生。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import r2048_core

from r2048.fileio import write_text_atomic

RECORD_FORMAT_VERSION = 1
BOARD_SIDE = 4
BOARD_CELLS = 16
# 4×4 的理论上界：2^17 = 131072（ADR 0005 已写明这个数）。键表到此为止。
MAX_EXPONENT = 17

# 方向名与动作顺序契约（ADR 0004：UP=0, RIGHT=1, DOWN=2, LEFT=3）逐一对应。
DIRECTIONS = ("up", "right", "down", "left")
ACTION_OF_DIRECTION = {name: action for action, name in enumerate(DIRECTIONS)}

# 键表：面值文本 → 指数。空格在棋谱里只有 `null` 一种写法——不给它数字写法，
# 否则 `set 0 a1` 与 `set null a1` 会成为同一条命令的两种拼法。
FACE_TO_EXPONENT = {str(1 << exponent): exponent for exponent in range(1, MAX_EXPONENT + 1)}
EXPONENT_TO_FACE = {exponent: face for face, exponent in FACE_TO_EXPONENT.items()}
MAX_FACE = str(1 << MAX_EXPONENT)
VALID_TILE_EXPONENTS = frozenset(range(1, MAX_EXPONENT + 1))
EMPTY_TOKEN = "null"

HEADER_KEYS = ("record_format", "rules_version")


class RecordError(Exception):
    """棋谱相关错误的基类：让调用方可以一次捕获下面两种。"""


class RecordSyntaxError(RecordError):
    """这一行不是一条合法命令（语法或值域不合法）。由 `parse` 抛。"""


class RecordApplyError(RecordError):
    """命令本身合法，但应用不到当前盘面上。由 `apply` 抛（ADR 0036 决策 7）。"""


@dataclass(frozen=True)
class Clear:
    """把棋盘复位为空。"""


@dataclass(frozen=True)
class SetTile:
    """在 `index` 处放下或清除一个块；`exponent` 为 None 表示清除（棋谱里的 `null`）。"""

    index: int
    exponent: int | None


@dataclass(frozen=True)
class Move:
    action: int


Command = Clear | SetTile | Move


# ---- 坐标 -----------------------------------------------------------------------


def coordinate_to_index(text: str) -> int:
    """`a1` = 左上角：字母是列（左→右 a..d），数字是行（上→下 1..4）。

    与 GTP 的行序（自下而上）相反，所以它必须写进规范，不能依赖"跟 GTP 一样"的默契。
    """
    if len(text) != 2 or not ("a" <= text[0] <= "d") or not ("1" <= text[1] <= "4"):
        raise RecordSyntaxError(f"坐标 {text!r} 不合法：形如 a1..d4（字母是列，数字是行）")
    column = ord(text[0]) - ord("a")
    row = ord(text[1]) - ord("1")
    return row * BOARD_SIDE + column


def index_to_coordinate(index: int) -> str:
    row, column = divmod(index, BOARD_SIDE)
    return f"{chr(ord('a') + column)}{row + 1}"


# ---- 解析（只管语法与键表）-------------------------------------------------------


def parse(text: str) -> list[Command]:
    """把棋谱文本解析成命令序列。

    只做语法与值域校验：不持有棋盘、不判断规则——"这一步走得合不合法"由 `apply` 回答。
    """
    header: dict[str, str] = {}
    commands: list[Command] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            word = line.split()
            if not commands and word[0] in HEADER_KEYS:
                header[word[0]] = _parse_header_value(word)
                continue
            commands.append(_parse_command(word))
        except RecordSyntaxError as error:
            raise RecordSyntaxError(f"棋谱第 {number} 行：{error}") from error
    _check_header(header)
    return commands


def _parse_header_value(word: list[str]) -> str:
    if len(word) != 2 or not word[1].isdigit():
        raise RecordSyntaxError(f"文件头应为 `{word[0]} <整数>`，收到 {' '.join(word)!r}")
    return word[1]


def _check_header(header: dict[str, str]) -> None:
    missing = [key for key in HEADER_KEYS if key not in header]
    if missing:
        raise RecordSyntaxError(
            "棋谱缺少文件头 " + "、".join(missing) + f"：每条记录都必须带上 {list(HEADER_KEYS)}"
        )
    if int(header["record_format"]) != RECORD_FORMAT_VERSION:
        raise RecordSyntaxError(
            f"棋谱格式版本 {header['record_format']} 本实现不认识（本实现是 {RECORD_FORMAT_VERSION}）"
        )
    rules = int(r2048_core.RULES_VERSION)
    if int(header["rules_version"]) != rules:
        raise RecordSyntaxError(
            f"棋谱的规则版本 {header['rules_version']} 与当前内核 {rules} 不一致："
            "用错语义解释一份记录比读不出来危险，所以拒收（ADR 0036 决策 2）"
        )


def _parse_command(word: list[str]) -> Command:
    verb = word[0]
    if verb == "clear":
        if len(word) != 1:
            raise RecordSyntaxError("clear 不接受参数")
        return Clear()
    if verb == "set":
        if len(word) != 3:
            raise RecordSyntaxError("set 需要 `<值> <坐标>` 两个参数")
        return SetTile(index=coordinate_to_index(word[2]), exponent=_parse_value(word[1]))
    if verb == "move":
        if len(word) != 2:
            raise RecordSyntaxError("move 需要一个方向参数")
        if word[1] not in ACTION_OF_DIRECTION:
            raise RecordSyntaxError(f"未知方向 {word[1]!r}，可选 {'/'.join(DIRECTIONS)}")
        return Move(action=ACTION_OF_DIRECTION[word[1]])
    raise RecordSyntaxError(f"未知命令 {verb!r}：只有 clear / set / move")


def _parse_value(text: str) -> int | None:
    if text == EMPTY_TOKEN:
        return None
    if text not in FACE_TO_EXPONENT:
        raise RecordSyntaxError(
            f"{text!r} 不是合法的块值：只接受 {EMPTY_TOKEN} 或 2 的 1..{MAX_EXPONENT} 次幂（2…{MAX_FACE}）"
        )
    return FACE_TO_EXPONENT[text]


# ---- 应用（把命令作用到棋盘上）---------------------------------------------------


@dataclass(frozen=True)
class ReplayResult:
    """重放一条棋谱之后的局面与统计。"""

    board: np.ndarray  # (16,) 指数
    total_score: int
    move_count: int
    max_exponent: int
    legal_mask: int


def apply(commands: Sequence[Command], board: np.ndarray | None = None) -> ReplayResult:
    """把命令序列作用到棋盘上，返回最终局面。

    应用不了就抛 `RecordApplyError`：`move` 不合法、`set` 打在非空格、`set null` 打在空格，
    以及"这一步会把局面带出键表"——两个 2^17 合并出 2^18 是唯一的越界路径（ADR 0036 决策 6）。
    """
    current = (
        np.zeros(BOARD_CELLS, dtype=np.uint8) if board is None else np.array(board, dtype=np.uint8, copy=True)
    )
    if current.shape != (BOARD_CELLS,):
        raise ValueError(f"棋盘必须是长度为 {BOARD_CELLS} 的一维数组")
    total_score = 0
    move_count = 0
    for command in commands:
        if isinstance(command, Clear):
            current = np.zeros(BOARD_CELLS, dtype=np.uint8)
        elif isinstance(command, SetTile):
            _apply_set(current, command)
        elif isinstance(command, Move):
            total_score += _apply_move(current, command.action)
            move_count += 1
        else:
            raise TypeError(f"未知命令 {command!r}")
    return ReplayResult(
        board=current,
        total_score=total_score,
        move_count=move_count,
        max_exponent=int(current.max()),
        legal_mask=int(r2048_core.legal_mask(current)),
    )


def _apply_set(board: np.ndarray, command: SetTile) -> None:
    if not 0 <= command.index < BOARD_CELLS:
        raise RecordApplyError(f"格下标 {command.index} 超出棋盘")
    where = index_to_coordinate(command.index)
    occupied = int(board[command.index]) != 0
    if command.exponent is None:
        if not occupied:
            raise RecordApplyError(f"{where} 本来就是空的，不能清除")
        board[command.index] = 0
        return
    if command.exponent not in VALID_TILE_EXPONENTS:
        raise RecordApplyError(f"块值指数 {command.exponent} 不在键表内（1..{MAX_EXPONENT}）")
    if occupied:
        raise RecordApplyError(f"{where} 上已经有块，不能放置（要替换先清除）")
    board[command.index] = command.exponent


def _apply_move(board: np.ndarray, action: int) -> int:
    outcome = r2048_core.apply_move(board, action)
    if not bool(outcome["valid"]):
        raise RecordApplyError(f"{DIRECTIONS[action]} 在当前盘面上不合法")
    result = np.asarray(outcome["board"], dtype=np.uint8)
    if int(result.max()) > MAX_EXPONENT:
        raise RecordApplyError(
            f"{DIRECTIONS[action]} 会把局面带出键表（最大块超过 2^{MAX_EXPONENT}）："
            "这种局面没有任何记录能表示它，所以在写下来之前就拒绝（ADR 0036 决策 6）"
        )
    board[:] = result  # 原地覆盖：调用方持有的仍是同一个数组
    return int(outcome["score"])


# ---- 渲染 -----------------------------------------------------------------------


def render(commands: Iterable[Command], rules_version: int | None = None) -> str:
    """渲染成棋谱文本。这是规范形式：`parse(render(x)) == list(x)`，且文本逐字节稳定。"""
    rules = int(r2048_core.RULES_VERSION) if rules_version is None else int(rules_version)
    lines = [f"record_format {RECORD_FORMAT_VERSION}", f"rules_version {rules}"]
    for command in commands:
        if isinstance(command, Clear):
            lines.append("clear")
        elif isinstance(command, SetTile):
            value = EMPTY_TOKEN if command.exponent is None else _face_of(command.exponent)
            lines.append(f"set {value} {index_to_coordinate(command.index)}")
        elif isinstance(command, Move):
            lines.append(f"move {DIRECTIONS[command.action]}")
        else:
            raise TypeError(f"未知命令 {command!r}")
    return "\n".join(lines) + "\n"


def _face_of(exponent: int) -> str:
    face = EXPONENT_TO_FACE.get(exponent)
    if face is None:
        raise ValueError(f"指数 {exponent} 不在键表内，写不进棋谱（1..{MAX_EXPONENT}）")
    return face


def commands_for_board(board: np.ndarray) -> list[Command]:
    """把一个盘面表示成 `clear` + 若干 `set`。

    编辑器造出来的局面、以及"以这个局面新开一局"，都用它变成棋谱前缀——手摆的局面与
    真打出来的棋因此共用同一种记录。
    """
    cells = np.asarray(board, dtype=np.uint8).reshape(-1)
    if cells.shape != (BOARD_CELLS,):
        raise ValueError(f"棋盘必须是 {BOARD_CELLS} 格")
    commands: list[Command] = [Clear()]
    commands.extend(
        SetTile(index=index, exponent=int(exponent)) for index, exponent in enumerate(cells) if exponent
    )
    return commands


# ---- 文件 -----------------------------------------------------------------------


def load(path: str | Path) -> list[Command]:
    return parse(Path(path).read_text(encoding="utf-8"))


def save(path: str | Path, commands: Iterable[Command]) -> Path:
    return write_text_atomic(path, render(commands))
