"""棋谱格式：一个契约，四条必须守住的界线（规格见 ADR 0036）。

为什么单独一个文件：棋谱是**别人会拿走的文件**——人会手写它、脚本会生成它、复盘页会回放它。
所以边界一旦松掉，症状不是报错而是"读起来和走出来的不是同一盘棋"，那种错误不会自己暴露。
这里把四条界线钉死：

1. **往返逐字节稳定**：`render` 是规范形式，`parse(render(x)) == x` 且文本不变；
2. **语法错误与应用错误是两种异常**（决策 7 的两层）——应用不了必须失败，不能静默跳过；
3. **键表闭合**：越界值（2^18、非 2 的幂）在解析处拒绝；两个 2^17 合并出 2^18 在应用处拒绝；
4. **重放不需要随机性**：生成块在棋谱里是显式的 `set`，所以同一条棋谱重放两次必然同盘——
   这正是 `undo` 不需要保存随机数状态、以及复盘能只喂 `set` 的前提。

第 5 条是跨实现的：`move` 的结果必须与 `reference/rules_reference.py` 逐条一致，
否则"棋谱"和"规则"就是两套语义。
"""

from __future__ import annotations

import numpy as np
import pytest
import r2048_core

from r2048 import records
from r2048.reference.rules_reference import move as reference_move

CANONICAL = """\
record_format 1
rules_version 1
clear
set 2 a1
set 4 b3
move left
set 2 c2
move up
"""


def _rules_version() -> int:
    return int(r2048_core.RULES_VERSION)


# ---- 1. 往返 ---------------------------------------------------------------------


def test_canonical_text_round_trips_byte_exactly() -> None:
    commands = records.parse(CANONICAL)
    assert records.render(commands) == CANONICAL


def test_comments_and_blank_lines_are_ignored() -> None:
    text = f"# 一局示范\n\nrecord_format 1\nrules_version {_rules_version()}\n\nclear\n# 摆棋\nset 2 a1\n"
    assert records.parse(text) == [records.Clear(), records.SetTile(index=0, exponent=1)]


def test_empty_value_written_as_null_only() -> None:
    """空格在棋谱里只有 `null` 一种写法：不给它数字写法，否则同一条命令有两种拼法。"""
    commands = [records.SetTile(index=0, exponent=None)]
    text = records.render(commands)
    assert "set null a1" in text
    assert records.parse(text) == commands


# ---- 2. 坐标 ---------------------------------------------------------------------


def test_coordinate_convention_is_top_left_row_major() -> None:
    """`a1` 是左上角、字母为列、数字为行——与 GTP 的行序相反，所以这是契约不是默契。"""
    assert records.coordinate_to_index("a1") == 0
    assert records.coordinate_to_index("d1") == 3
    assert records.coordinate_to_index("a4") == 12
    assert records.coordinate_to_index("b3") == 9  # 第 3 行第 2 列 → (3-1)*4+1
    assert records.coordinate_to_index("d4") == 15
    for index in range(16):
        assert records.coordinate_to_index(records.index_to_coordinate(index)) == index


@pytest.mark.parametrize("bad", ["a0", "a5", "e1", "aa", "1a", "", "a"])
def test_bad_coordinates_are_refused(bad: str) -> None:
    with pytest.raises(records.RecordSyntaxError):
        records.coordinate_to_index(bad)


# ---- 3. 键表 ---------------------------------------------------------------------


def test_key_table_covers_null_and_2_pow_1_to_17() -> None:
    assert records.MAX_FACE == "131072"
    for exponent in range(1, records.MAX_EXPONENT + 1):
        assert records.FACE_TO_EXPONENT[str(1 << exponent)] == exponent
    # 面值与内部指数是两套写法：2048 的面值是 2048，指数是 11。
    assert records.FACE_TO_EXPONENT["2048"] == 11


@pytest.mark.parametrize("bad", ["1", "0", "3", "6", "262144", "2048.0", "-2", "2^11"])
def test_values_outside_the_table_are_refused(bad: str) -> None:
    with pytest.raises(records.RecordSyntaxError):
        records.parse(f"record_format 1\nrules_version {_rules_version()}\nset {bad} a1\n")


# ---- 4. 应用：合法性与越界 --------------------------------------------------------


def _parse_body(body: str) -> list[records.Command]:
    return records.parse(f"record_format 1\nrules_version {_rules_version()}\n{body}")


def test_replay_is_randomness_free() -> None:
    """同一条棋谱重放两次必然同盘：生成块是显式的 `set`，重放不碰任何随机流。"""
    commands = _parse_body("clear\nset 2 a1\nset 4 b3\nmove left\nset 2 c2\n")
    first = records.apply(commands)
    second = records.apply(commands)
    assert np.array_equal(first.board, second.board)
    # 最后那个 set 就是"生成块"：它写在棋谱里，所以它不依赖种子。
    assert first.board[records.coordinate_to_index("c2")] == 1


def test_spawn_is_not_implied_by_move() -> None:
    """`move` 不隐含生成块——这正是记录必须显式写下它的原因。"""
    without_spawn = records.apply(_parse_body("clear\nset 2 a1\nset 4 b3\nmove left\n"))
    assert int((without_spawn.board != 0).sum()) == 2  # 还是那两个块，没有任何新块


def test_illegal_move_is_an_apply_error() -> None:
    with pytest.raises(records.RecordApplyError, match="left"):
        records.apply(_parse_body("clear\nset 2 a1\nmove left\n"))


def test_set_on_occupied_and_clear_on_empty_are_apply_errors() -> None:
    with pytest.raises(records.RecordApplyError, match="已经有块"):
        records.apply(_parse_body("clear\nset 2 a1\nset 4 a1\n"))
    with pytest.raises(records.RecordApplyError, match="本来就是空"):
        records.apply(_parse_body("clear\nset null a1\n"))


def test_merging_past_the_key_table_is_refused() -> None:
    """两个 2^17 合并出 2^18 是唯一能越界的路径；拒绝它之后键表在一切操作下闭合。"""
    body = f"clear\nset {records.MAX_FACE} a1\nset {records.MAX_FACE} b1\nmove left\n"
    with pytest.raises(records.RecordApplyError, match="带出键表"):
        records.apply(_parse_body(body))
    # 对照组：两个 2^16 合并到 2^17 是键表之内的，必须放行。
    allowed = records.apply(_parse_body("clear\nset 65536 a1\nset 65536 b1\nmove left\n"))
    assert allowed.max_exponent == records.MAX_EXPONENT


def test_syntax_error_and_apply_error_are_different_types() -> None:
    """两层必须可区分：语法错误在解析处就暴露，应用错误只有拿着棋盘才知道。"""
    assert issubclass(records.RecordSyntaxError, records.RecordError)
    assert issubclass(records.RecordApplyError, records.RecordError)
    assert records.RecordSyntaxError is not records.RecordApplyError
    with pytest.raises(records.RecordSyntaxError):
        records.parse(f"record_format 1\nrules_version {_rules_version()}\nwobble\n")
    with pytest.raises(records.RecordApplyError):
        records.apply([records.Move(action=records.ACTION_OF_DIRECTION["left"])])


# ---- 5. 文件头与版本闸门 ----------------------------------------------------------


def test_header_is_required() -> None:
    for text in ["clear\n", "set 2 a1\n", f"rules_version {_rules_version()}\nclear\n"]:
        with pytest.raises(records.RecordSyntaxError):
            records.parse(text)


def test_unknown_format_version_is_refused() -> None:
    with pytest.raises(records.RecordSyntaxError, match="格式版本"):
        records.parse(f"record_format 99\nrules_version {_rules_version()}\nclear\n")


def test_mismatched_rules_version_is_refused() -> None:
    """用错语义解释一份记录比读不出来危险——所以版本不匹配直接拒收（ADR 0007/0036）。"""
    wrong = _rules_version() + 1
    with pytest.raises(records.RecordSyntaxError, match="规则版本"):
        records.parse(f"record_format 1\nrules_version {wrong}\nclear\n")


def test_header_after_commands_is_not_a_header() -> None:
    with pytest.raises(records.RecordSyntaxError, match="未知命令"):
        records.parse(f"record_format 1\nrules_version {_rules_version()}\nclear\nrecord_format 2\n")


# ---- 6. 与规则参考实现对照（跨实现判据）------------------------------------------


def test_move_matches_the_rules_reference() -> None:
    """`apply` 走的是 C++ `apply_move`；这里用**不共享任何代码**的参考实现当判据。

    参考实现在 Python 侧只被测试使用（见 reference/__init__.py），它存在的理由正是：
    同一套查表算出来的期望值发现不了查表本身的错误。
    """
    rng = np.random.default_rng(20261008)
    checked = 0
    for _ in range(300):
        board = rng.integers(0, 9, size=records.BOARD_CELLS).astype(np.uint8)
        for action, name in enumerate(records.DIRECTIONS):
            expected = reference_move(tuple(int(cell) for cell in board), action)
            if not expected.valid:
                with pytest.raises(records.RecordApplyError):
                    records.apply([records.Move(action=action)], board=board)
                continue
            result = records.apply([records.Move(action=action)], board=board)
            assert tuple(int(cell) for cell in result.board) == expected.board, name
            assert result.total_score == expected.score, name
            checked += 1
    assert checked > 100  # 随机局面里合法移动占多数；少于这个数说明用例构造错了


# ---- 7. 编辑器接口 ---------------------------------------------------------------


def test_commands_for_board_round_trips() -> None:
    board = np.zeros(records.BOARD_CELLS, dtype=np.uint8)
    board[0], board[9], board[15] = 1, 5, records.MAX_EXPONENT
    commands = records.commands_for_board(board)
    assert commands[0] == records.Clear()
    assert np.array_equal(records.apply(commands).board, board)


def test_clear_resets_mid_record() -> None:
    result = records.apply(_parse_body("clear\nset 2 a1\nclear\nset 4 b2\n"))
    assert result.board.sum() == 2
    assert result.board[records.coordinate_to_index("a1")] == 0
    assert result.board[records.coordinate_to_index("b2")] == 2
