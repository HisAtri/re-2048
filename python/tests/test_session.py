"""会话：状态机、文本协议、以及"棋谱是唯一事实源"这条不变量。

为什么单独一个文件：会话是界面唯一会碰的那一层，它一旦把"记录的棋"和"屏幕上的棋"弄成
两回事，复盘时就会给出一个看不出来的错误答案（而且不报错）。所以这里钉三件事：

1. **盘面永远等于重放棋谱的结果**——编辑、落子、undo、载入都只改棋谱，状态由重放算出；
2. **`undo` 不需要随机数状态**：生成块在棋谱里是显式的 `set`（ADR 0036），所以退回上一个
   `move` 只需截断再重放；
3. **协议与棋谱共用同一套词汇与值域**：`set` 的值域与坐标校验直接走 `records` 的解析器，
   不会出现"能写进棋谱但不能通过协议摆"这种两套规则。

需要模型的那几条用一个随机初始化的小 checkpoint（`sh-s6c64` 的极小配置），跑在 CPU 上，
所以整文件是秒级的，不进 GPU。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import r2048_core

from r2048 import records
from r2048.session import AnalysisReport, Session, SessionConfig
from r2048.session_io import SessionClient, SessionError, SessionServer, _split_response, session_argv


def _tiny_checkpoint(path: Path) -> str:
    """一个极小但真实可加载的 checkpoint：CPU 上跑几十次模拟只要毫秒级。"""
    from r2048.nn.model import save_checkpoint
    from r2048.nn.registry import build

    model = build("sh-s6c64", {"channels": 8, "blocks": 1, "head_width": 16, "groups": 2})
    save_checkpoint(str(path), model, model.metadata("tiny-test"))
    return str(path)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> str:
    return _tiny_checkpoint(tmp_path_factory.mktemp("session") / "tiny.pt")


def _session(checkpoint: str = "", **overrides) -> Session:
    config = SessionConfig(checkpoint=checkpoint, device="cpu", simulations=8, cache_entries=1 << 10)
    for name, value in overrides.items():
        setattr(config, name, value)
    return Session(config)


def _position(session: Session) -> None:
    """摆一个可走几步的盘面（不用开局生成器：会话的起点由调用方给）。"""
    session.set_tile(records.coordinate_to_index("a1"), 1)
    session.set_tile(records.coordinate_to_index("b2"), 2)
    session.set_tile(records.coordinate_to_index("c3"), 3)


def _response(lines: list[str]) -> tuple[str, dict | None]:
    """把一次应答拆成（标记, payload）。只有 `=` 的 body 是 JSON，`?` 的是人话。"""
    assert lines and lines[-1] == "", f"应答必须以空行结束，收到 {lines!r}"
    body_lines = [line for line in lines if line.strip()]
    assert len(body_lines) == 1, f"非流式命令应当只回一行内容，收到 {lines!r}"
    mark, body = _split_response(body_lines[0])
    return mark, (json.loads(body) if mark == "=" and body.strip() else None)


def _run(server: SessionServer, command: str) -> tuple[str, dict | None]:
    lines: list[str] = []
    assert server.handle(command, lines.append) is True
    return _response(lines)


# ---- 1. 棋谱是唯一事实源 ----------------------------------------------------------


def test_editing_only_changes_the_record_and_state_follows() -> None:
    session = _session()
    session.set_tile(records.coordinate_to_index("a1"), 1)
    session.set_tile(records.coordinate_to_index("d4"), 5)
    assert session.state()["board"][0] == 1
    assert session.state()["board"][15] == 5
    # 不变量：盘面 == 重放棋谱。
    assert np.array_equal(session.board, records.apply(session.record).board)
    assert session.state()["moves"] == 0 and session.score == 0

    session.set_tile(records.coordinate_to_index("a1"), None)
    assert session.state()["board"][0] == 0
    assert np.array_equal(session.board, records.apply(session.record).board)


def test_bad_edit_does_not_touch_the_record() -> None:
    session = _session()
    session.set_tile(records.coordinate_to_index("a1"), 1)
    length = len(session.record)
    with pytest.raises(records.RecordApplyError):
        session.set_tile(records.coordinate_to_index("a1"), 2)  # 已有块，不能再放
    assert len(session.record) == length, "应用不了的命令不能留在棋谱里"


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    session = _session()
    _position(session)
    path = session.save(tmp_path / "game.txt")
    assert path.read_text().startswith("record_format 1\nrules_version")
    restored = _session()
    restored.load(path)
    assert np.array_equal(restored.board, session.board)
    assert restored.render() == session.render()


def test_undo_needs_no_random_state(checkpoint: str) -> None:
    """生成块写在棋谱里，所以退回一步只是截断再重放——不保存任何随机数状态。"""
    session = _session(checkpoint)
    _position(session)
    session.play(None)
    after_one = session.board.copy()
    assert session.state()["moves"] == 1
    # 棋谱里那一步由 Move + 生成块的 SetTile 组成（生成块是显式的，这正是关键）。
    assert isinstance(session.record[-1], records.SetTile)

    session.play(None)
    assert session.state()["moves"] == 2
    assert session.undo() is True
    assert np.array_equal(session.board, after_one), "undo 必须回到走第二步之前的盘面"
    assert session.state()["moves"] == 1
    assert session.undo() is True
    assert session.state()["moves"] == 0
    assert session.undo() is False, "没有可退的步时返回 False，而不是抛异常"


# ---- 2. 分析：任意局面 + 引擎同口径 ------------------------------------------------


def test_empty_board_analysis_says_what_to_do_not_a_cryptic_end(checkpoint: str) -> None:
    session = _session(checkpoint)
    with pytest.raises(RuntimeError, match="空"):
        session.analyze()


def test_analysis_reports_every_direction_and_agrees_with_the_engine(checkpoint: str) -> None:
    session = _session(checkpoint)
    _position(session)
    report = session.analyze()
    assert isinstance(report, AnalysisReport)
    assert report.legal_mask == session.legal_mask
    payload = report.payload()
    assert len(payload["directions"]) == 4
    for action, direction in enumerate(payload["directions"]):
        assert direction["name"] == records.DIRECTIONS[action]
        assert direction["legal"] == bool((session.legal_mask >> action) & 1)

    # 与引擎那条同源的不变量：恰好一条模拟用于求值根本身（见 ADR 0037）。
    assert sum(report.visits) == report.simulations - 1 == 8 - 1
    assert report.best_action >= 0
    # 分析口径必须跟着结果一起回给界面：会话的批组成与评测不同（ADR 0017）。
    assert report.protocol["lanes"] == 1 and report.protocol["kind"] == "session-analysis"

    # 紧接着落子必须走的正是面板显示的那一步——同一棵树，不该有两套口径。
    step = session.play(None)
    assert step["action"] == report.best_action


def test_greedy_strategies_analyze_without_searching(checkpoint: str) -> None:
    """策略/价值两档不做树搜索：它们的"分析"就是把根求值一次（预算 1）。

    也顺带钉住一件容易搞错的事：这两档的 `best_action` **不是**访问数最大（它们没有访问数），
    而是引擎自己的选招（价值档看 r+A、策略档看先验）——所以面板在两档下同样能显示"会走哪一步"。
    """
    for strategy in ("policy", "value"):
        session = _session(checkpoint, strategy=strategy)
        _position(session)
        report = session.analyze()
        assert report.simulations == 1, f"{strategy} 档不该跑树搜索"
        assert sum(report.visits) == 0, "没有搜索就没有访问数"
        assert report.best_action >= 0, f"{strategy} 档也要能说出引擎会走哪一步"
        assert session.play(None)["action"] == report.best_action


def test_play_with_an_explicit_direction_uses_the_engine_spawn(checkpoint: str) -> None:
    session = _session(checkpoint)
    _position(session)
    before = session.board.copy()
    action = records.ACTION_OF_DIRECTION["left"]
    expected = np.asarray(r2048_core.apply_move(before, action)["board"], dtype=np.uint8)

    step = session.play(action)
    assert step["action"] == action
    assert step["terminal"] is False
    assert session.state()["moves"] == 1
    # 生成块由引擎生成并回报，会话把它写进棋谱：除了那一格，盘面必须与 afterstate 完全一致。
    # 注意那一格在**走之前**可以是满的（合并会把格子让出来），所以要跟 afterstate 比。
    spawn = step["spawn_index"]
    assert 0 <= spawn < records.BOARD_CELLS
    assert expected[spawn] == 0, "生成块落在 afterstate 的空格上"
    assert step["spawn_exponent"] in (1, 2), "只会生成 2（指数 1）或 4（指数 2）"
    assert session.board[spawn] == step["spawn_exponent"]
    for index in range(records.BOARD_CELLS):
        if index != spawn:
            assert session.board[index] == expected[index]


def test_analysis_needs_a_model_but_editing_does_not() -> None:
    session = _session()
    session.set_tile(records.coordinate_to_index("a1"), 1)
    with pytest.raises(RuntimeError, match="checkpoint"):
        session.analyze()


# ---- 2b. 开局 -------------------------------------------------------------------


def test_new_game_starts_a_standard_opening() -> None:
    """`new_game` 走 C++ 的 opening_board——与跑谱同一个函数，不是 Python 另写一遍。

    棋谱写下来的仍然是显式的 `set`，于是"哪两枚块、在哪"被固定进文件：同 seed 必然同盘。
    """
    session = _session()
    assert session.new_game(20261009) == 20261009
    tiles = [int(cell) for cell in session.board if cell]
    assert len(tiles) == 2, "标准开局是两枚块"
    assert all(tile in (1, 2) for tile in tiles), "只会出 2（指数 1）或 4（指数 2）"
    assert session.state()["moves"] == 0 and session.score == 0
    assert session.state()["seed"] == 20261009
    assert session.legal_mask != 0, "开局必然有合法方向，否则没法玩"

    assert isinstance(session.record[0], records.Clear)
    spawns = [c for c in session.record if isinstance(c, records.SetTile)]
    assert len(spawns) == 2, "开局的每一枚块都是棋谱里的一条 set"
    assert records.apply(session.record).move_count == 0, "开局不是一步棋"

    same = _session()
    same.new_game(20261009)
    assert np.array_equal(same.board, session.board), "同 seed 必须同盘"


def test_new_game_is_playable_and_undoable(checkpoint: str) -> None:
    """开一局之后整条链路要能跑：分析 → 落子 → 撤销回开局。"""
    session = _session(checkpoint)
    session.new_game(7)
    assert session.analyze().best_action >= 0
    assert session.play(None)["action"] >= 0
    assert session.state()["moves"] == 1
    assert session.undo() is True
    assert session.state()["moves"] == 0
    assert np.array_equal(session.board, records.apply(session.record).board)


def test_new_game_without_a_seed_picks_one_and_records_it() -> None:
    session = _session()
    seed = session.new_game()
    assert seed > 0 and session.state()["seed"] == seed
    # 不给种子也必须能被重放：棋谱里是显式的 set，盘面由重放算出。
    assert np.array_equal(session.board, records.apply(session.record).board)


# ---- 2c. 规则正确性（回归）-------------------------------------------------------
#
# 这一组是**回归**：曾经有过两个很难看的 bug，都是"看起来正常、玩起来错"。
#
# ① 生成块固定在同一格、且永远只出 2。根因：会话每走一步都会重放并 `set_root`，而
#    `set_root` 会把引擎的步号清零；生成块的随机流是 `derive(种子, …, 步号, …)`，
#    步号恒为 0 → 每一步抽到同一个种子 → 同一格、同一个 90/10 结果。
# ② 终局的那一步被丢掉。根因：`play()` 见到 `terminal` 就整条 return，于是棋谱比引擎
#    少一步——界面显示的盘面落后一步却报"终局"，看起来就是"终局判定错了"。
#
# 两条都属于"不报错、只给出错误结果"，所以必须由测试钉住，而不是靠看一眼。


def _drive(session: Session, moves: int = 5000) -> list[tuple[int, int]]:
    """按固定顺序试四个方向落子，返回每步的（生成位置, 生成指数）。

    用**显式方向**，所以完全不经过搜索——纯 C++ 的提交路径，几百步也是毫秒级。

    `moves` 是**防御性的上限**：真正的终局根本用不到它（一局最多几千步），而一旦出现"走不完"
    的退化，它能让测试干净地失败。这不是假想——bug ②（终局那一步被丢掉）的表现就是**挂住**：
    会话的盘面永远停在终局之前，于是这个循环永远转下去（实测把 pytest 挂到超时）。
    """
    spawns: list[tuple[int, int]] = []
    while not session.terminal() and not session.empty() and len(spawns) < moves:
        for action in range(4):
            try:
                step = session.play(action)
            except RuntimeError:
                continue
            spawns.append((int(step["spawn_index"]), int(step["spawn_exponent"])))
            break
        else:
            break  # 四个方向都走不了：引擎已经报终局了，上面那个 while 会退出
    return spawns


def test_engine_step_advances_across_moves(checkpoint: str) -> None:
    """**这条直接钉住 bug ① 的机制**：落子之后不能把引擎的步号清零。

    生成块的随机流是 `derive(引擎种子, …, 步号, …)`；步号恒为 0 就意味着每一步抽到同一个
    种子。这条比"位置看起来分散"结实得多——我先前写的那版位置检查在 bug 放回去之后**照样
    通过**：`below(n)` 的返回值随空格数变化，所以"位置分散"能被骗过。断言步号本身骗不过去。
    """
    session = _session(checkpoint)
    session.new_game(4242)
    engine = session.require_engine()
    for expected in (1, 2, 3):
        for action in range(4):
            try:
                session.play(action)
                break
            except RuntimeError:
                continue
        else:
            pytest.fail("开局几步之内不该无路可走")
        assert engine.debug_state()["lanes"][0]["step"] == expected, (
            f"落子后引擎步号应当是 {expected}；被清零说明会话在落子后又 set_root 了一次"
        )


def test_spawn_values_vary_within_one_game(checkpoint: str) -> None:
    """90/10 要**在一局之内**逐步生效。

    判据是"**某一局单独**同时出现 2 与 4"而不是"几局合起来出现过两种"：步号被清零时，
    一整局抽到的是同一个种子，于是这一局的值恒定——可能是全 2、也可能是全 4。把几局的
    结果合起来看，恰好会凑齐两种值，那个 bug 就漏过去了（我第一版就是这么写的，实测漏过）。
    """
    session = _session(checkpoint)
    both_in_one_game = False
    samples = 0
    for seed in (1, 2, 3):
        session.new_game(seed)
        spawns = _drive(session)
        values = {exponent for _, exponent in spawns}
        samples += len(spawns)
        assert values <= {1, 2}, "只会出 2（指数 1）或 4（指数 2）"
        both_in_one_game = both_in_one_game or values == {1, 2}
    assert samples >= 150, f"样本量太小（{samples} 步），判不出来"
    assert both_in_one_game, (
        "没有任何一局同时出现过 2 与 4：说明随机流没有随步号推进（每步抽到同一个种子）"
    )


def test_terminal_move_is_written_to_the_record(checkpoint: str) -> None:
    """终局的那一步必须落在棋谱里——这条直接对着 bug ②。"""
    session = _session(checkpoint)
    session.new_game(20261009)
    _drive(session)

    assert session.terminal(), "这条用例要真的走到终局"
    assert not session.empty()
    assert session.legal_mask == 0
    # 棋谱与盘面必须一致，而且末条命令是那一步的生成块（说明 move 也写进去了）。
    replayed = records.apply(session.record)
    assert np.array_equal(replayed.board, session.board)
    assert replayed.move_count == session.moves
    assert isinstance(session.record[-1], records.SetTile), "最后落下的那枚生成块要在棋谱里"
    assert any(isinstance(c, records.Move) for c in session.record)
    assert session.score == replayed.total_score
    # 终局之后落子/分析要明确报终局，而不是静默走一步或给出空结果。
    with pytest.raises(RuntimeError, match="终局"):
        session.play(None)
    with pytest.raises(RuntimeError, match="终局"):
        session.analyze()


def test_undo_from_a_terminal_position_restores_a_playable_board(checkpoint: str) -> None:
    session = _session(checkpoint)
    session.new_game(20261009)
    _drive(session)
    assert session.terminal()
    assert session.undo() is True
    assert not session.terminal(), "退回一步之后必须又能走"
    assert session.legal_mask != 0
    assert np.array_equal(session.board, records.apply(session.record).board)


def test_empty_board_is_not_reported_as_terminal(checkpoint: str) -> None:
    """"空"与"终局"都没有合法方向，但下一步动作完全不同——不能混成一句话。"""
    session = _session(checkpoint)
    assert session.empty() and not session.terminal()
    state = session.state()
    assert state["empty"] is True and state["terminal"] is False
    with pytest.raises(RuntimeError, match="空"):
        session.analyze()


def test_same_seed_replays_the_same_game_exactly(checkpoint: str) -> None:
    """可复现没有被上面两个修复破坏：同 seed + 同操作序列 → 同一局、逐字节同棋谱。"""
    first = _session(checkpoint)
    first.new_game(7)
    spawns_first = _drive(first, moves=25)

    second = _session(checkpoint)
    second.new_game(7)
    spawns_second = _drive(second, moves=25)

    assert spawns_first == spawns_second
    assert np.array_equal(first.board, second.board)
    assert first.render() == second.render()
    assert first.score == second.score


# ---- 3. 协议 ---------------------------------------------------------------------


def test_protocol_echoes_ids_and_marks_errors(checkpoint: str) -> None:
    server = SessionServer(_session())
    assert _run(server, "7 protocol_version")[0] == "="
    lines: list[str] = []
    server.handle("12 name", lines.append)
    assert lines[0].startswith("12 = "), "带 id 的请求必须把 id 回显，才能把应答对上"
    mark, _ = _run(server, "wobble")
    assert mark == "?"
    mark, payload = _run(server, "list_commands")
    assert mark == "=" and "analyze" in payload["commands"] and "genmove" in payload["commands"]


def test_protocol_set_shares_the_record_grammar() -> None:
    """`set` 的值域与坐标直接走 records 的解析器：协议与棋谱不会有两套规则。"""
    server = SessionServer(_session())
    mark, payload = _run(server, "set 2048 b3")
    assert mark == "="
    board = payload["state"]["board"]
    assert board[records.coordinate_to_index("b3")] == 11  # 面值 2048 → 指数 11
    # 越界值与非法坐标都被拒；报错里点名了合法范围。
    mark, _ = _run(server, "set 262144 a1")
    assert mark == "?"
    mark, _ = _run(server, "set 3 a1")
    assert mark == "?"
    mark, _ = _run(server, "set 2 e9")
    assert mark == "?"


def test_protocol_commands_change_state_and_report_it() -> None:
    server = SessionServer(_session())
    _run(server, "set 2 a1")
    _run(server, "set 2 b1")
    mark, payload = _run(server, "clear")
    assert mark == "=" and sum(payload["state"]["board"]) == 0
    mark, payload = _run(server, "set null a1")  # 空位清除 → 报错
    assert mark == "?"
    _run(server, "set 2 a1")
    mark, payload = _run(server, "undo")
    assert mark == "=" and payload["undone"] is False, "没有走过步，undo 应当如实说没有"


def test_protocol_newgame_starts_an_opening() -> None:
    """`newgame` 让界面不用先手摆棋就能直接玩：开一局 → 分析 → 落子。"""
    server = SessionServer(_session())
    mark, payload = _run(server, "newgame seed=123")
    assert mark == "=" and payload["seed"] == 123
    board = payload["state"]["board"]
    assert len([cell for cell in board if cell]) == 2
    mark, again = _run(server, "newgame seed=123")
    assert again["state"]["board"] == board, "同一个 seed 必然同一局（棋谱里是显式的 set）"
    mark, _ = _run(server, "newgame seed=abc")
    assert mark == "?"
    mark, payload = _run(server, "newgame")  # 不给种子也要能用
    assert mark == "=" and payload["seed"] > 0


def test_analyze_streams_frames_and_ends_with_a_final_one(checkpoint: str) -> None:
    session = _session(checkpoint)
    _position(session)
    lines: list[str] = []
    SessionServer(session).handle("analyze simulations=8", lines.append)
    assert lines[-1] == "", "整条应答必须以空行结束"
    assert lines[0].strip() == "=", "先认领这次请求，再开始出帧"
    frames = [json.loads(_split_response(line)[1]) for line in lines[1:-1]]
    assert frames, "至少要有一帧"
    assert all(frame["is_during_search"] for frame in frames[:-1]), "中间帧都标记为搜索进行中"
    assert frames[-1]["is_during_search"] is False, "最后一帧才是终帧"
    assert frames[-1]["simulations"] == 8
    assert len(frames[-1]["directions"]) == 4


def test_analyze_accepts_one_shot_overrides(checkpoint: str) -> None:
    session = _session(checkpoint)
    _position(session)
    lines: list[str] = []
    SessionServer(session).handle("analyze simulations=4 strategy=value", lines.append)
    final = json.loads(_split_response(lines[-2])[1])
    assert final["strategy"] == "value" and final["simulations"] == 1
    assert session.config.strategy == "value"
    mark, _ = _run(SessionServer(session), "analyze nonsense=1")
    assert mark == "?"


# ---- 4. 真子进程冒烟 -------------------------------------------------------------


def test_session_process_speaks_the_protocol_over_pipes(tmp_path: Path) -> None:
    """真实管道驱动（没有模型的那个形态）：编辑与复盘必须完全独立于模型可用性。"""
    client = SessionClient(session_argv())
    try:
        frames = client.request("list_commands")
        assert "quit" in frames[0]["commands"]
        frames = client.request("set 4 c2")
        assert frames[0]["state"]["board"][records.coordinate_to_index("c2")] == 2
        assert client.request("state")[0]["state"]["moves"] == 0
        path = tmp_path / "out.txt"
        assert Path(client.request(f"save {path}")[0]["path"]).exists()
        # 没有模型：分析与落子必须回一条清楚的错误，而不是崩掉或给出空结果。
        with pytest.raises(SessionError, match="checkpoint"):
            client.request("analyze")
        with pytest.raises(SessionError, match="方向"):
            client.request("move sideways")
    finally:
        client.close()
    assert client.returncode == 0, "quit 之后会话进程应当干净退出"
