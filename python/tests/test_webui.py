"""WebUI：渲染、清单层，以及界面实际发出的那条命令流。

界面本身不做棋规、不选招，所以值得测的是三件事：

1. **渲染**：棋盘必须带坐标（a..d 是列、1..4 是行）——用户要能对着棋谱文字数格子；
   面板上"策略头先验"与"引擎选招"必须分开标（项目从不让探针 argmax，选招口径只有引擎一处）。
2. **清单**：模型一律从输出根扫出来，界面里不手输路径。坏掉的棋谱也要能被列出来（标成读不出来），
   而不是从列表里静静消失。
3. **界面发出的命令流**：编辑器摆局面 = `clear` + `set ...`；落子 = `move`；存档 = `save`。
   这些命令在真实会话上跑一遍，就等于把"点按钮"这件事验过了——不需要浏览器。

要 gradio 的那条（真的把 Blocks 搭起来）在缺依赖时跳过，好让 `make test` 在没装可选组时仍然是绿的。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from r2048 import records
from r2048.session import Session, SessionConfig
from r2048.session_io import SessionClient, session_argv
from r2048.webui import inventory
from r2048.webui.app import AppState, render_analysis, render_board, render_status


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> str:
    from r2048.nn.model import build, save_checkpoint

    path = tmp_path_factory.mktemp("webui") / "tiny.pt"
    model = build("sh-s6c64", {"channels": 8, "blocks": 1, "head_width": 16, "groups": 2})
    save_checkpoint(str(path), model, model.metadata("webui-tiny"))
    return str(path)


# ---- 渲染 -----------------------------------------------------------------------


def test_board_shows_coordinates_and_face_values() -> None:
    board = np.zeros(16, dtype=np.uint8)
    board[0] = 1  # a1 = 2
    board[15] = 11  # d4 = 2048
    html = render_board(board, spawn_index=5)
    for label in ("a", "b", "c", "d", "1", "2", "3", "4"):
        assert f">{label}<" in html, f"棋盘必须显示坐标 {label}"
    # 显示的是**面值**（2048），不是内部指数（11）——棋谱与界面都用人能读的那种写法。
    assert ">2048<" in html and ">2<" in html
    assert ">11<" not in html


def test_board_is_a_real_tile_component() -> None:
    """棋盘是**组件**，不是"能看就行"的表格：方块有自己的类与标准配色，布局靠网格。

    这条挡的是样式回退——上一版用裸 `<table>` 加内联宽度，格子会被容器拉变形，
    而变形了也没人会发现。
    """
    board = np.zeros(16, dtype=np.uint8)
    board[0] = 1  # 2  → 米色
    board[5] = 6  # 64 → 橙红
    html = render_board(board, spawn_index=5)
    assert "r2048-board" in html and "r2048-tile" in html and "r2048-label" in html
    # 底色与字色**必须成对**地断言：只查"颜色出现过"是不够的——把两者对调照样能通过，
    # 而那正是"配色写反了"的形态（2 会变成深底浅字，看起来只是丑，不会报错）。
    assert "background:#eee4da;color:#776e65" in html, "2 的方块：米色底 + 深色字"
    assert "background:#f65e3b;color:#f9f6f2" in html, "64 的方块：橙红底 + 白字"
    assert "r2048-spawn" in html, "生成块要能被标出来"
    assert "grid-template-columns" in html, "布局靠网格，不靠表格列宽"


def test_board_scales_to_its_container() -> None:
    """棋盘宽度必须**跟着容器走**，不能由内容决定。

    上一版是"固定格子尺寸 + width:max-content"：父容器的宽度是 flex 算出来的、猜不准，
    只要比棋盘窄一点点就直接溢出成一条横向滚动条——用户实测截图里就是那样（棋盘被挤成
    一条缝，底下一条滚动条）。这条钉住三件事：宽度取 100%、列宽用 1fr、方块用 aspect-ratio
    保持正方形。只断言"棋盘画出来了"是拦不住这次的回归的。
    """
    html = render_board(np.zeros(16, dtype=np.uint8))
    assert "width: 100%" in html, "棋盘宽度要跟着容器，不能由内容决定"
    assert "repeat(4, 1fr)" in html, "四列要等分容器宽度"
    assert "aspect-ratio: 1" in html, "方块要靠 aspect-ratio 保持正方形"
    # 断言的是**声明**而不是出现过的字符串：注释里正解释着这条规则为什么被换掉。
    assert "width: max-content" not in html, "width:max-content 正是溢出成滚动条的那条规则"


def test_board_marks_the_engine_choice() -> None:
    board = np.zeros(16, dtype=np.uint8)
    board[0] = 1
    html = render_board(board, best_action=records.ACTION_OF_DIRECTION["right"])
    assert "引擎选招" in html and "右" in html


def test_analysis_labels_prior_and_engine_choice_separately() -> None:
    payload = {
        "strategy": "search",
        "simulations": 8,
        "legal_mask": 6,
        "best_action": 1,
        "root_value": 12.5,
        "seconds": 0.01,
        "protocol": {"strategy": "search", "note": "交互分析口径：批组成与评测不同"},
        "directions": [
            {
                "name": name,
                "action": action,
                "legal": True,
                "visits": 4 if action == 1 else 0,
                "q_mean": 10.0,
                "one_ply": 8.0,
                "prior": 0.25,
            }
            for action, name in enumerate(records.DIRECTIONS)
        ],
    }
    html = render_analysis(payload)
    assert "策略头先验" in html, "先验必须被标成策略头的倾向，而不是引擎的选择"
    assert "引擎选招" in html
    assert "批组成与评测不同" in html, "口径差异必须显式出现，否则看的人会以为跟评测是同一个数"


def test_analysis_context_and_narrow_table_are_explicit() -> None:
    payload = {
        "best_action": 0,
        "simulations": 8,
        "seconds": 0.1,
        "root_value": 1,
        "protocol": {"strategy": "search"},
        "directions": [],
    }
    rendered = render_analysis(payload, context="上一步决策")
    assert "上一步决策的搜索结果" in rendered
    assert "r2048-analysis-scroll" in rendered and "aria-label='方向搜索分析'" in rendered
    assert "Q（含立即分）" in rendered and "策略头先验" in rendered


def test_attach_recreates_session_when_device_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    from r2048.webui import app as webui_app

    created = []

    class FakeClient:
        def __init__(self, argv) -> None:
            self.argv = argv
            self.closed = False
            created.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(webui_app, "SessionClient", FakeClient)
    monkeypatch.setattr(webui_app, "session_argv", lambda **kwargs: kwargs)
    state = AppState()
    state.attach("model.pt", "cpu", "search", 64, 0)
    original = state.client
    state.attach("model.pt", "cpu", "search", 64, 0)
    assert state.client is original and len(created) == 1

    state.attach("model.pt", "cuda", "search", 64, 0)
    assert original.closed
    assert state.client is not original and state.device == "cuda"
    previous = state.client
    state.attach("other-model.pt", "cuda", "search", 64, 0)
    assert previous.closed and state.checkpoint == "other-model.pt"
    state.close()


def test_analysis_without_result_is_not_an_error() -> None:
    assert "还没有分析结果" in render_analysis(None)
    assert "会话未连接" in render_status(None)
    assert "终局" in render_status(
        {"score": 1, "moves": 2, "max_face": 4, "terminal": True, "model_id": "m", "device": "cpu"}
    )


# ---- 清单 -----------------------------------------------------------------------


def _fake_root(tmp_path: Path) -> Path:
    """造一个"输出根"：一个模型、一批数据、两份棋谱（其中一份是坏的）。"""
    from r2048.nn.model import build, save_checkpoint

    root = tmp_path / "output"
    (root / "data" / "batch-A").mkdir(parents=True)
    (root / "data" / "batch-A" / "manifest.json").write_text(
        json.dumps(
            {
                "batch_id": "batch-A",
                "policy": "nn",
                "simulations": 64,
                "samples": 1000,
                "episodes": 10,
                "state": "complete",
                "model_id": "m1",
            }
        )
    )
    run = root / "runs" / "some-run"
    run.mkdir(parents=True)
    model = build("sh-s6c64", {"channels": 8, "blocks": 1, "head_width": 16, "groups": 2})
    metadata = model.metadata("m1")
    # 模型定义本身不带 training_target（那是训练阶段写进去的），但清单层要靠它判断"完整模型"。
    metadata["training_target"] = "dual"
    save_checkpoint(str(run / "model.pt"), model, metadata)

    records_dir = root / "records"
    records_dir.mkdir()
    session = Session(SessionConfig(checkpoint="", simulations=4))
    session.set_tile(0, 1)
    session.set_tile(5, 2)
    session.save(records_dir / "good.txt")
    (records_dir / "broken.txt").write_text("record_format 1\nrules_version 1\nwobble\n")
    return root


def test_inventory_finds_models_by_metadata(tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    models = inventory.list_models(root)
    assert [entry.model_id for entry in models] == ["m1"]
    assert models[0].architecture.startswith("sh-"), "架构要跟着显示出来，否则同名模型分不清"
    assert "model.pt" in models[0].label()
    found = inventory.find_model(models[0].label(), root)
    assert found is not None and found.path == models[0].path


def test_inventory_labels_carry_the_run_and_hide_resume_points(tmp_path: Path) -> None:
    """下拉里的每一行必须**认得出来**，而且不能一半是续跑点。

    输出根里同名 checkpoint 到处都是（每个 run 下都有 `value/model.pt`、`policy-head/model.pt`），
    只写"父目录/文件名"会得到一列谁也分不清的 `value/model.pt`——这正是这一页最初的形态。
    相对输出根的路径把 run 身份带回来，于是相邻行共享前缀、读起来成组。
    """
    root = _fake_root(tmp_path)
    inner = root / "runs" / "some-run" / "value"
    inner.mkdir()
    shutil.copy(root / "runs" / "some-run" / "model.pt", inner / "model.pt")
    shutil.copy(root / "runs" / "some-run" / "model.pt", inner / "last.pt")

    labels = [entry.label() for entry in inventory.list_models(root)]
    assert labels == sorted(labels)
    assert any(label.startswith("runs/some-run/model.pt") for label in labels)
    # 同名文件必须靠相对路径区分（这正是"读不出来"的那个 bug）。
    assert any(label.startswith("runs/some-run/value/model.pt") for label in labels)
    assert len(set(labels)) == len(labels), "标签不能撞车（撞车就等于选不中）"
    assert not any("last.pt" in label for label in labels), "续跑点默认不列出来"

    with_resume = [entry.label() for entry in inventory.list_models(root, include_resume=True)]
    assert any(label.startswith("runs/some-run/value/last.pt") for label in with_resume), (
        "要的时候必须能列出来"
    )


def test_default_model_prefers_a_complete_one(tmp_path: Path) -> None:
    """默认值往往就是用户直接点的那一个，所以它得是**完整模型**而不是恰好排在前面的半模型。"""
    from r2048.nn.model import build, save_checkpoint

    root = _fake_root(tmp_path)
    half = root / "runs" / "aaa-first" / "value"
    half.mkdir(parents=True)
    model = build("sh-s6c64", {"channels": 8, "blocks": 1, "head_width": 16, "groups": 2})
    metadata = model.metadata("half")
    metadata["training_target"] = "value"  # 只训了价值的半模型
    save_checkpoint(str(half / "model.pt"), model, metadata)

    entries = inventory.list_models(root)
    assert entries[0].label().startswith("runs/aaa-first"), "列表按路径排，半模型确实排在最前"
    assert entries[0].training_target == "value"
    assert inventory.default_model(entries).endswith("[dual]"), "默认选中要挑完整模型"


def test_inventory_reads_batches_and_records(tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    batches = inventory.list_batches(root)
    assert len(batches) == 1 and batches[0].batch_id == "batch-A" and batches[0].samples == 1000

    found = {entry.path.name: entry for entry in inventory.list_records(root)}
    assert found["good.txt"].moves == 0, "只摆子没走过步，步数是 0"
    # 坏棋谱不能被静默丢掉：它得出现在列表里，且步数标成"读不出来"。
    assert found["broken.txt"].moves == -1


def test_inventory_on_a_missing_root_is_empty_not_an_error(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    assert inventory.list_models(missing) == []
    assert inventory.list_batches(missing) == []


# ---- 界面实际发出的命令流（无浏览器）--------------------------------------------


def test_ui_command_flow_newgame_analyze_move_save_replay(checkpoint: str, tmp_path: Path) -> None:
    """把三个标签页会发的命令在真会话上跑一遍。

    新开一局 = `newgame`；分析 = `analyze`；落子 = `genmove`/`move <方向>`；存档 = `save`；
    复盘 = 读回棋谱并逐前缀重放。这条路径不依赖任何浏览器组件，所以没有浏览器也能验。
    """
    client = SessionClient(session_argv(checkpoint=checkpoint, device="cpu", simulations=8))
    try:
        client.request("set_simulations 8")
        # 「新开一局」：这是界面上从零开始的那一步，不再需要先手工摆棋
        opened = client.request("newgame seed=4242")[0]
        assert opened["seed"] == 4242
        assert len([cell for cell in opened["state"]["board"] if cell]) == 2

        # 分析：必须回报四个方向与一个引擎选招
        report = client.request("analyze")[-1]
        assert report["is_during_search"] is False
        assert len(report["directions"]) == 4 and report["best_action"] >= 0
        assert report["protocol"]["lanes"] == 1

        # 落子（自动抓），生成块要被写进棋谱
        frames = client.request("genmove")
        step = frames[0]["step"]
        assert step["action"] == report["best_action"], "引擎走的那一步必须与分析面板一致"
        assert step["spawn_index"] >= 0

        # 玩家模式：按指定方向走一步（这条路径不经过搜索）
        frames = client.request("move down")
        assert frames[0]["step"]["action"] == records.ACTION_OF_DIRECTION["down"]

        # 存档，然后读回来逐前缀重放
        path = tmp_path / "ui.txt"
        client.request(f"save {path}")
        commands = records.load(path)
        assert isinstance(commands[0], records.Clear)
        assert isinstance(commands[-1], records.SetTile), "生成块必须是棋谱里的显式 set"
        final = records.apply(commands)
        assert final.move_count == 2, "genmove 与 move 各算一步"
        assert np.array_equal(final.board, records.apply(commands).board)

        # 复盘滑块：从空到终局的每一个前缀都必须能重放（这正是不需要随机数状态的理由）
        for upto in range(len(commands) + 1):
            records.apply(commands[:upto])

        # 编辑器：`clear` + `set ...` 也照样能用（手摆局面）
        assert client.request("clear")[0]["state"]["moves"] == 0
        payload = client.request("set 4 c3")[0]["state"]
        assert payload["board"][records.coordinate_to_index("c3")] == 2  # 4 → 指数 2
    finally:
        client.close()
    assert client.returncode == 0


def test_build_app_constructs_the_blocks() -> None:
    pytest.importorskip("gradio", reason="webui 是可选组件（ADR 0039）：uv sync --group webui")
    from r2048.webui.app import build_app

    demo, state = build_app()
    try:
        assert demo is not None
    finally:
        state.close()


# ---- 界面接线（Gradio 组件层）-----------------------------------------------------
#
# 这一组防的是"构造时不报、一点就崩"的那类问题。Gradio 的坑很集中：
#   ① 处理函数返回的个数与 `outputs` 申明的个数不一致；
#   ② 生成器里用 `return` 交回视图——返回值被丢掉，界面表现为"点了没反应"。
# 两条都不会在 `build_app()` 时暴露，只有真跑一次处理函数才会。


def _dummy(component) -> object:
    """给一个控件造一个"看起来合理"的输入值，好让处理函数走到它该走的分支。"""
    import gradio as gr

    label = getattr(component, "label", "") or ""
    if label.startswith(("模型", "棋谱")):
        return None  # 不给模型/棋谱：这一组测试不该真的拉起会话子进程
    if isinstance(component, gr.Dropdown):
        if not component.choices:
            return None
        # Gradio 把 choices 归一成 (label, value) 对，这里要取**值**而不是整个对。
        first = component.choices[0]
        return first[1] if isinstance(first, (tuple, list)) else first
    if isinstance(component, (gr.Number, gr.Slider)):
        return 0
    if isinstance(component, gr.Checkbox):
        return False
    if isinstance(component, gr.Textbox):
        return ""
    return None


def _fresh_app():
    from r2048.webui.app import AppState, build_app

    return build_app(AppState())


def test_every_handler_matches_its_declared_outputs() -> None:
    pytest.importorskip("gradio", reason="webui 是可选组件（ADR 0039）")
    import inspect

    demo, state = _fresh_app()
    try:
        checked = 0
        for descriptor in demo.fns.values():
            name = descriptor.fn.__name__
            result = descriptor.fn(*[_dummy(component) for component in descriptor.inputs])
            if inspect.isgenerator(result):
                result = next(result, None)
            if result is None:  # 生成器一次都没产出：上面那条"生成器不许 return"就属于这种情况
                pytest.fail(f"{name} 是生成器但第一帧没有产出任何东西")
            declared = len(descriptor.outputs)
            if isinstance(result, (tuple, list)):
                assert len(result) == declared, (
                    f"{name} 返回了 {len(result)} 个值，但申明了 {declared} 个 outputs"
                )
            else:
                # Gradio 也接受"单个值对应单个 output"（`gr.update(...)` 是 dict，不是序列）。
                assert declared == 1, (
                    f"{name} 只返回了一个 {type(result).__name__}，但申明了 {declared} 个 outputs"
                )
            checked += 1
        assert checked >= 15, f"只核对了 {checked} 个处理函数，太少"
    finally:
        state.close()


def test_run_and_stop_share_one_slot() -> None:
    """运行与停止是**同一个位置的两个状态**，不同时存在；点下去才互换，手动控件同时置灰。"""
    pytest.importorskip("gradio", reason="webui 是可选组件（ADR 0039）")

    demo, state = _fresh_app()
    try:
        state.last_state = {
            "board": [1, 0, 0, 0] + [0] * 12,
            "model_id": "test",
            "device": "cpu",
            "score": 0,
            "moves": 0,
            "max_face": 2,
            "seed": 1,
            "empty": False,
            "terminal": False,
        }
        buttons = [b for b in demo.blocks.values() if type(b).__name__ == "Button"]
        start = next(b for b in buttons if str(b.value).startswith("▶"))
        stop = next(b for b in buttons if str(b.value).startswith("■"))
        assert start.visible is True and stop.visible is False, "初始只该看到「自动执行」"
        stop_handler = next(d for d in demo.fns.values() if d.fn.__name__ == "on_auto_stop")
        assert stop_handler.queue is False, "停止不能等待自动执行的生成器退出"

        descriptor = next(d for d in demo.fns.values() if d.fn.__name__ == "on_auto_start")
        first = next(descriptor.fn(*[_dummy(component) for component in descriptor.inputs]))
        assert first[4]["visible"] is False, "开始执行后「自动执行」要藏起来"
        assert first[5]["visible"] is True, "并露出「停止」"
        assert all(update["interactive"] is False for update in first[6:]), "自动执行期间手动控件要置灰"
    finally:
        state.close()


def test_editor_preview_and_failed_replay_do_not_keep_stale_positions() -> None:
    pytest.importorskip("gradio", reason="webui 是可选组件（ADR 0039）")
    demo, state = _fresh_app()
    try:
        handlers = {descriptor.fn.__name__: descriptor.fn for descriptor in demo.fns.values()}
        rendered = handlers["on_editor_preview"]("2048", *[records.EMPTY_TOKEN] * 15)
        assert ">2048<" in rendered and int(state.editor[0]) == 11
        assert "<style>" not in rendered, "实时预览不能每次重新注入全局样式"
        state.replay = [records.Clear()]
        state.replay_upto = 1
        slider, _, message = handlers["on_replay_load"](None)
        assert not state.replay and state.replay_upto == 0
        assert slider["interactive"] is False and 'role="alert"' in message
        exported = handlers["on_export"]()
        assert "请先载入棋谱" in exported[-1]
    finally:
        state.close()


def test_auto_execution_rejects_empty_or_terminal_positions() -> None:
    pytest.importorskip("gradio", reason="webui 是可选组件（ADR 0039）")
    demo, state = _fresh_app()
    try:
        descriptor = next(d for d in demo.fns.values() if d.fn.__name__ == "on_auto_start")
        values = [_dummy(component) for component in descriptor.inputs]
        for condition, message in (("empty", "盘面为空"), ("terminal", "已终局")):
            state.last_state = {
                "board": [0] * 16 if condition == "empty" else [1] * 16,
                "model_id": "test",
                "device": "cpu",
                "score": 0,
                "moves": 0,
                "max_face": 2,
                "seed": 1,
                "empty": condition == "empty",
                "terminal": condition == "terminal",
            }
            frame = next(descriptor.fn(*values))
            assert message in frame[3]
            assert state.auto_running is False
    finally:
        state.close()
