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
from pathlib import Path

import numpy as np
import pytest

from r2048 import records
from r2048.session import Session, SessionConfig
from r2048.session_io import SessionClient, session_argv
from r2048.webui import inventory
from r2048.webui.app import render_analysis, render_board, render_status


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
    html = render_board(board, legal_mask=0, spawn_index=5)
    for label in ("a", "b", "c", "d", "1", "2", "3", "4"):
        assert f">{label}<" in html, f"棋盘必须显示坐标 {label}"
    # 显示的是**面值**（2048），不是内部指数（11）——棋谱与界面都用人能读的那种写法。
    assert ">2048<" in html and ">2<" in html
    assert ">11<" not in html


def test_board_marks_the_engine_choice() -> None:
    board = np.zeros(16, dtype=np.uint8)
    board[0] = 1
    html = render_board(board, legal_mask=6, best_action=records.ACTION_OF_DIRECTION["right"])
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
    save_checkpoint(str(run / "model.pt"), model, model.metadata("m1"))

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


def test_ui_command_flow_editor_analyze_move_save_replay(checkpoint: str, tmp_path: Path) -> None:
    """把三个标签页会发的命令在真会话上跑一遍。

    编辑器 = `clear` + `set ...`；分析 = `analyze`；落子 = `move <方向>`；
    存档 = `save`；复盘 = 读回棋谱并逐前缀重放。这条路径不依赖任何浏览器组件。
    """
    client = SessionClient(session_argv(checkpoint=checkpoint, device="cpu", simulations=8))
    try:
        client.request("set_simulations 8")
        # 编辑器：摆一个能走几步的局面
        client.request("clear")
        for face, coordinate in (("2", "a1"), ("4", "c3"), ("8", "b2")):
            payload = client.request(f"set {face} {coordinate}")[0]["state"]
        assert payload["board"][records.coordinate_to_index("c3")] == 2  # 4 → 指数 2

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

        # 存档，然后读回来逐前缀重放
        path = tmp_path / "ui.txt"
        client.request(f"save {path}")
        commands = records.load(path)
        assert isinstance(commands[-1], records.SetTile), "生成块必须是棋谱里的显式 set"
        final = records.apply(commands)
        assert final.move_count == 1

        # 复盘滑块：从空到终局的每一个前缀都必须能重放（这正是不需要随机数状态的理由）
        for upto in range(len(commands) + 1):
            records.apply(commands[:upto])
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
