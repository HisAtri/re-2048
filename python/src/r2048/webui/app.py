"""WebUI：计算/分析、棋盘编辑器、复盘三个标签页。

界面**不做棋规、不选招**：它只把命令交给会话进程，再把会话回报的状态画出来。所有引擎调用
都经 `SessionClient`（文本协议 + 独立进程），所以 Gradio 的多线程回调不会碰到引擎——
引擎不是为并发独立搜索设计的（见 ADR 0038）。

自动执行的节奏按 ADR 0037 的约定：**间隔只是速率上限，搜索不被中断**。用 `gr.Timer` 定时
尝试走一步；拿不到锁（正在搜索）或还没到间隔就跳过这一次，于是既不会并发碰会话，
也不会把搜索砍断。
"""

from __future__ import annotations

import html
import threading
import time
from pathlib import Path

import numpy as np

from r2048 import paths, records
from r2048.session_io import SessionClient, SessionError, session_argv
from r2048.webui import inventory

MODES = ("只分析", "自动执行", "玩家")
STRATEGY_LABELS = {
    "搜索（NN-MCTS）": "search",
    "策略头（测试用）": "policy",
    "价值头（测试用）": "value",
}
DIRECTION_LABELS = {"up": "上", "right": "右", "down": "下", "left": "左"}
# 方块配色（指数 → 底色），照 2048 的常见配色，让"看盘面"不用读数字也能认出来。
TILE_COLORS = {
    0: "#cdc1b4",
    1: "#eee4da",
    2: "#ede0c8",
    3: "#f2b179",
    4: "#f59563",
    5: "#f67c5f",
    6: "#f65e3b",
    7: "#edcf72",
    8: "#edcc61",
    9: "#edc850",
    10: "#edc53f",
    11: "#edc22e",
}


def _tile_style(exponent: int) -> str:
    color = TILE_COLORS.get(exponent, "#3c3a32")
    ink = "#776e65" if exponent <= 2 else "#f9f6f2"
    return f"background:{color};color:{ink}"


def render_board(
    board,
    *,
    legal_mask: int = 0,
    best_action: int | None = None,
    spawn_index: int | None = None,
    note: str = "",
) -> str:
    """把 16 个指数画成带坐标的网格。

    **坐标必须显示出来**（a..d 是列、1..4 是行，a1 在左上）——手摆与复盘时，用户要能对着
    棋谱文字数格子（ADR 0036 决策 4）。
    """
    cells = np.asarray(board, dtype=np.uint8).reshape(-1)
    header = "".join(
        f'<th style="padding:2px 6px;color:#999;font-weight:400">{chr(ord("a") + column)}</th>'
        for column in range(4)
    )
    rows = []
    for row in range(4):
        tiles = []
        for column in range(4):
            index = row * 4 + column
            exponent = int(cells[index])
            value = (1 << exponent) if exponent else ""
            border = "3px solid #776e65" if index == spawn_index else "2px solid transparent"
            tiles.append(
                f'<td style="{_tile_style(exponent)};{border};width:64px;height:64px;'
                f"text-align:center;vertical-align:middle;font-size:20px;font-weight:700;"
                f'border-radius:5px">{value}</td>'
            )
        head = f'<th style="padding:2px 6px;color:#999;font-weight:400">{row + 1}</th>'
        rows.append(f"<tr>{head}{''.join(tiles)}</tr>")
    extra = ""
    if best_action is not None and best_action >= 0:
        name = records.DIRECTIONS[best_action]
        extra += (
            f'<div style="margin-top:6px;color:#776e65">引擎选招：'
            f"<b>{DIRECTION_LABELS[name]}</b>（{name}）</div>"
        )
    if note:
        extra += f'<div style="margin-top:6px;color:#999;font-size:12px">{html.escape(note)}</div>'
    return (
        '<table style="border-spacing:5px;background:#bbada0;border-radius:6px;padding:5px">'
        f"<tr><th></th>{header}</tr>{''.join(rows)}</table>{extra}"
    )


def render_analysis(payload: dict | None) -> str:
    """分析面板。

    **"策略头先验"与"引擎选招"分开标**：项目从一开始就不让探针做 argmax（选招口径只有引擎
    一处，见 ADR 0006），所以这里显示的先验只是"策略头觉得哪个方向好"，而引擎真走哪一步
    看的是选招。另外会话的批组成与评测不同（单盘面，ADR 0017），这件事跟着结果一起标出来。
    """
    if not payload:
        return '<div style="color:#999">还没有分析结果</div>'
    rows = []
    for direction in payload["directions"]:
        chosen = direction["action"] == payload.get("best_action")
        mark = " ← 引擎选招" if chosen else ""
        color = "#776e65" if direction["legal"] else "#ccc"
        rows.append(
            f'<tr style="color:{color}">'
            f'<td style="padding:2px 10px">{DIRECTION_LABELS.get(direction["name"], direction["name"])}</td>'
            f"<td>{'合法' if direction['legal'] else '非法'}</td>"
            f"<td>{direction['visits']}</td><td>{direction['q_mean']:.0f}</td>"
            f"<td>{direction['one_ply']:.0f}</td>"
            f'<td>{direction["prior"]:.3f}<span style="color:#c0392b">{mark}</span></td></tr>'
        )
    protocol = payload.get("protocol", {})
    footer = (
        f"策略 {protocol.get('strategy')}，实际完成 {payload['simulations']} 次模拟，"
        f"耗时 {payload['seconds']:.3f}s，根值 {payload['root_value']:.0f}"
    )
    note = protocol.get("note", "")
    return (
        "<table style='border-collapse:collapse'>"
        "<tr style='color:#999;font-size:12px'><th></th><th>合法</th><th>访问数</th>"
        "<th>Q（含立即分）</th><th>r + A(u)</th><th>策略头先验</th></tr>"
        + "".join(rows)
        + "</table>"
        + f"<div style='margin-top:8px;color:#999;font-size:12px'>{html.escape(footer)}"
        + (f"<br>{html.escape(str(note))}" if note else "")
        + "</div>"
    )


def render_status(state: dict | None) -> str:
    if not state:
        return '<div style="color:#999">会话未连接</div>'
    ending = '<b style="color:#c0392b">终局</b>' if state["terminal"] else "可走"
    return (
        f"<div>得分 <b>{state['score']}</b>　步数 <b>{state['moves']}</b>　"
        f"最大块 <b>{state['max_face'] or '-'}</b>　{ending}　"
        f"模型 {html.escape(state['model_id'] or '-')}（{state['device']}）</div>"
    )


class AppState:
    """界面持有的全部可变状态。**会话只有一个**，所有命令都用同一把锁串起来。"""

    def __init__(self) -> None:
        self.client: SessionClient | None = None
        self.checkpoint = ""
        self.lock = threading.Lock()
        self.last_move_at = 0.0
        self.editor = np.zeros(records.BOARD_CELLS, dtype=np.uint8)
        self.replay: list[records.Command] = []
        self.replay_upto = 0

    def attach(self, checkpoint: str, device: str, strategy: str, simulations: int, time_ms: int) -> None:
        if self.client is not None and checkpoint == self.checkpoint:
            return  # 同一个模型不重建会话：重建会丢掉当前盘面与棋谱
        self.close()
        self.client = SessionClient(
            session_argv(
                checkpoint=checkpoint,
                device=device,
                strategy=strategy,
                simulations=simulations,
                time_ms=time_ms,
            )
        )
        self.checkpoint = checkpoint

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def call(self, command: str, timeout: float = 300.0) -> list[dict]:
        if self.client is None:
            raise SessionError("还没有连接会话：先在上方选一个 checkpoint 并点「连接 / 应用设置」")
        return self.client.request(command, timeout=timeout)


def build_app(state: AppState | None = None):
    """搭出界面。返回 (demo, state)。"""
    import gradio as gr

    state = state or AppState()
    skip = gr.skip()

    def last_state(frames: list[dict]) -> dict | None:
        for frame in reversed(frames):
            if "state" in frame:
                return frame["state"]
        return None

    def last_analysis(frames: list[dict]) -> dict | None:
        for frame in reversed(frames):
            if "directions" in frame:
                return frame
        return None

    def redraw(frames: list[dict], note: str = ""):
        current = last_state(frames)
        report = last_analysis(frames)
        spawn = None
        for frame in frames:
            step = frame.get("step")
            if step and int(step.get("spawn_index", -1)) >= 0:
                spawn = int(step["spawn_index"])
        board = current["board"] if current else state.editor.tolist()
        legal = current["legal_mask"] if current else 0
        return (
            render_board(
                board, legal_mask=legal, best_action=(report or {}).get("best_action"), spawn_index=spawn
            ),
            render_status(current),
            render_analysis(report),
            f'<div style="color:#999">{html.escape(note)}</div>' if note else "",
        )

    def fail(error: Exception):
        return (
            render_board(state.editor.tolist()),
            render_status(None),
            render_analysis(None),
            f'<div style="color:#c0392b">{html.escape(str(error))}</div>',
        )

    def apply_settings(strategy_label: str, simulations, time_ms) -> None:
        state.call(f"set_strategy {STRATEGY_LABELS[strategy_label]}")
        state.call(f"set_simulations {max(0, int(simulations or 0))}")
        state.call(f"set_time_ms {max(0, int(time_ms or 0))}")

    with gr.Blocks(title="re-2048") as demo:
        gr.Markdown(
            f"# re-2048　交互分析\n输出根 `{paths.output_root()}`　棋盘坐标：字母是列、数字是行，`a1` 在左上"
        )

        models = inventory.list_models()
        with gr.Row():
            model_box = gr.Dropdown(
                choices=[entry.label() for entry in models],
                value=models[0].label() if models else None,
                label="模型（扫 runs/ 与 models/）",
                scale=4,
            )
            refresh_models = gr.Button("重新扫描", scale=1)
            device_box = gr.Dropdown(choices=["cpu", "cuda"], value="cpu", label="设备", scale=1)

        with gr.Row():
            strategy_box = gr.Dropdown(
                choices=list(STRATEGY_LABELS), value="搜索（NN-MCTS）", label="AI 策略", scale=2
            )
            sims_box = gr.Number(value=256, precision=0, label="搜索模拟数（0 = 不限）", scale=1)
            time_box = gr.Number(value=0, precision=0, label="最大搜索时间 毫秒（0 = 不限）", scale=1)
            interval_box = gr.Number(value=2000, precision=0, label="自走间隔 毫秒（≥100）", scale=1)
            connect = gr.Button("连接 / 应用设置", variant="primary", scale=1)

        with gr.Tabs():
            with gr.Tab("计算 / 分析"):
                mode_box = gr.Radio(choices=list(MODES), value="只分析", label="模式（可随时切换）")
                with gr.Row():
                    with gr.Column(scale=2):
                        board_html = gr.HTML(render_board(np.zeros(16, dtype=np.uint8)))
                        status_html = gr.HTML(render_status(None))
                    with gr.Column(scale=3):
                        analysis_html = gr.HTML(render_analysis(None))
                note_html = gr.HTML()
                with gr.Row():
                    analyze_btn = gr.Button("分析当前局面", variant="primary")
                    undo_btn = gr.Button("撤销一步")
                    clear_btn = gr.Button("清空棋盘")
                gr.Markdown("玩家模式：按方向落子（生成块由引擎生成并记进棋谱）")
                with gr.Row():
                    direction_buttons = [gr.Button(DIRECTION_LABELS[name]) for name in records.DIRECTIONS]
                with gr.Row():
                    record_name = gr.Textbox(value="game", label="棋谱名", scale=2)
                    save_btn = gr.Button("保存棋谱到 records/", scale=2)

            with gr.Tab("棋盘编辑器"):
                gr.Markdown(
                    "摆好之后发到计算页。**没有合法方向的局面会被拒绝**；"
                    "上限是 131072（2^17，4×4 的理论上界，ADR 0036 决策 6）。"
                )
                tile_boxes: list = []
                face_choices = [records.EMPTY_TOKEN] + [
                    str(1 << exponent) for exponent in range(1, records.MAX_EXPONENT + 1)
                ]
                for row in range(4):
                    with gr.Row():
                        for column in range(4):
                            tile_boxes.append(
                                gr.Dropdown(
                                    choices=face_choices,
                                    value=records.EMPTY_TOKEN,
                                    label=f"{chr(ord('a') + column)}{row + 1}",
                                    scale=1,
                                )
                            )
                with gr.Row():
                    editor_new = gr.Button("新建空局面")
                    editor_to_game = gr.Button("发送到计算页", variant="primary")

            with gr.Tab("复盘"):
                gr.Markdown(
                    "载入一份棋谱并逐步回放。回放喂的是棋谱里的 `set`，**不重新生成块**——"
                    "这正是棋谱不含随机性（ADR 0036 决策 3）的用处。"
                )
                with gr.Row():
                    record_box = gr.Dropdown(label="棋谱（扫 records/）", scale=3)
                    refresh_records = gr.Button("重新扫描", scale=1)
                    replay_load = gr.Button("载入", variant="primary", scale=1)
                    export_btn = gr.Button("以当前局面开新一局", scale=1)
                replay_slider = gr.Slider(minimum=0, maximum=1, value=0, step=1, label="步进")
                replay_html = gr.HTML()
                replay_note = gr.HTML()

        # ---- 回调 -----------------------------------------------------------

        def on_refresh_models():
            labels = [entry.label() for entry in inventory.list_models()]
            return gr.update(choices=labels, value=labels[0] if labels else None)

        def on_connect(model_label, device, strategy_label, simulations, time_ms):
            entry = inventory.find_model(model_label) if model_label else None
            checkpoint = str(entry.path) if entry else ""
            try:
                with state.lock:
                    state.attach(
                        checkpoint=checkpoint,
                        device=device,
                        strategy=STRATEGY_LABELS[strategy_label],
                        simulations=max(0, int(simulations or 0)),
                        time_ms=max(0, int(time_ms or 0)),
                    )
                    frames = state.call("state")
                note = "已连接" if checkpoint else "没有选模型：编辑与复盘可用，分析与落子会报错"
                return redraw(frames, note)
            except (SessionError, RuntimeError) as error:
                return fail(error)

        def on_analyze(strategy_label, simulations, time_ms):
            try:
                with state.lock:
                    apply_settings(strategy_label, simulations, time_ms)
                    frames = state.call("analyze")
                return redraw(frames, "分析完成（口径与评测不同，见面板下方说明）")
            except (SessionError, RuntimeError) as error:
                return fail(error)

        def on_direction(direction: str, strategy_label, simulations, time_ms):
            try:
                with state.lock:
                    apply_settings(strategy_label, simulations, time_ms)
                    frames = state.call(f"move {direction}")
                return redraw(frames)
            except (SessionError, RuntimeError) as error:
                return fail(error)

        def on_undo():
            try:
                with state.lock:
                    frames = state.call("undo")
                return redraw(frames)
            except (SessionError, RuntimeError) as error:
                return fail(error)

        def on_clear():
            try:
                with state.lock:
                    frames = state.call("clear")
                return redraw(frames)
            except (SessionError, RuntimeError) as error:
                return fail(error)

        def on_save(name: str):
            target = paths.record_path((name or "game").strip() or "game")
            try:
                with state.lock:
                    state.call(f"save {target}")
                return f'<div style="color:#2d7">已写入 {html.escape(str(target))}</div>'
            except (SessionError, RuntimeError) as error:
                return f'<div style="color:#c0392b">{html.escape(str(error))}</div>'

        def on_tick(mode, strategy_label, simulations, time_ms, interval_ms):
            """自动执行的一次心跳：拿不到锁或还没到间隔就跳过。"""
            idle = (skip, skip, skip, skip)
            if mode != "自动执行":
                return idle
            interval = max(100, int(interval_ms or 100)) / 1000.0
            if time.monotonic() - state.last_move_at < interval:
                return idle
            if not state.lock.acquire(blocking=False):
                return idle  # 正在搜索：跳过这一次，搜索不被打断
            try:
                apply_settings(strategy_label, simulations, time_ms)
                frames = state.call("genmove")
                state.last_move_at = time.monotonic()
            except (SessionError, RuntimeError) as error:
                return fail(error)
            finally:
                state.lock.release()
            return redraw(frames, "自动执行中")

        def on_editor_new():
            state.editor = np.zeros(records.BOARD_CELLS, dtype=np.uint8)
            return [gr.update(value=records.EMPTY_TOKEN) for _ in range(16)]

        def on_editor_send(*values):
            board = np.zeros(records.BOARD_CELLS, dtype=np.uint8)
            for index, value in enumerate(values):
                text = str(value or records.EMPTY_TOKEN)
                if text != records.EMPTY_TOKEN:
                    board[index] = records.FACE_TO_EXPONENT[text]
            state.editor = board
            commands = ["clear"] + [
                f"set {1 << int(exponent)} {records.index_to_coordinate(index)}"
                for index, exponent in enumerate(board)
                if exponent
            ]
            try:
                with state.lock:
                    frames: list[dict] = []
                    for command in commands:
                        frames = state.call(command)
                return redraw(frames, "已发送到计算页")
            except (SessionError, RuntimeError) as error:
                return fail(error)

        def on_refresh_records():
            labels = [entry.label() for entry in inventory.list_records()]
            return gr.update(choices=labels, value=labels[0] if labels else None)

        def _record_path(label: str) -> Path | None:
            for entry in inventory.list_records():
                if entry.label() == label:
                    return entry.path
            return None

        def on_replay_load(label: str):
            path = _record_path(label)
            if path is None:
                return (
                    gr.update(maximum=1, value=0),
                    render_board(np.zeros(16, dtype=np.uint8)),
                    (
                        '<div style="color:#c0392b">没有选中棋谱：先在 records/ 里有一份，'
                        "或在计算页保存一份</div>"
                    ),
                )
            try:
                state.replay = records.load(path)
            except records.RecordError as error:
                return (
                    gr.update(maximum=1, value=0),
                    "",
                    (f'<div style="color:#c0392b">{html.escape(str(error))}</div>'),
                )
            state.replay_upto = len(state.replay)
            steps = sum(1 for command in state.replay if isinstance(command, records.Move))
            board = records.apply(state.replay).board
            return (
                gr.update(maximum=max(1, len(state.replay)), value=len(state.replay)),
                render_board(board, note=f"{path.name}（{steps} 步）"),
                (
                    f'<div style="color:#999">已载入 {html.escape(path.name)}：'
                    f"{len(state.replay)} 条命令，{steps} 步</div>"
                ),
            )

        def on_replay_step(prefix) -> tuple[str, str]:
            if not state.replay:
                return render_board(np.zeros(16, dtype=np.uint8)), ""
            state.replay_upto = max(0, min(len(state.replay), int(prefix)))
            result = records.apply(state.replay[: state.replay_upto])
            return (
                render_board(result.board, note=f"前 {state.replay_upto} 条命令"),
                f'<div style="color:#999">得分 {result.total_score}，已走 {result.move_count} 步</div>',
            )

        def on_export():
            """把回放到的局面**当成新开一局**发给计算页。

            这里只送 `clear` + `set`，**不送 `move`**：协议里的 `move` 会由引擎生成一个块，
            而棋谱里的 `move` 后面跟着的是显式的 `set`。把棋谱的 move 当协议命令发，会每步
            多生成一个块——盘面立刻就不是棋谱里的那盘了。
            """
            if not state.replay:
                return fail(SessionError("还没有载入棋谱"))
            board = records.apply(state.replay[: state.replay_upto]).board
            try:
                with state.lock:
                    frames = state.call("clear")
                    for index, exponent in enumerate(board):
                        if exponent:
                            face = records.EXPONENT_TO_FACE[int(exponent)]
                            frames = state.call(f"set {face} {records.index_to_coordinate(index)}")
                return redraw(frames, f"已以第 {state.replay_upto} 条命令处的局面开新一局")
            except (SessionError, RuntimeError) as error:
                return fail(error)

        # ---- 绑定 -----------------------------------------------------------

        outputs = [board_html, status_html, analysis_html, note_html]
        refresh_models.click(on_refresh_models, outputs=[model_box])
        connect.click(
            on_connect,
            inputs=[model_box, device_box, strategy_box, sims_box, time_box],
            outputs=outputs,
        )
        analyze_btn.click(on_analyze, inputs=[strategy_box, sims_box, time_box], outputs=outputs)
        undo_btn.click(on_undo, outputs=outputs)
        clear_btn.click(on_clear, outputs=outputs)
        for name, button in zip(records.DIRECTIONS, direction_buttons):
            button.click(
                lambda strategy_label, simulations, time_ms, direction=name: on_direction(
                    direction, strategy_label, simulations, time_ms
                ),
                inputs=[strategy_box, sims_box, time_box],
                outputs=outputs,
            )
        save_btn.click(on_save, inputs=[record_name], outputs=[note_html])
        # 心跳固定 0.25s：真正的节奏由「自走间隔」控制，这样改间隔不需要重建定时器。
        timer = gr.Timer(value=0.25)
        timer.tick(
            on_tick,
            inputs=[mode_box, strategy_box, sims_box, time_box, interval_box],
            outputs=outputs,
        )

        editor_new.click(on_editor_new, outputs=tile_boxes)
        editor_to_game.click(on_editor_send, inputs=tile_boxes, outputs=outputs)
        refresh_records.click(on_refresh_records, outputs=[record_box])
        replay_load.click(
            on_replay_load, inputs=[record_box], outputs=[replay_slider, replay_html, replay_note]
        )
        replay_slider.change(on_replay_step, inputs=[replay_slider], outputs=[replay_html, replay_note])
        export_btn.click(on_export, outputs=outputs)

    return demo, state
