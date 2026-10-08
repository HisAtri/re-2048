"""WebUI：计算/分析、棋盘编辑器、复盘。

界面**不做棋规、不选招**：它只把命令交给会话进程，再把会话回报的状态画出来。所有引擎调用
都经 `SessionClient`（文本协议 + 独立进程），所以 Gradio 的多线程回调不会碰到引擎——
引擎不是为并发独立搜索设计的（见 ADR 0038）。

三条交互上的取舍：

* **运行与停止是同一个位置上的两个状态**，不是两个并存的按钮：点「自动执行」之后它变成
  「停止」，停下来再变回来（靠 `visible` 互换；不能只用一个按钮，因为 Gradio 会把同一控件的
  第二次点击排进队列，那样"停止"就点不动了）。自动执行期间手动控件置灰。
* **设置改动立即生效**：每次分析/落子前都把设置发给会话，所以没有"应用设置"这一步。
  不常用的旋钮（模拟数、限时、自走间隔）收进折叠面板，主视图只留模型、设备与策略。
* **自动执行与分析都是流式的**：一次生成器调用里逐帧 yield，浏览器只维持一条长连接。
  曾经用 `gr.Timer` 每 250ms 打一次会话，实测一屏 50 req/s 的请求洪水（ADR 0039 决策 4）。
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

STRATEGY_LABELS = {
    "搜索（NN-MCTS）": "search",
    "策略头（测试用）": "policy",
    "价值头（测试用）": "value",
}
STRATEGY_SHORT = {"search": "搜索", "policy": "策略头", "value": "价值头"}
DIRECTION_LABELS = {"up": "上", "right": "右", "down": "下", "left": "左"}

# 标准 2048 配色（指数 → 底色/字色）。数字块是这一页唯一的视觉主体，配色不对就认不出局面。
_PALETTE = {
    1: ("#eee4da", "#776e65"),
    2: ("#ede0c8", "#776e65"),
    3: ("#f2b179", "#f9f6f2"),
    4: ("#f59563", "#f9f6f2"),
    5: ("#f67c5f", "#f9f6f2"),
    6: ("#f65e3b", "#f9f6f2"),
    7: ("#edcf72", "#f9f6f2"),
    8: ("#edcc61", "#f9f6f2"),
    9: ("#edc850", "#f9f6f2"),
    10: ("#edc53f", "#f9f6f2"),
    11: ("#edc22e", "#f9f6f2"),
}
_PALETTE_BEYOND = ("#3c3a32", "#f9f6f2")
_EMPTY_CELL = "rgba(238, 228, 218, 0.35)"
_FONT = 'system-ui, "Segoe UI", "Helvetica Neue", Arial, "PingFang SC", sans-serif'

# 页面样式自带一份（带 r2048- 前缀），免得跟 Gradio 自己的 CSS 打架。
_STYLE = f"""<style>
/* 宽度**跟着容器走**（width:100% + 四列 1fr），而不是由内容决定。
   上一版是"固定格子尺寸 + width:max-content"：父容器的宽度是 flex 算出来的、猜不准，
   只要比棋盘窄一点点，棋盘就直接溢出成一条横向滚动条（实测就是那样）。
   方块用 aspect-ratio 保持正方形，于是格子随容器缩放而不变形，也不会溢出。 */
.r2048-board {{ display: grid; grid-template-columns: 16px repeat(4, 1fr); gap: 6px;
  width: 100%; min-width: 0; max-width: 380px; box-sizing: border-box;
  background: #bbada0; padding: 9px; border-radius: 10px; }}
.r2048-label {{ display: flex; align-items: center; justify-content: center;
  color: #efe6dd; font-size: 12px; font-weight: 600; }}
.r2048-tile {{ aspect-ratio: 1; display: flex; align-items: center; justify-content: center;
  border-radius: 6px; font-weight: 800; font-variant-numeric: tabular-nums;
  font-family: {_FONT}; }}
.r2048-d1, .r2048-d2 {{ font-size: 32px; }} .r2048-d3 {{ font-size: 26px; }}
.r2048-d4 {{ font-size: 21px; }} .r2048-d5 {{ font-size: 17px; }} .r2048-d6 {{ font-size: 14px; }}
.r2048-spawn {{ box-shadow: inset 0 0 0 3px #8f7a66; }}
.r2048-badge {{ display: inline-block; margin-top: 10px; padding: 4px 10px; border-radius: 6px;
  background: #efe6dd; color: #776e65; font-size: 14px; }}
.r2048-dim {{ color: #a99b90; font-weight: 400; }}
.r2048-chips {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
  font-family: {_FONT}; }}
.r2048-chip {{ padding: 3px 10px; border-radius: 999px; font-size: 13px;
  background: #f1ece6; color: #776e65; }}
.r2048-chip.key {{ background: #efe6dd; }}
.r2048-chip.ok {{ background: #e3f2e8; color: #2d7a4f; }}
.r2048-chip.warn {{ background: #fff4e0; color: #a56a12; }}
.r2048-chip.bad {{ background: #fde8e6; color: #c0392b; }}
.r2048-chip.muted {{ background: #f5f5f5; color: #9a9a9a; }}
/* 方向键：Gradio 的按钮默认会 flex-grow 把整列撑满，四个键就变成四条横杠。
   这里钉住宽度并居中，让它读起来是一个十字键。 */
#r2048-dpad {{ align-items: center; }}
#r2048-dpad button {{ width: 46px; max-width: 46px; flex-grow: 0; min-width: 0; }}
.r2048-panel {{ border-collapse: collapse; font-variant-numeric: tabular-nums;
  font-family: {_FONT}; }}
.r2048-panel th {{ color: #999; font-size: 12px; font-weight: 600; padding: 3px 10px; }}
.r2048-panel td {{ padding: 3px 10px; font-size: 14px; }}
.r2048-panel tr.chosen td {{ background: #fff3e0; font-weight: 700; }}
.r2048-panel tr.legal td {{ color: #553b2e; }} .r2048-panel tr.illegal td {{ color: #ccc; }}
.r2048-note {{ margin-top: 8px; color: #999; font-size: 12px; font-family: {_FONT}; }}
.r2048-err {{ margin-top: 8px; color: #c0392b; font-size: 13px; font-family: {_FONT}; }}
.r2048-ok {{ margin-top: 8px; color: #2d7a4f; font-size: 13px; font-family: {_FONT}; }}
</style>"""


def _chip(text: str, tone: str = "", bold: str = "") -> str:
    body = f"<b>{html.escape(bold)}</b> {html.escape(text)}" if bold else html.escape(text)
    return f'<span class="r2048-chip {tone}">{body}</span>'


def render_status(state: dict | None, settings: dict | None = None, running: bool = False) -> str:
    """一行状态：玩的是什么（模型/设备/策略）、走到哪了（分数/步数/最大块）、能不能走。

    状态放在一块，是因为"我现在看着哪个模型、什么口径"必须一眼可见——它决定了屏幕上的数字
    是什么意思。**空盘面与终局分开标**：两者都没有合法方向，但下一步动作完全不同。
    """
    parts: list[str] = []
    if running:
        parts.append(_chip("自动执行中", "ok"))
    if not state:
        parts.append(_chip("会话未连接", "muted"))
        return _STYLE + f'<div class="r2048-chips">{"".join(parts)}</div>'
    settings = settings or {}
    parts.append(_chip((state.get("model_id") or "-").split("+")[0], "key"))
    parts.append(_chip(state.get("device", "")))
    parts.append(
        _chip(
            f"{STRATEGY_SHORT.get(settings.get('strategy', ''), '')} · {settings.get('simulations', 0)} 模拟"
        )
    )
    parts.append(_chip(f"得分 {state['score']} · 步数 {state['moves']} · 最大块 {state['max_face'] or '-'}"))
    if state.get("seed"):
        parts.append(_chip(f"种子 {state['seed']}"))
    if state.get("empty"):
        parts.append(_chip("空盘面 · 点「新开一局」开始", "warn"))
    elif state.get("terminal"):
        parts.append(_chip("终局", "bad"))
    else:
        parts.append(_chip("可走", "ok"))
    return _STYLE + f'<div class="r2048-chips">{"".join(parts)}</div>'


def _font_class(face: int) -> str:
    return f"r2048-d{min(len(str(face)), 6)}"


def render_board(
    board,
    *,
    best_action: int | None = None,
    spawn_index: int | None = None,
    note: str = "",
) -> str:
    """棋盘组件：4×4 网格 + 行列坐标（a..d 是列、1..4 是行，a1 在左上）。

    坐标必须在棋盘上显示出来——手摆与复盘时用户要能对着棋谱文字数格子（ADR 0036 决策 4）。
    """
    cells = np.asarray(board, dtype=np.uint8).reshape(-1)
    parts = [_STYLE, '<div class="r2048-board">', '<div class="r2048-label"></div>']
    parts += [f'<div class="r2048-label">{chr(ord("a") + column)}</div>' for column in range(4)]
    for row in range(4):
        parts.append(f'<div class="r2048-label">{row + 1}</div>')
        for column in range(4):
            index = row * 4 + column
            exponent = int(cells[index])
            ring = " r2048-spawn" if index == spawn_index else ""
            if exponent:
                background, ink = _PALETTE.get(exponent, _PALETTE_BEYOND)
                face = 1 << exponent
                parts.append(
                    f'<div class="r2048-tile {_font_class(face)}{ring}"'
                    f' style="background:{background};color:{ink}">{face}</div>'
                )
            else:
                parts.append(f'<div class="r2048-tile{ring}" style="background:{_EMPTY_CELL}"></div>')
    parts.append("</div>")
    if best_action is not None and best_action >= 0:
        name = records.DIRECTIONS[best_action]
        parts.append(
            f'<div class="r2048-badge">引擎选招 <b>{DIRECTION_LABELS[name]}</b>'
            f'<span class="r2048-dim">（{name}）</span></div>'
        )
    if note:
        parts.append(f'<div class="r2048-note">{html.escape(note)}</div>')
    return "".join(parts)


def render_analysis(payload: dict | None) -> str:
    """分析面板。

    **"策略头先验"与"引擎选招"分开标**：项目从一开始就不让探针做 argmax（选招口径只有引擎
    一处，见 ADR 0006），所以先验只是"策略头觉得哪个方向好"，引擎真走哪一步看选招。
    另外会话的批组成与评测不同（单盘面，ADR 0017），这件事跟着结果一起标出来。
    """
    if not payload:
        return (
            f'{_STYLE}<div class="r2048-note">还没有分析结果：'
            "点「新开一局」或摆好盘面，再点「分析当前局面」</div>"
        )
    rows = []
    for direction in payload["directions"]:
        chosen = direction["action"] == payload.get("best_action")
        classes = " ".join(
            name
            for name, on in (
                ("chosen", chosen),
                ("legal", direction["legal"]),
                ("illegal", not direction["legal"]),
            )
            if on
        )
        rows.append(
            f'<tr class="{classes}"><td>{DIRECTION_LABELS.get(direction["name"], direction["name"])}'
            f"{'　<b>← 引擎选招</b>' if chosen else ''}</td>"
            f"<td>{'合法' if direction['legal'] else '非法'}</td>"
            f"<td>{direction['visits']}</td><td>{direction['q_mean']:.0f}</td>"
            f"<td>{direction['one_ply']:.0f}</td><td>{direction['prior']:.3f}</td></tr>"
        )
    protocol = payload.get("protocol", {})
    footer = (
        f"策略 {protocol.get('strategy')}，实际完成 {payload['simulations']} 次模拟，"
        f"耗时 {payload['seconds']:.3f}s，根值 {payload['root_value']:.0f}"
    )
    note = protocol.get("note", "")
    return (
        _STYLE + "<table class='r2048-panel'>"
        "<tr><th>方向</th><th>合法</th><th>访问数</th><th>Q（含立即分）</th>"
        "<th>r + A(u)</th><th>策略头先验</th></tr>"
        + "".join(rows)
        + "</table>"
        + f'<div class="r2048-note">{html.escape(footer)}'
        + (f"<br>{html.escape(str(note))}" if note else "")
        + "</div>"
    )


class AppState:
    """界面持有的全部可变状态。**会话只有一个**，所有命令都用同一把锁串起来。"""

    def __init__(self) -> None:
        self.client: SessionClient | None = None
        self.checkpoint = ""
        self.lock = threading.Lock()
        self.last_move_at = 0.0
        self.auto_running = False
        self.auto_stop = False
        self.editor = np.zeros(records.BOARD_CELLS, dtype=np.uint8)
        self.replay: list[records.Command] = []
        self.replay_upto = 0
        # 最近一次从会话收到的 state。分析帧里没有盘面，渲染时必须有个可回落的来源，
        # 否则"分析中"的几帧会把棋盘画成另一个局面。
        self.last_state: dict | None = None

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
        self.last_state = None

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def call(self, command: str, timeout: float = 300.0) -> list[dict]:
        if self.client is None:
            raise SessionError("还没有连接模型：先选一个 checkpoint 并点「连接模型」")
        return self.client.request(command, timeout=timeout)

    def stream(self, command: str, timeout: float = 300.0):
        if self.client is None:
            raise SessionError("还没有连接模型：先选一个 checkpoint 并点「连接模型」")
        return self.client.stream(command, timeout=timeout)


def build_app(state: AppState | None = None):
    """搭出界面。返回 (demo, state)。"""
    import gradio as gr

    state = state or AppState()

    def last_state_of(frames: list[dict]) -> dict | None:
        for frame in reversed(frames):
            if "state" in frame:
                return frame["state"]
        return None

    def last_analysis_of(frames: list[dict]) -> dict | None:
        for frame in reversed(frames):
            if "directions" in frame:
                return frame
        return None

    def apply_settings(strategy_label: str, simulations, time_ms) -> dict:
        """把设置发给会话。每次动作前都调，所以改设置立即生效、不需要「应用」。"""
        state.call(f"set_strategy {STRATEGY_LABELS[strategy_label]}")
        state.call(f"set_simulations {max(0, int(simulations or 0))}")
        state.call(f"set_time_ms {max(0, int(time_ms or 0))}")
        return {
            "strategy": STRATEGY_LABELS[strategy_label],
            "simulations": max(0, int(simulations or 0)),
        }

    with gr.Blocks(title="re-2048") as demo:
        gr.Markdown(f"**re-2048** · 输出根 `{paths.output_root()}`")

        models = inventory.list_models()
        with gr.Row():
            model_box = gr.Dropdown(
                choices=[entry.label() for entry in models],
                value=inventory.default_model(models),
                label="模型（相对输出根的路径；可直接输入过滤）",
                scale=5,
            )
            device_box = gr.Dropdown(choices=["cpu", "cuda"], value="cpu", label="设备", scale=1)
            connect = gr.Button("连接模型", variant="primary", scale=1)
            refresh_models = gr.Button("重新扫描", scale=1)
        with gr.Accordion("搜索设置（改动立即生效）", open=False), gr.Row():
            strategy_box = gr.Dropdown(
                choices=list(STRATEGY_LABELS), value="搜索（NN-MCTS）", label="AI 策略", scale=2
            )
            sims_box = gr.Number(value=256, precision=0, label="搜索模拟数（0 = 不限）", scale=1)
            time_box = gr.Number(value=0, precision=0, label="最大搜索时间 毫秒（0 = 不限）", scale=1)
            interval_box = gr.Number(value=2000, precision=0, label="自走间隔 毫秒（≥100）", scale=1)
            include_resume = gr.Checkbox(value=False, label="模型列表含续跑点 last.pt", scale=1)

        with gr.Tabs():
            with gr.Tab("计算 / 分析"), gr.Row():
                # 左：棋盘 + 方向键 + 动作。方向键与棋盘是**同一件事**，摆在一起——上一版把方向键
                # 单独放一行，结果整块被挤到折线以下，一进页面根本看不到它。
                with gr.Column(scale=0, min_width=470):
                    status_html = gr.HTML(render_status(None))
                    with gr.Row(equal_height=False):
                        board_html = gr.HTML(render_board(np.zeros(16, dtype=np.uint8)))
                        with gr.Column(scale=0, min_width=130, elem_id="r2048-dpad"):
                            dpad_up = gr.Button(DIRECTION_LABELS["up"], min_width=44)
                            with gr.Row():
                                dpad_left = gr.Button(DIRECTION_LABELS["left"], min_width=44)
                                dpad_right = gr.Button(DIRECTION_LABELS["right"], min_width=44)
                            dpad_down = gr.Button(DIRECTION_LABELS["down"], min_width=44)
                    direction_buttons = [dpad_up, dpad_right, dpad_down, dpad_left]
                    with gr.Row():
                        newgame_btn = gr.Button("新开一局", variant="primary", min_width=96)
                        analyze_btn = gr.Button("分析当前局面", min_width=96)
                        undo_btn = gr.Button("撤销一步", min_width=84)
                        clear_btn = gr.Button("清空", min_width=64)
                    with gr.Row():
                        # 运行与停止**同一个位置**：点自动执行之后它变成停止，停下来再变回来。
                        auto_start = gr.Button("▶ 自动执行", variant="primary")
                        auto_stop = gr.Button("■ 停止", variant="stop", visible=False)
                # 右：分析结果。这是"看着数字决定下一步"的地方，给足宽度。
                with gr.Column(scale=1, min_width=430):
                    analysis_html = gr.HTML(render_analysis(None))
                    with gr.Row():
                        record_name = gr.Textbox(value="game", label="棋谱名", scale=3)
                        save_btn = gr.Button("保存棋谱", scale=1, min_width=90)
                    note_html = gr.HTML()

            with gr.Tab("棋盘编辑器"):
                gr.Markdown(
                    "手摆一个局面发到计算页（「新开一局」与之无关：那是标准开局）。"
                    "**没有合法方向的局面会被拒绝**；上限 131072（2^17，4×4 的理论上界）。"
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
                    editor_new = gr.Button("清空编辑器")
                    editor_to_game = gr.Button("发送到计算页", variant="primary")

            with gr.Tab("复盘"):
                gr.Markdown(
                    "载入一份棋谱逐步回放。回放喂的是棋谱里的 `set`，**不重新生成块**——"
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

        manual = [newgame_btn, analyze_btn, undo_btn, clear_btn, *direction_buttons]
        outputs = [
            board_html,
            status_html,
            analysis_html,
            note_html,
            auto_start,
            auto_stop,
            *manual,
        ]

        def view(frames: list[dict], note: str = "", settings: dict | None = None, running=None):
            """把若干帧渲染成整屏视图。**运行/停止的可见性也在这里切**，所以它只有一处。"""
            running = state.auto_running if running is None else running
            fresh = last_state_of(frames)
            if fresh is not None:
                state.last_state = fresh
            current = state.last_state
            report = last_analysis_of(frames)
            spawn = None
            for frame in frames:
                step = frame.get("step")
                if step and int(step.get("spawn_index", -1)) >= 0:
                    spawn = int(step["spawn_index"])
            board = current["board"] if current else [0] * records.BOARD_CELLS
            return (
                render_board(board, best_action=(report or {}).get("best_action"), spawn_index=spawn),
                render_status(current, settings, running=running),
                render_analysis(report),
                f'<div class="r2048-note">{html.escape(note)}</div>' if note else "",
                gr.update(visible=not running),
                gr.update(visible=running),
                *[gr.update(interactive=not running) for _ in manual],
            )

        def fail(error: Exception):
            return view([], str(error))

        def on_refresh_models(include_resume):
            found = inventory.list_models(include_resume=include_resume)
            return gr.update(choices=[entry.label() for entry in found], value=inventory.default_model(found))

        def on_connect(model_label, include_resume, device, strategy_label, simulations, time_ms):
            entry = inventory.find_model(model_label, include_resume=include_resume) if model_label else None
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
                    settings = apply_settings(strategy_label, simulations, time_ms)
                    frames = state.call("state")
                note = (
                    "已连接；盘面是空的——点「新开一局」开始一局标准对局"
                    if checkpoint
                    else "没有选模型：编辑与复盘可用，分析与落子会报错"
                )
                return view(frames, note, settings)
            except (SessionError, RuntimeError) as error:
                return fail(error)

        def on_newgame(strategy_label, simulations, time_ms):
            if state.auto_running:
                return view([], "自动执行中：先点「停止」")
            try:
                with state.lock:
                    settings = apply_settings(strategy_label, simulations, time_ms)
                    frames = state.call("newgame")
                return view(frames, f"新开一局（种子 {frames[0]['seed']}）", settings)
            except (SessionError, RuntimeError) as error:
                return fail(error)

        def on_analyze(strategy_label, simulations, time_ms):
            """分析：把协议的流式帧直接推给界面（一帧一条消息，不另发请求）。"""
            if state.auto_running:
                # 这个函数是**生成器**，所以不能用 `return` 交回视图——Gradio 会把这个返回值
                # 丢掉，于是"自动执行中点分析"变成什么都不发生（不报错，只是没反应）。
                yield view([], "自动执行中：先点「停止」")
                return
            settings = None
            try:
                with state.lock:
                    settings = apply_settings(strategy_label, simulations, time_ms)
                    for frame in state.stream("analyze"):
                        done = not frame.get("is_during_search", False)
                        yield view([frame], "分析完成" if done else "分析中…", settings)
            except (SessionError, RuntimeError) as error:
                yield fail(error)

        def on_direction(direction: str, strategy_label, simulations, time_ms):
            if state.auto_running:
                return view([], "自动执行中：先点「停止」")
            try:
                with state.lock:
                    settings = apply_settings(strategy_label, simulations, time_ms)
                    frames = state.call(f"move {direction}")
                return view(frames, "", settings)
            except (SessionError, RuntimeError) as error:
                return fail(error)

        def on_undo():
            try:
                with state.lock:
                    frames = state.call("undo")
                return view(frames)
            except (SessionError, RuntimeError) as error:
                return fail(error)

        def on_clear():
            if state.auto_running:
                return view([], "自动执行中：先点「停止」")
            try:
                with state.lock:
                    frames = state.call("clear")
                return view(frames, "已清空（点「新开一局」开始一局标准对局）")
            except (SessionError, RuntimeError) as error:
                return fail(error)

        def on_save(name: str):
            target = paths.record_path((name or "game").strip() or "game")
            try:
                with state.lock:
                    state.call(f"save {target}")
                return f'<div class="r2048-ok">已写入 {html.escape(str(target))}</div>'
            except (SessionError, RuntimeError) as error:
                return f'<div class="r2048-err">{html.escape(str(error))}</div>'

        def on_auto_start(strategy_label, simulations, time_ms, interval_ms):
            """自动执行：**一次生成器调用**里每一步流式推给界面，不轮询。

            节奏按 ADR 0037：`下一次移动 = max(上次移动 + 间隔, 搜索完成)`。所以这里在一步
            做完之后才补睡"距上次移动还差多少"——搜索永不被砍断，间隔只限制最快速度。
            """
            if state.auto_running:
                yield view([], "已经在自动执行")
                return
            state.auto_running = True
            state.auto_stop = False
            interval = max(100, int(interval_ms or 100)) / 1000.0
            settings = None
            try:
                yield view([], "自动执行开始", settings, running=True)
                while not state.auto_stop:
                    with state.lock:
                        settings = apply_settings(strategy_label, simulations, time_ms)
                        frames = state.call("genmove")
                        state.last_move_at = time.monotonic()
                    yield view(frames, f"自动执行中（间隔 {interval * 1000:.0f} ms）", settings)
                    remaining = interval - (time.monotonic() - state.last_move_at)
                    if remaining > 0:
                        time.sleep(remaining)
            except (SessionError, RuntimeError) as error:
                state.auto_running = False
                yield view([], str(error))
                return
            finally:
                state.auto_running = False
            yield view([], "自动执行已停止")

        def on_auto_stop():
            state.auto_stop = True
            return '<div class="r2048-note">正在停止（走完当前这一步）…</div>'

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
                return view(frames, "已发送到计算页")
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
                    ('<div class="r2048-err">没有选中棋谱：先在 records/ 里有一份，或在计算页保存一份</div>'),
                )
            try:
                state.replay = records.load(path)
            except records.RecordError as error:
                return (
                    gr.update(maximum=1, value=0),
                    render_board(np.zeros(16, dtype=np.uint8)),
                    f'<div class="r2048-err">{html.escape(str(error))}</div>',
                )
            state.replay_upto = len(state.replay)
            steps = sum(1 for command in state.replay if isinstance(command, records.Move))
            board = records.apply(state.replay).board
            return (
                gr.update(maximum=max(1, len(state.replay)), value=len(state.replay)),
                render_board(board, note=f"{path.name}（{steps} 步）"),
                (
                    f'<div class="r2048-note">已载入 {html.escape(path.name)}：'
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
                f'<div class="r2048-note">得分 {result.total_score}，已走 {result.move_count} 步</div>',
            )

        def on_export():
            """把回放到的局面**当成新开一局**发给计算页。

            只送 `clear` + `set`，**不送 `move`**：协议里的 `move` 会由引擎生成一个块，而棋谱
            里的 `move` 后面跟着的是显式的 `set`。把棋谱的 move 当协议命令发，每步会多生成
            一个块——盘面立刻就不是棋谱里那盘了。
            """
            if not state.replay:
                return fail(SessionError("还没有载入棋谱"))
            if state.auto_running:
                return view([], "自动执行中：先点「停止」")
            board = records.apply(state.replay[: state.replay_upto]).board
            try:
                with state.lock:
                    frames = state.call("clear")
                    for index, exponent in enumerate(board):
                        if exponent:
                            face = records.EXPONENT_TO_FACE[int(exponent)]
                            frames = state.call(f"set {face} {records.index_to_coordinate(index)}")
                return view(frames, f"已以第 {state.replay_upto} 条命令处的局面开新一局")
            except (SessionError, RuntimeError) as error:
                return fail(error)

        # ---- 绑定 -----------------------------------------------------------

        settings_inputs = [strategy_box, sims_box, time_box]
        refresh_models.click(on_refresh_models, inputs=[include_resume], outputs=[model_box])
        connect.click(
            on_connect,
            inputs=[model_box, include_resume, device_box, *settings_inputs],
            outputs=outputs,
        )
        newgame_btn.click(on_newgame, inputs=settings_inputs, outputs=outputs)
        analyze_btn.click(on_analyze, inputs=settings_inputs, outputs=outputs)
        undo_btn.click(on_undo, outputs=outputs)
        clear_btn.click(on_clear, outputs=outputs)
        for name, button in zip(records.DIRECTIONS, direction_buttons):
            button.click(
                lambda strategy_label, simulations, time_ms, direction=name: on_direction(
                    direction, strategy_label, simulations, time_ms
                ),
                inputs=settings_inputs,
                outputs=outputs,
            )
        save_btn.click(on_save, inputs=[record_name], outputs=[note_html])
        auto_start.click(on_auto_start, inputs=[*settings_inputs, interval_box], outputs=outputs)
        auto_stop.click(on_auto_stop, outputs=[note_html])

        editor_new.click(on_editor_new, outputs=tile_boxes)
        editor_to_game.click(on_editor_send, inputs=tile_boxes, outputs=outputs)
        refresh_records.click(on_refresh_records, outputs=[record_box])
        replay_load.click(
            on_replay_load, inputs=[record_box], outputs=[replay_slider, replay_html, replay_note]
        )
        replay_slider.change(on_replay_step, inputs=[replay_slider], outputs=[replay_html, replay_note])
        export_btn.click(on_export, outputs=outputs)

    return demo, state
