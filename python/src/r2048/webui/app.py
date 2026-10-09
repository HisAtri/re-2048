"""WebUI：计算/分析、棋盘编辑器、复盘。

界面**不做棋规、不选招**：它只把命令交给会话进程，再把会话回报的状态画出来。所有引擎调用
都经 `SessionClient`（文本协议 + 独立进程），所以 Gradio 的多线程回调不会碰到引擎——
引擎不是为并发独立搜索设计的（见 ADR 0038）。

三条交互上的取舍：

* **运行与停止是同一个位置上的两个状态**，不是两个并存的按钮：点「自动执行」之后它变成
  「停止」，停下来再变回来（靠 `visible` 互换；不能只用一个按钮，因为 Gradio 会把同一控件的
  第二次点击排进队列，那样"停止"就点不动了）。自动执行期间手动控件置灰。
* **设置改动立即生效**：每次分析/落子前都把设置发给会话，所以没有"应用设置"这一步。
  模拟数、限时、自走间隔常驻显示，方便在分析时核对；说明文字放在字段行下方以保持对齐。
* **自动执行与分析都是流式的**：一次生成器调用里逐帧 yield，浏览器只维持一条长连接。
  曾经用 `gr.Timer` 每 250ms 打一次会话，实测一屏 50 req/s 的请求洪水（ADR 0039 决策 4）。
"""

from __future__ import annotations

import html
import threading
import time
from functools import partial
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


def render_status(
    state: dict | None, settings: dict | None = None, running: bool = False, *, include_styles: bool = True
) -> str:
    """一行状态：玩的是什么（模型/设备/策略）、走到哪了（分数/步数/最大块）、能不能走。

    状态放在一块，是因为"我现在看着哪个模型、什么口径"必须一眼可见——它决定了屏幕上的数字
    是什么意思。**空盘面与终局分开标**：两者都没有合法方向，但下一步动作完全不同。
    """
    parts: list[str] = []
    if running:
        parts.append(_chip("自动执行中", "ok"))
    if not state:
        parts.append(_chip("会话未连接", "muted"))
        return (_STYLE if include_styles else "") + f'<div class="r2048-chips">{"".join(parts)}</div>'
    settings = settings or {}
    parts.append(_chip((state.get("model_id") or "-").split("+")[0], "key"))
    parts.append(_chip(state.get("device", "")))
    parts.append(
        _chip(
            f"{STRATEGY_SHORT.get(settings.get('strategy', ''), '')} · {settings.get('simulations', 0)} 模拟"
        )
    )
    parts.append(_chip(f"得分 {state['score']} · 步数 {state['moves']} · 最大块 {state['max_face'] or '-'}"))
    if state.get("seed") is not None:
        parts.append(_chip(f"种子 {state['seed']}"))
    if state.get("empty"):
        parts.append(_chip("空盘面 · 点「新开一局」开始", "warn"))
    elif state.get("terminal"):
        parts.append(_chip("终局", "bad"))
    else:
        parts.append(_chip("可走", "ok"))
    return (_STYLE if include_styles else "") + f'<div class="r2048-chips">{"".join(parts)}</div>'


def _font_class(face: int) -> str:
    return f"r2048-d{min(len(str(face)), 6)}"


def render_board(
    board,
    *,
    best_action: int | None = None,
    spawn_index: int | None = None,
    note: str = "",
    include_styles: bool = True,
) -> str:
    """棋盘组件：4×4 网格 + 行列坐标（a..d 是列、1..4 是行，a1 在左上）。

    坐标必须在棋盘上显示出来——手摆与复盘时用户要能对着棋谱文字数格子（ADR 0036 决策 4）。
    """
    cells = np.asarray(board, dtype=np.uint8).reshape(-1)
    parts = [_STYLE if include_styles else "", '<div class="r2048-board">', '<div class="r2048-label"></div>']
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


def render_analysis(payload: dict | None, *, context: str = "当前局面", include_styles: bool = True) -> str:
    """分析面板。

    **"策略头先验"与"引擎选招"分开标**：项目从一开始就不让探针做 argmax（选招口径只有引擎
    一处，见 ADR 0006），所以先验只是"策略头觉得哪个方向好"，引擎真走哪一步看选招。
    另外会话的批组成与评测不同（单盘面，ADR 0017），这件事跟着结果一起标出来。
    """
    stylesheet = _STYLE if include_styles else ""
    if not payload:
        return (
            f'{stylesheet}<div class="r2048-empty"><div class="r2048-empty-symbol" aria-hidden="true">↗</div>'
            "<h3>还没有分析结果</h3><p>先连接会话，再点「新开一局」或发送一个局面。"
            "<br>点击「分析当前局面」，比较四个方向的搜索结果。</p></div>"
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
        stylesheet
        + f'<div class="r2048-analysis-context">{html.escape(context)}的搜索结果</div>'
        + '<div class="r2048-metrics">'
        + f'<div class="r2048-metric"><span>实际模拟</span><strong>{payload["simulations"]:,}</strong></div>'
        + f'<div class="r2048-metric"><span>搜索耗时</span><strong>{payload["seconds"]:.3f}s</strong></div>'
        + f'<div class="r2048-metric"><span>根值</span><strong>{payload["root_value"]:.0f}</strong></div></div>'
        + "<div class='r2048-analysis-scroll' tabindex='0' aria-label='方向搜索分析'>"
        + "<table class='r2048-panel'><thead>"
        "<tr><th scope='col'>方向</th><th scope='col'>合法</th><th scope='col'>访问数</th>"
        "<th scope='col'>Q（含立即分）</th><th scope='col'>r + A(u)</th>"
        "<th scope='col'>策略头先验</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + f'<div class="r2048-note">{html.escape(footer)}'
        + (f"<br>{html.escape(str(note))}" if note else "")
        + "</div>"
    )


class AppState:
    """界面持有的全部可变状态。**会话只有一个**，所有命令都用同一把锁串起来。"""

    def __init__(self) -> None:
        self.client: SessionClient | None = None
        self.checkpoint = ""
        self.device = ""
        self.settings: dict = {}
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
        if self.client is not None and checkpoint == self.checkpoint and device == self.device:
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
        self.device = device
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
        state.settings = {
            "strategy": STRATEGY_LABELS[strategy_label],
            "simulations": max(0, int(simulations or 0)),
        }
        return state.settings

    # 页面只加载一次样式；独立渲染函数保留自带样式，供报告和无浏览器验证使用。
    board_view = partial(render_board, include_styles=False)
    status_view = partial(render_status, include_styles=False)
    analysis_view = partial(render_analysis, include_styles=False)
    ui_html = partial(gr.HTML, padding=False)

    def heading(kicker, title, description):
        return (
            f'<div class="r2048-heading"><span class="r2048-eyebrow">{kicker}</span>'
            f"<h2>{title}</h2><p>{description}</p></div>"
        )

    css = _STYLE.removeprefix("<style>").removesuffix("</style>")
    css += Path(__file__).with_name("workspace.css").read_text(encoding="utf-8")
    theme = gr.themes.Soft(
        primary_hue="emerald",
        neutral_hue="slate",
        radius_size="lg",
        font=["Segoe UI", "Microsoft YaHei", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#f5f6f1",
        body_text_color="#24352e",
        body_text_color_subdued="#64746b",
        block_label_background_fill="transparent",
        block_label_text_color="#24352e",
        block_label_padding="0",
        block_label_margin="0",
        block_label_text_size="13px",
        input_border_color="#dce4dc",
        input_border_width="1px",
        input_shadow="none",
        input_shadow_focus="0 0 0 2px #86b69b",
        button_secondary_shadow="none",
        button_secondary_shadow_hover="none",
        block_title_background_fill="transparent",
        block_title_text_color="#24352e",
        block_title_padding="0",
    )
    # 棋盘沿用标准 2048 配色，控件也统一为纸面主题，避免系统暗色把输入区变成另一套色板。
    theme.set(
        **{
            key: getattr(theme, key[:-5])
            for key in vars(theme)
            if key.endswith("_dark") and key[:-5] in vars(theme)
        }
    )
    js = Path(__file__).with_name("workspace.js").read_text(encoding="utf-8")
    with gr.Blocks(title="re-2048 · 对局工作台", theme=theme, css=css, js=js) as demo:
        ui_html(
            '<header class="r2048-brand"><div class="r2048-logo">2048</div><div>'
            "<h1>对局工作台</h1><p>re-2048 / 探索局面，理解每一步决策</p></div></header>"
        )
        models = inventory.list_models()
        with gr.Column(elem_id="r2048-session"):
            ui_html(
                heading(
                    "SESSION",
                    "模型与会话",
                    "选择模型和运行设备，连接后开始对局。更换模型或设备后请重新连接。",
                )
            )
            with gr.Row(elem_id="r2048-connection-row"):
                model_box = gr.Dropdown(
                    choices=[entry.label() for entry in models],
                    value=inventory.default_model(models) or None,
                    label="模型",
                    scale=6,
                    min_width=240,
                    elem_classes=["r2048-session-field"],
                )
                device_box = gr.Dropdown(
                    choices=[("CPU", "cpu"), ("CUDA · GPU", "cuda")],
                    value="cpu",
                    label="运行设备",
                    scale=1,
                    min_width=130,
                    elem_classes=["r2048-session-field"],
                )
                with gr.Column(scale=2, min_width=300, elem_id="r2048-connection-actions"):
                    connect = gr.Button("连接 / 重新连接", variant="primary")
                    refresh_models = gr.Button("刷新模型列表")
            ui_html(
                '<div class="r2048-settings-help">模型支持输入关键词筛选，名称包含训练目录与目标。'
                "更换模型或设备会重建会话，请先保存需要保留的棋谱。</div>"
            )
            with gr.Column(elem_id="r2048-settings"):
                ui_html('<div class="r2048-settings-title">搜索与执行设置</div>')
                with gr.Row(elem_id="r2048-settings-grid"):
                    strategy_box = gr.Dropdown(
                        choices=list(STRATEGY_LABELS),
                        value="搜索（NN-MCTS）",
                        label="决策策略",
                        scale=2,
                        min_width=200,
                        elem_classes=["r2048-session-field"],
                    )
                    sims_box = gr.Number(
                        value=256,
                        precision=0,
                        minimum=0,
                        label="模拟预算",
                        min_width=130,
                        elem_classes=["r2048-session-field"],
                    )
                    time_box = gr.Number(
                        value=0,
                        precision=0,
                        minimum=0,
                        label="搜索限时 / ms",
                        min_width=130,
                        elem_classes=["r2048-session-field"],
                    )
                    interval_box = gr.Number(
                        value=2000,
                        precision=0,
                        minimum=100,
                        label="自动落子间隔 / ms",
                        min_width=150,
                        elem_classes=["r2048-session-field"],
                    )
                ui_html(
                    '<div class="r2048-settings-help">模拟预算与搜索限时设为 0 表示不限；'
                    "落子间隔控制最快速度，最小 100 ms。参数在下一次分析或落子时生效。</div>"
                )
                include_resume = gr.Checkbox(value=False, label="显示续跑 checkpoint（last.pt）")

        with gr.Tabs(elem_id="r2048-tabs") as tabs:
            with gr.Tab("对局与分析", id="play"):
                status_html = ui_html(status_view(None))
                note_html = ui_html(elem_id="r2048-feedback")
                with gr.Row(elem_id="r2048-workspace"):
                    with gr.Column(scale=5, min_width=300, elem_id="r2048-play"):
                        ui_html(
                            heading(
                                "BOARD", "当前局面", "绿色边框标记最近生成的数字块。坐标从左上角 a1 开始。"
                            )
                        )
                        with gr.Row():
                            newgame_btn = gr.Button("新开一局", min_width=110)
                            auto_start = gr.Button("▶ 自动执行", variant="primary", min_width=110)
                            auto_stop = gr.Button("■ 停止", variant="stop", visible=False, min_width=110)
                        board_html = ui_html(board_view(np.zeros(16, dtype=np.uint8)), elem_id="r2048-board")
                        with gr.Row(elem_id="r2048-board-tools"):
                            ui_html(
                                '<div class="r2048-control-hint"><strong>手动落子</strong>方向按钮或键盘 ↑ ↓ ← →</div>'
                            )
                            with gr.Column(scale=0, min_width=158, elem_id="r2048-dpad"):
                                dpad_up = gr.Button("↑ 上", min_width=46, elem_id="r2048-up")
                                with gr.Row():
                                    dpad_left = gr.Button("← 左", min_width=46, elem_id="r2048-left")
                                    dpad_right = gr.Button("右 →", min_width=46, elem_id="r2048-right")
                                dpad_down = gr.Button("↓ 下", min_width=46, elem_id="r2048-down")
                        direction_buttons = [dpad_up, dpad_right, dpad_down, dpad_left]
                        with gr.Row(elem_id="r2048-actions"):
                            undo_btn = gr.Button("撤销一步", min_width=110)
                            clear_btn = gr.Button("清空局面", min_width=110)
                    with gr.Column(scale=7, min_width=300, elem_id="r2048-insight"):
                        ui_html(
                            heading(
                                "INSIGHT", "决策分析", "查看引擎选招及方向评分，策略头先验仅表示模型倾向。"
                            )
                        )
                        analyze_btn = gr.Button("分析当前局面", variant="primary")
                        analysis_html = ui_html(analysis_view(None))
                        with gr.Accordion("如何阅读指标", open=False):
                            gr.Markdown(
                                "**访问数**：搜索分配给该方向的次数。**Q**：含立即得分的搜索均值。"
                                "**r + A(u)**：立即得分与一步估值之和。**策略头先验**：模型的方向倾向。"
                                "引擎选招以标记为准；交互分析的批组成与离线评测不同。"
                            )
                        with gr.Row(elem_id="r2048-save"):
                            record_name = gr.Textbox(value="game", label="棋谱名称", scale=3, min_width=160)
                            save_btn = gr.Button("保存棋谱", scale=1, min_width=100)

            with gr.Tab("局面编辑", id="editor"), gr.Row(elem_id="r2048-editor-layout"):
                with gr.Column(scale=6, min_width=300, elem_id="r2048-editor-controls"):
                    ui_html(
                        heading("COMPOSE", "摆一个局面", "按坐标选择数字，右侧同步预览。空格表示无数字块。")
                    )
                    tile_boxes: list = []
                    face_choices = [("空", records.EMPTY_TOKEN)] + [
                        (str(1 << exponent), str(1 << exponent))
                        for exponent in range(1, records.MAX_EXPONENT + 1)
                    ]
                    with gr.Column(elem_id="r2048-editor-grid"):
                        for row in range(4):
                            with gr.Row():
                                for column in range(4):
                                    tile_boxes.append(
                                        gr.Dropdown(
                                            choices=face_choices,
                                            value=records.EMPTY_TOKEN,
                                            label=f"{chr(ord('a') + column)}{row + 1}",
                                            scale=1,
                                            min_width=0,
                                            filterable=False,
                                        )
                                    )
                    with gr.Row():
                        editor_new = gr.Button("清空编辑器")
                        editor_to_game = gr.Button("发送到对局", variant="primary")
                    editor_note = ui_html(
                        '<div class="r2048-note">数字上限 131072。发送后可在对局页分析或继续走棋。</div>'
                    )
                with gr.Column(scale=5, min_width=300, elem_id="r2048-editor-preview"):
                    ui_html(heading("PREVIEW", "局面预览", "检查数字与位置，确认后发送到对局工作台。"))
                    editor_board = ui_html(board_view(state.editor), elem_id="r2048-editor-board")

            with gr.Tab("棋谱复盘", id="replay"), gr.Row(elem_id="r2048-replay-layout"):
                with gr.Column(scale=5, min_width=300, elem_id="r2048-replay-controls"):
                    ui_html(
                        heading(
                            "REPLAY",
                            "回看每一步",
                            "载入已保存的棋谱，拖动进度查看局面，或从某个局面继续对局。",
                        )
                    )
                    record_entries = inventory.list_records()
                    record_box = gr.Dropdown(
                        choices=[entry.label() for entry in record_entries],
                        label="棋谱",
                        value=None,
                        info="在对局页保存棋谱后刷新列表",
                    )
                    with gr.Row():
                        replay_load = gr.Button("载入棋谱", variant="primary")
                        refresh_records = gr.Button("刷新列表")
                    replay_slider = gr.Slider(
                        minimum=0, maximum=1, value=0, step=1, label="记录位置", interactive=False
                    )
                    with gr.Row():
                        replay_prev = gr.Button("← 上一条")
                        replay_next = gr.Button("下一条 →")
                    replay_note = ui_html(
                        '<div class="r2048-note">尚未载入棋谱。记录位置包含开局与数字生成；实际步数会单独显示。</div>'
                    )
                    export_btn = gr.Button("从此局面继续对局")
                with gr.Column(scale=6, min_width=300, elem_id="r2048-replay-board"):
                    ui_html(heading("POSITION", "回放局面", "按棋谱还原数字块，保留每一步的真实位置。"))
                    replay_html = ui_html(
                        board_view(np.zeros(16, dtype=np.uint8)), elem_id="r2048-replay-view"
                    )

        ui_html(
            '<footer class="r2048-footer"><span>re-2048 · 对局 / 分析 / 复盘</span>'
            f"<details><summary>存储位置</summary>{html.escape(str(paths.output_root()))}</details></footer>"
        )

        # ---- 回调 -----------------------------------------------------------

        manual = [
            newgame_btn,
            analyze_btn,
            undo_btn,
            clear_btn,
            *direction_buttons,
            connect,
            model_box,
            device_box,
            refresh_models,
            include_resume,
            strategy_box,
            sims_box,
            time_box,
            interval_box,
            editor_to_game,
            export_btn,
        ]
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
                # 落子帧的分析属于移动前的根，不能把旧方向标在移动后的棋盘上。
                board_view(
                    board,
                    best_action=None if spawn is not None else (report or {}).get("best_action"),
                    spawn_index=spawn,
                ),
                status_view(current, settings or state.settings, running=running),
                analysis_view(report, context="上一步决策" if spawn is not None else "当前局面"),
                f'<div class="r2048-note" role="status">{html.escape(note)}</div>' if note else "",
                gr.update(visible=not running),
                gr.update(visible=running),
                *[gr.update(interactive=not running) for _ in manual],
            )

        def fail(error: Exception):
            result = list(view([]))
            result[3] = f'<div class="r2048-err" role="alert">{html.escape(str(error))}</div>'
            return tuple(result)

        def on_identity_change():
            return '<div class="r2048-note" role="status">选择已更改，请重新连接以使用当前模型与设备。</div>'

        def on_refresh_models(include_resume, selected):
            found = inventory.list_models(include_resume=include_resume)
            choices = [entry.label() for entry in found]
            return gr.update(
                choices=choices,
                value=selected if selected in choices else inventory.default_model(found) or None,
            )

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
                    "已连接；点「新开一局」开始对局，或继续操作当前局面"
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
                return f'<div class="r2048-ok" role="status">已写入 {html.escape(str(target))}</div>'
            except (SessionError, RuntimeError) as error:
                return f'<div class="r2048-err" role="alert">{html.escape(str(error))}</div>'

        def on_auto_start(strategy_label, simulations, time_ms, interval_ms):
            """自动执行：**一次生成器调用**里每一步流式推给界面，不轮询。

            节奏按 ADR 0037：`下一次移动 = max(上次移动 + 间隔, 搜索完成)`。所以这里在一步
            做完之后才补睡"距上次移动还差多少"——搜索永不被砍断，间隔只限制最快速度。
            """
            if state.auto_running:
                yield view([], "已经在自动执行")
                return
            if not state.last_state or state.last_state.get("empty"):
                yield view([], "盘面为空：请连接会话并新开一局")
                return
            if state.last_state.get("terminal"):
                yield view([], "已终局：请新开一局或撤销一步")
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
                    if state.last_state and state.last_state.get("terminal"):
                        break
                    remaining = interval - (time.monotonic() - state.last_move_at)
                    if remaining > 0:
                        time.sleep(remaining)
            except (SessionError, RuntimeError) as error:
                state.auto_running = False
                yield fail(error)
                return
            finally:
                state.auto_running = False
            yield view(
                [],
                "对局已结束" if state.last_state and state.last_state.get("terminal") else "自动执行已停止",
            )

        def on_auto_stop():
            state.auto_stop = True
            return '<div class="r2048-note" role="status">正在停止（走完当前这一步）…</div>'

        def on_editor_new():
            state.editor = np.zeros(records.BOARD_CELLS, dtype=np.uint8)
            return [*[gr.update(value=records.EMPTY_TOKEN) for _ in range(16)], board_view(state.editor)]

        def on_editor_preview(*values):
            state.editor = np.array(
                [
                    records.FACE_TO_EXPONENT[str(value)] if value and value != records.EMPTY_TOKEN else 0
                    for value in values
                ],
                dtype=np.uint8,
            )
            return board_view(state.editor)

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
                    if state.client is None:
                        state.attach("", "cpu", "search", 256, 0)
                    frames: list[dict] = []
                    for command in commands:
                        frames = state.call(command)
                return (
                    *view(frames, "已发送到对局页"),
                    gr.update(selected="play"),
                    '<div class="r2048-ok" role="status">局面已发送</div>',
                )
            except (SessionError, RuntimeError) as error:
                return (
                    *fail(error),
                    gr.update(),
                    f'<div class="r2048-err" role="alert">{html.escape(str(error))}</div>',
                )

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
            # 载入失败必须撤销旧回放，否则空预览旁的「继续对局」会发送上一份棋谱。
            state.replay = []
            state.replay_upto = 0
            if path is None:
                return (
                    gr.update(maximum=1, value=0, interactive=False),
                    board_view(np.zeros(16, dtype=np.uint8)),
                    '<div class="r2048-err" role="alert">请先选择棋谱；可以在对局页保存一份后刷新列表。</div>',
                )
            try:
                commands = records.load(path)
                result = records.apply(commands)
            except records.RecordError as error:
                return (
                    gr.update(maximum=1, value=0, interactive=False),
                    board_view(np.zeros(16, dtype=np.uint8)),
                    f'<div class="r2048-err" role="alert">{html.escape(str(error))}</div>',
                )
            state.replay = commands
            state.replay_upto = len(state.replay)
            steps = sum(1 for command in state.replay if isinstance(command, records.Move))
            board = result.board
            return (
                gr.update(maximum=max(1, len(state.replay)), value=len(state.replay), interactive=True),
                board_view(board, note=f"{path.name}（{steps} 步）"),
                (
                    f'<div class="r2048-note">已载入 {html.escape(path.name)}：'
                    f"{len(state.replay)} 条命令，{steps} 步</div>"
                ),
            )

        def on_replay_step(prefix) -> tuple[str, str]:
            if not state.replay:
                return board_view(np.zeros(16, dtype=np.uint8)), ""
            state.replay_upto = max(0, min(len(state.replay), int(prefix)))
            result = records.apply(state.replay[: state.replay_upto])
            return (
                board_view(result.board, note=f"前 {state.replay_upto} 条命令"),
                f'<div class="r2048-note">得分 {result.total_score}，已走 {result.move_count} 步</div>',
            )

        def on_replay_prev():
            return max(0, state.replay_upto - 1)

        def on_replay_next():
            return min(len(state.replay), state.replay_upto + 1)

        def on_export():
            """把回放到的局面**当成新开一局**发给计算页。

            只送 `clear` + `set`，**不送 `move`**：协议里的 `move` 会由引擎生成一个块，而棋谱
            里的 `move` 后面跟着的是显式的 `set`。把棋谱的 move 当协议命令发，每步会多生成
            一个块——盘面立刻就不是棋谱里那盘了。
            """
            if not state.replay:
                return (
                    *fail(SessionError("还没有载入棋谱")),
                    gr.update(),
                    '<div class="r2048-err" role="alert">请先载入棋谱</div>',
                )
            if state.auto_running:
                return (
                    *view([], "自动执行中：先点「停止」"),
                    gr.update(),
                    '<div class="r2048-note" role="status">请先停止自动执行</div>',
                )
            board = records.apply(state.replay[: state.replay_upto]).board
            try:
                with state.lock:
                    if state.client is None:
                        state.attach("", "cpu", "search", 256, 0)
                    frames = state.call("clear")
                    for index, exponent in enumerate(board):
                        if exponent:
                            face = records.EXPONENT_TO_FACE[int(exponent)]
                            frames = state.call(f"set {face} {records.index_to_coordinate(index)}")
                return (
                    *view(frames, f"已以第 {state.replay_upto} 条命令处的局面开新一局"),
                    gr.update(selected="play"),
                    '<div class="r2048-ok" role="status">局面已发送到对局页</div>',
                )
            except (SessionError, RuntimeError) as error:
                return (
                    *fail(error),
                    gr.update(),
                    f'<div class="r2048-err" role="alert">{html.escape(str(error))}</div>',
                )

        # ---- 绑定 -----------------------------------------------------------

        settings_inputs = [strategy_box, sims_box, time_box]
        refresh_models.click(on_refresh_models, inputs=[include_resume, model_box], outputs=[model_box])
        include_resume.change(on_refresh_models, inputs=[include_resume, model_box], outputs=[model_box])
        model_box.change(on_identity_change, outputs=[note_html])
        device_box.change(on_identity_change, outputs=[note_html])
        connect.click(
            on_connect,
            inputs=[model_box, include_resume, device_box, *settings_inputs],
            outputs=outputs,
        )
        newgame_btn.click(on_newgame, inputs=settings_inputs, outputs=outputs)
        # 分析和自走会流式持有一条请求；Gradio 默认会在所有输入/输出上画持续进度线，
        # 视觉上像整块控件被描边。运行状态由页面自己的 status 与反馈区呈现。
        analyze_btn.click(on_analyze, inputs=settings_inputs, outputs=outputs, show_progress="hidden")
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
        auto_start.click(
            on_auto_start,
            inputs=[*settings_inputs, interval_box],
            outputs=outputs,
            show_progress="hidden",
        )
        # 停止不能排到正在运行的生成器后面，否则按钮可见但永远点不动。
        auto_stop.click(on_auto_stop, outputs=[note_html], queue=False, concurrency_limit=None)

        editor_new.click(on_editor_new, outputs=[*tile_boxes, editor_board])
        for tile in tile_boxes:
            tile.input(on_editor_preview, inputs=tile_boxes, outputs=[editor_board])
        editor_to_game.click(on_editor_send, inputs=tile_boxes, outputs=[*outputs, tabs, editor_note])
        refresh_records.click(on_refresh_records, outputs=[record_box])
        replay_load.click(
            on_replay_load, inputs=[record_box], outputs=[replay_slider, replay_html, replay_note]
        )
        replay_slider.change(on_replay_step, inputs=[replay_slider], outputs=[replay_html, replay_note])
        replay_prev.click(on_replay_prev, outputs=[replay_slider])
        replay_next.click(on_replay_next, outputs=[replay_slider])
        export_btn.click(on_export, outputs=[*outputs, tabs, replay_note])

    return demo, state
