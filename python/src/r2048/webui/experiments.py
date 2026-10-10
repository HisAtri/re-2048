"""实验页面只生成结构化请求；计算、状态和产物由后台工作流维护。"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import gradio as gr
import yaml

from r2048 import paths
from r2048.fileio import read_json
from r2048.webui import inventory
from r2048.webui.experiment_views import KINDS, STATES, protocol_label, request_summary, task_summary
from r2048.webui.tasks import TERMINAL
from r2048.webui.worker import validate_request
from r2048.workflows.assessment import rating_report


def recipes() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[4] / "configs"
    return {path.name: path for path in sorted(root.glob("*.yaml"))}


def ratings_table():
    report = rating_report()
    rows = []
    for track, groups in report["tracks"].items():
        for group in groups:
            for identity, stats in group["models"].items():
                rows.append(
                    [
                        "搜索评测" if track == "search" else "直接选招",
                        protocol_label(group["protocol"] or {}, group["protocol_hash"]),
                        identity,
                        stats["rating"],
                        f"{stats['rating'] - stats['ci95']:.1f} ～ {stats['rating'] + stats['ci95']:.1f}"
                        if stats["ci95"] is not None
                        else None,
                        stats["n"],
                        stats["mean_score"],
                    ]
                )
    missing = [line for line in ("search", "pure") if line not in report["anchors"].get("tracks", {})]
    if missing:
        from r2048.workflows.rating import load_pool

        groups = {}
        for row in load_pool(paths.output_root() / "rating/pool.jsonl"):
            if row["track"] in missing:
                groups.setdefault((row["track"], row["protocol_hash"], row["model_id"]), []).append(row)
        for key, games in sorted(groups.items()):
            rows.append(
                [
                    "搜索评测" if key[0] == "search" else "直接选招",
                    protocol_label(games[0]["protocol"], key[1]),
                    key[2],
                    None,
                    None,
                    len(games),
                    sum(game["score"] for game in games) / len(games),
                ]
            )
    return rows, "尚未设置评分基准：" + "、".join(
        "搜索评测" if line == "search" else "直接选招" for line in missing
    ) if missing else "仅比较相同评测协议下的评分。基准可在“设置评分基准”中更换。"


def bind_experiments(state, play_model, workspace_tabs, compute_controls):
    manager = state.tasks
    subscriptions = {}

    def catalogs():
        models = {
            entry.label(): entry
            for entry in inventory.list_models()
            if entry.training_target in ("dual", "joint")
        }
        batches = {entry.batch_id: entry for entry in inventory.list_batches()}
        iterations = {
            str(path.parent.relative_to(paths.output_root())): path.parent
            for path in paths.output_root().rglob("iteration.json")
            if ".publish-" not in str(path)
        }
        evaluations = {
            str(path.relative_to(paths.output_root())): path
            for path in paths.output_root().rglob("*.jsonl")
            if path.parent.name == "eval" or "eval" in path.parts
        }
        return models, batches, iterations, evaluations

    def model_choices(entries, official=False):
        return [
            (entry.model_id or entry.path.name, label)
            if entry.path.suffix == ".safetensor"
            else (f"{entry.model_id or entry.path.stem}（训练产物）", label)
            for label, entry in entries.items()
            if not official or entry.path.suffix == ".safetensor"
        ]

    def batch_choices(entries, stopped=False):
        states = {
            "complete": "已完成",
            "finished": "已完成",
            "completed": "已完成",
            "paused": "已暂停",
            "stopped": "已停止",
            "running": "生成中",
        }
        return [
            (f"{label} · {entry.samples:,} 条 · {states.get(entry.state, entry.state)}", label)
            for label, entry in entries.items()
            if not stopped or entry.state in ("paused", "stopped")
        ]

    models, batches, iterations, evaluations = catalogs()
    preset_files = recipes()
    published = model_choices(models, official=True)
    gr.Markdown(
        "## 实验工作台\n任务在后台执行，关闭浏览器不影响计算。长任务依次运行，当前任务计算期间对局推理暂停，编辑与复盘保留。"
    )
    draft = gr.State(None)
    device_choices = [("显卡（CUDA）", "cuda"), ("处理器（CPU）", "cpu")]

    def training_form():
        with gr.Row():
            preset = gr.Dropdown(
                list(preset_files),
                value=next(iter(preset_files), None),
                label="训练配方",
            )
            device = gr.Dropdown(device_choices, value="cuda", label="训练设备")
        with gr.Row():
            epochs = gr.Number(8, precision=0, label="每个头训练轮数")
            batch = gr.Number(4096, precision=0, label="每批训练样本数", interactive=False)
            lr = gr.Number(0.0003, label="学习率")
        with gr.Accordion("高级训练设置", open=False):
            validation = gr.Number(0.05, label="验证集比例")
            split = gr.Number(0, precision=0, label="训练与验证划分的随机种子")
        automatic = gr.Checkbox(True, label="自动使用本机实测的训练批大小")
        automatic.change(
            lambda enabled: gr.update(interactive=not enabled),
            inputs=[automatic],
            outputs=[batch],
            queue=False,
        )
        return {0: preset, 1: device, 2: epochs, 3: batch, 4: lr, 16: validation, 17: split, 19: automatic}

    def generation_form():
        with gr.Row():
            samples = gr.Number(1000000, precision=0, label="目标样本数")
            sims = gr.Number(64, precision=0, label="每步搜索次数")
            workers = gr.Number(4, precision=0, label="同时运行的跑谱进程数", interactive=False)
        with gr.Accordion("高级跑谱设置", open=False):
            seed = gr.Number(20261010, precision=0, label="随机种子")
            with gr.Row():
                lanes = gr.Number(64, precision=0, label="每个进程同时进行的对局数", interactive=False)
                flight = gr.Number(8, precision=0, label="每局同时等待的推理请求数", interactive=False)
            tree = gr.Number(2, precision=0, label="每次搜索采集的树内局面数")
            plays = gr.Number(8, precision=0, label="每个树内局面的续弈次数")
        automatic = gr.Checkbox(True, label="自动使用本机跑谱性能画像（实际值见配置摘要）")
        automatic.change(
            lambda enabled: [gr.update(interactive=not enabled) for _ in range(3)],
            inputs=[automatic],
            outputs=[workers, lanes, flight],
            queue=False,
        )
        return {
            5: samples,
            6: sims,
            7: seed,
            8: workers,
            9: lanes,
            10: flight,
            11: tree,
            12: plays,
            18: automatic,
        }

    def evaluation_form(parallel=True):
        with gr.Row():
            games = gr.Number(400, precision=0, label="评测局数")
            sims = gr.Number(64, precision=0, label="评测每步搜索次数")
        result = {13: games, 14: sims}
        with gr.Accordion("高级评测设置", open=False):
            result[15] = gr.Number(20262053, precision=0, label="评测随机种子")
            if parallel:
                result[9] = gr.Number(64, precision=0, label="同时进行的对局数", interactive=False)
                result[10] = gr.Number(8, precision=0, label="每局同时等待的推理请求数", interactive=False)
        if parallel:
            automatic = gr.Checkbox(True, label="自动使用本机评测性能画像（实际值见配置摘要）")
            automatic.change(
                lambda enabled: [gr.update(interactive=not enabled) for _ in range(2)],
                inputs=[automatic],
                outputs=[result[9], result[10]],
                queue=False,
            )
            result[20] = automatic
        return result

    # 操作页独占宽度；任务管理是另一个页面，避免长期占用表单的一半空间。
    with gr.Tabs() as experiment_tabs:
        with gr.Tab("跑谱", id="generate"):
            gr.Markdown("生成用于训练的对局数据。数据批名称由系统分配。")
            teacher = gr.Dropdown(["纯 MCTS", *published], value="纯 MCTS", label="如何生成对局")
            generation_controls = generation_form()
            generation_device = gr.Dropdown(
                device_choices, value="cuda", label="教师模型推理设备", visible=False
            )
            generation_controls[1] = generation_device
            generate_preview = gr.Button("检查跑谱配置")
            with gr.Accordion("继续未完成的数据批", open=False):
                stopped_batch = gr.Dropdown(
                    batch_choices(batches, stopped=True),
                    label="选择已停止的数据批",
                )
                gr.Markdown("沿用原教师与搜索设置；目标样本数、进程数和推理设备使用本页设置。")
                batch_preview = gr.Button("检查继续跑谱配置")
        with gr.Tab("双头训练", id="train"):
            gr.Markdown("依次训练价值头和策略头，合并后自动发布正式模型。")
            datasets = gr.CheckboxGroup(batch_choices(batches), label="选择训练数据")
            mix = gr.Dataframe(
                headers=["数据集", "使用权重", "使用前多少条（0=全部）"],
                datatype=["str", "number", "number"],
                value=[],
                type="array",
                label="各数据集的使用方式",
            )
            train_parent = gr.Dropdown(published, value=None, label="从哪个模型继续训练（留空从头训练）")
            training_controls = training_form()
            train_preview = gr.Button("检查训练配置")
        with gr.Tab("评测与评分", id="rating"):
            gr.Markdown("对比模型与基线，完成后自动计算评分。")
            eval_model = gr.Dropdown(
                model_choices(models), value=published[-1][1] if published else None, label="选择评测模型"
            )
            strategy = gr.Dropdown(
                [
                    ("模型辅助搜索（NN-MCTS）", "nn_mcts"),
                    ("策略头直接落子", "nn_policy"),
                    ("价值头选招", "nn_value"),
                ],
                value="nn_mcts",
                label="模型如何选招",
            )
            evaluation_controls = evaluation_form()
            evaluation_controls[1] = gr.Dropdown(device_choices, value="cuda", label="模型推理设备")
            gr.Markdown("搜索评测对比相同搜索次数的纯 MCTS；直接选招评测对比随机落子。")
            eval_preview = gr.Button("检查评测配置")
            with gr.Accordion("导入已有评测结果", open=False):
                files = gr.CheckboxGroup(list(evaluations), label="选择已完成的评测记录")
                import_preview = gr.Button("检查导入配置")
            with gr.Accordion("查看评分结果", open=False):
                score_table = gr.Dataframe(
                    headers=["评分类型", "评测协议", "模型", "评分", "95% 区间", "局数", "平均得分"],
                    interactive=False,
                    label="评分结果",
                )
                score_note = gr.Markdown()
                score_refresh = gr.Button("刷新评分")
                with gr.Accordion("设置评分基准", open=False):
                    gr.Markdown("选择评分池中的模型作为基准。更换后，该类型的评分会重新计算。")
                    anchor_track = gr.Dropdown(
                        [("搜索评测", "search"), ("直接选招", "pure")], value="search", label="评分类型"
                    )
                    anchor_identity = gr.Dropdown([], label="基准模型")
                    anchor_refresh = gr.Button("载入可选模型")
                    anchor_preview = gr.Button("检查评分基准设置")
        with gr.Tab("完整迭代", id="iterate"):
            gr.Markdown("一键执行：跑谱 → 双头训练 → 发布 → 评测 → 评分。以下设置仅用于本次迭代。")
            iteration_parent = gr.Dropdown(
                published, value=None, label="父模型（留空从头训练并用纯 MCTS 跑谱）"
            )
            gr.Markdown("### 1. 生成训练数据")
            iteration_controls = generation_form()
            gr.Markdown("### 2. 训练模型")
            iteration_controls.update(training_form())
            gr.Markdown("### 3. 评测新模型")
            iteration_controls.update(evaluation_form(parallel=False))
            gr.Markdown("跑谱与评测使用相同的对局并行设置；教师推理、训练和评测使用所选训练设备。")
            iterate_preview = gr.Button("检查完整迭代配置")
            with gr.Accordion("继续或重做历史迭代", open=False):
                history = gr.Dropdown(list(iterations), label="选择历史迭代")
                steps = gr.CheckboxGroup(
                    [
                        ("跑谱", "selfplay"),
                        ("训练", "train"),
                        ("发布", "publish"),
                        ("评测", "evaluate"),
                        ("评分", "rate"),
                    ],
                    label="执行阶段（留空恢复原选择）",
                )
                redo = gr.Checkbox(False, label="重做所选阶段，并使下游记录失效（旧发布保留）")
                iteration_preview = gr.Button("检查继续或重做配置")
        with gr.Tab("任务", id="tasks"):
            gr.Markdown("后台任务依次执行。关闭浏览器不会中止任务。")
            task_refresh = gr.Button("刷新任务与模型、数据清单")
            queue_table = gr.Dataframe(headers=["任务", "状态", "操作"], interactive=False, label="任务队列")
            task_box = gr.Dropdown([], label="选择任务查看进度与结果")
            detail = gr.Markdown("选择任务后查看进度和结果。", elem_classes=["r2048-experiment-summary"])
            stop = gr.Button("停止并保存", variant="stop", visible=False)
            resume = gr.Button("继续任务", visible=False)
            task_message = gr.Markdown()
            with gr.Accordion("运行日志", open=False):
                log = gr.Textbox(lines=10, max_lines=18, interactive=False, label="最近日志（最多100行）")
            with gr.Row():
                use_play = gr.Button("用于对局", visible=False)
                use_eval = gr.Button("用于评测", visible=False)
                use_parent = gr.Button("作为下一轮父模型", visible=False)

    with gr.Group() as review:
        summary = gr.Markdown(
            "设置参数后，点击所在页面的“检查配置”按钮。", elem_classes=["r2048-experiment-summary"]
        )
        submit = gr.Button("确认并加入后台队列", variant="primary", interactive=False)
        message = gr.Markdown()
        view_tasks = gr.Button("查看任务进度")
    view_tasks.click(lambda: gr.update(selected="tasks"), outputs=[experiment_tabs], queue=False)
    teacher.change(
        lambda model: gr.update(visible=model != "纯 MCTS"),
        inputs=[teacher],
        outputs=[generation_device],
        queue=False,
    )

    # 每个预览只读取本页组件。未参与本操作的字段使用固定默认值，绝不借用其他标签的配置。
    defaults = [
        next(iter(preset_files), None),
        "cuda",
        8,
        4096,
        0.0003,
        1000000,
        64,
        20261010,
        4,
        64,
        8,
        2,
        8,
        400,
        64,
        20262053,
        0.05,
        0,
        True,
        True,
        True,
    ]

    def prepare(
        kind,
        values,
        selected=None,
        rows=None,
        parent=None,
        track=None,
        record_files=None,
        history_label=None,
        selected_steps=None,
        force=False,
        anchor=None,
    ):
        try:
            (
                recipe_label,
                gpu,
                n_epochs,
                n_batch,
                lr,
                n_samples,
                sims,
                data_seed,
                n_workers,
                n_lanes,
                in_flight,
                nodes,
                plays,
                n_games,
                budget,
                assessment_seed,
                validation_fraction,
                partition_seed,
                automatic_generation,
                automatic_training,
                automatic_evaluation,
            ) = values
            current_models, current_batches, current_iterations, current_files = catalogs()
            recipe = {}
            if kind in ("train", "iterate"):
                recipe = yaml.safe_load(preset_files[recipe_label].read_text()) if recipe_label else {}
                recipe.setdefault("train", {}).update(
                    epochs=int(n_epochs),
                    batch_size=int(n_batch),
                    lr=float(lr),
                    device=gpu,
                    val_fraction=float(validation_fraction),
                    split_seed=int(partition_seed),
                )
            model = str(current_models[parent].path) if parent else ""
            generation = {
                "samples": int(n_samples),
                "simulations": int(sims),
                "data_seed": int(data_seed),
                "data_workers": int(n_workers),
                "lanes": int(n_lanes),
                "in_flight": int(in_flight),
                "tree_nodes": int(nodes),
                "tree_label_plays": int(plays),
                "device": gpu,
            }
            evaluation = {
                "games": int(n_games),
                "seed": int(assessment_seed),
                "simulations": int(budget),
                "lanes": int(n_lanes),
                "in_flight": int(in_flight),
                "device": gpu,
            }
            config = {}
            if kind in ("selfplay", "iterate"):
                config = {"checkpoint": model}
                if kind == "selfplay":
                    config["selfplay"] = generation
                else:
                    config["recipe"] = recipe
                    config["iteration"] = {
                        **generation,
                        "eval_games": int(n_games),
                        "eval_simulations": int(budget),
                        "eval_seed": int(assessment_seed),
                        "baselines": ["random", f"mcts{int(budget)}"],
                    }
            elif kind == "train":
                specs = []
                if len(rows or []) != len(selected or []) or {row[0] for row in rows or []} != set(
                    selected or []
                ):
                    raise ValueError("数据配置表必须完整对应选中的数据集，每个数据集只出现一次")
                for label, weight, limit in rows or []:
                    if label not in (selected or []):
                        raise ValueError("数据配置表必须对应选中的数据集")
                    specs.append(
                        {
                            "path": str(current_batches[label].path),
                            "weight": float(weight),
                            "limit": int(limit),
                        }
                    )
                config = {"datasets": specs, "checkpoint": model, "recipe": recipe}
            elif kind == "assessment":
                config = {"checkpoint": model, "track": track, "evaluation": evaluation}
            elif kind == "rating":
                if not record_files:
                    raise ValueError("请选择完整评测记录")
                config = {"files": [str(current_files[label]) for label in record_files]}
            elif kind == "anchor":
                if not anchor[1]:
                    raise ValueError("请选择评分池中的锚点模型")
                config = {"anchors": {anchor[0]: anchor[1]}}
            elif kind == "resume_iteration":
                config = {
                    "resume": str(current_iterations[history_label]),
                    "steps": selected_steps or [],
                    "force": force,
                }
            elif kind == "resume_batch":
                batch = current_batches[history_label]
                metadata = read_json(batch.path / "manifest.json")
                params = metadata.get("params", {})
                if metadata["policy"] != "mcts":
                    matches = [
                        entry for entry in current_models.values() if entry.model_id == metadata["model_id"]
                    ]
                    if len(matches) != 1:
                        raise ValueError("原教师模型不可唯一确认，不能继续该数据批")
                    model = str(matches[0].path)
                else:
                    model = ""
                generation.update(
                    simulations=metadata["simulations"],
                    data_seed=metadata["seed"],
                    tree_nodes=metadata["tree_nodes"],
                    tree_label_plays=int(metadata.get("tree_label_plays", generation["tree_label_plays"])),
                    tree_label_policy="top_visits"
                    if metadata.get("tree_label_policy", 1) == 1
                    else "depth_stratified",
                    tree_label_rollout="greedy" if metadata.get("tree_label_rollout", 1) == 1 else "uniform",
                )
                for source, destination in (
                    ("lanes", "lanes"),
                    ("in_flight_per_lane", "in_flight"),
                    ("tree_label_plays", "tree_label_plays"),
                    ("tree_label_policy", "tree_label_policy"),
                    ("tree_label_rollout", "tree_label_rollout"),
                    ("rollout_policy", "rollout_policy"),
                    ("search_threads", "search_threads"),
                    ("max_batch", "max_batch"),
                    ("c_puct", "c_puct"),
                ):
                    if source in params:
                        generation[destination] = params[source]
                for key in ("lanes", "in_flight", "max_batch", "search_threads", "tree_label_plays"):
                    if key in generation:
                        generation[key] = int(generation[key])
                config = {"batch": str(batch.path), "checkpoint": model, "selfplay": generation}
            if kind in ("train", "iterate"):
                # 配方 ID 与架构 ID 各自保留，避免展示别名掩盖不同配方或模型定义。
                config["recipe_id"] = recipe_label
            from r2048.performance import resolve

            performance = []
            if kind in ("selfplay", "resume_batch", "iterate"):
                settings = config["iteration"] if kind == "iterate" else config["selfplay"]
                original_device = settings["device"]
                if not config.get("checkpoint"):
                    settings["device"] = "cpu"
                settings["workers"] = None if automatic_generation else settings["data_workers"]
                for key in ("lanes", "in_flight", "max_batch"):
                    if kind != "resume_batch":
                        settings[key] = None if automatic_generation else settings.get(key, 16384)
                effective, info = resolve(
                    "selfplay",
                    settings,
                    config.get("checkpoint", ""),
                    mode="auto" if automatic_generation else "off",
                )
                settings.update({key: effective[key] for key in ("lanes", "in_flight", "max_batch")})
                settings["data_workers"] = effective["workers"]
                settings.pop("workers")
                if kind == "iterate":
                    settings["device"] = original_device
                performance.append(info)
            if kind == "assessment":
                settings = config["evaluation"]
                if automatic_evaluation:
                    settings["lanes"] = settings["in_flight"] = settings["max_batch"] = None
                settings["track"] = track
                effective, info = resolve(
                    "evaluate",
                    settings,
                    model,
                    mode="auto" if automatic_evaluation else "off",
                    default_values={"lanes": 64, "in_flight": 8},
                )
                settings.pop("track")
                settings.update({key: effective[key] for key in ("lanes", "in_flight", "max_batch")})
                performance.append(info)
            if kind in ("train", "iterate"):
                training = recipe["train"]
                settings = {**training, "batch_size": None if automatic_training else training["batch_size"]}
                effective, info = resolve(
                    "train", settings, model, recipe=training, mode="auto" if automatic_training else "off"
                )
                training["batch_size"] = effective["batch_size"]
                training["performance"] = info
                performance.append(info)
            if performance:
                config["performance"] = performance
            request = {"kind": kind, "config": config}
            validate_request(request, paths.output_root().resolve())
            prepared = {"request": request, "token": uuid.uuid4().hex}
            return prepared, request_summary(request), "请核对摘要。确认后，后台任务将使用这份配置。"
        except (ValueError, KeyError, TypeError, OSError) as error:
            return None, "配置检查未通过。", f"无法提交：{error}"

    def snapshot(kind, controls, extras, tuned_indices):
        indices = list(controls)

        def handler(*values):
            settings = list(defaults)
            for index, value in zip(indices, values):
                settings[index] = value
            prepared, description, note = prepare(kind, settings, **extras(values[len(indices) :]))
            updates = []
            if prepared:
                config = prepared["request"]["config"]
                generation = config.get("iteration", config.get("selfplay", {}))
                evaluation = config.get("evaluation", {})
                actual = {
                    8: generation.get("data_workers"),
                    9: generation.get("lanes", evaluation.get("lanes")),
                    10: generation.get("in_flight", evaluation.get("in_flight")),
                    3: config.get("recipe", {}).get("train", {}).get("batch_size"),
                }
                updates = [gr.update(value=actual[index]) for index in tuned_indices]
            else:
                updates = [gr.update() for _ in tuned_indices]
            return prepared, description, note, gr.update(interactive=bool(prepared)), *updates

        return handler

    previews = [
        (
            generate_preview,
            "selfplay",
            generation_controls,
            [teacher],
            lambda v: {"parent": None if v[0] == "纯 MCTS" else v[0]},
        ),
        (
            train_preview,
            "train",
            training_controls,
            [datasets, mix, train_parent],
            lambda v: {"selected": v[0], "rows": v[1], "parent": v[2]},
        ),
        (
            eval_preview,
            "assessment",
            evaluation_controls,
            [eval_model, strategy],
            lambda v: {"parent": v[0], "track": v[1]},
        ),
        (iterate_preview, "iterate", iteration_controls, [iteration_parent], lambda v: {"parent": v[0]}),
        (import_preview, "rating", {}, [files], lambda v: {"record_files": v[0]}),
        (
            iteration_preview,
            "resume_iteration",
            {},
            [history, steps, redo],
            lambda v: {"history_label": v[0], "selected_steps": v[1], "force": v[2]},
        ),
        (
            batch_preview,
            "resume_batch",
            generation_controls,
            [stopped_batch],
            lambda v: {"history_label": v[0]},
        ),
        (anchor_preview, "anchor", {}, [anchor_track, anchor_identity], lambda v: {"anchor": v}),
    ]
    for button, kind, controls, inputs, extra in previews:
        tuned_indices = [index for index in (8, 9, 10, 3) if index in controls]
        button.click(
            snapshot(kind, controls, extra, tuned_indices),
            inputs=[*controls.values(), *inputs],
            outputs=[draft, summary, message, submit, *[controls[index] for index in tuned_indices]],
        )

    def invalidate():
        return None, "参数已变化，请重新检查配置。", "", gr.update(interactive=False)

    # 浏览器的手动输入才使摘要失效；清单刷新不会清掉已冻结的请求。
    form_controls = {
        component for _, _, controls, inputs, _ in previews for component in [*controls.values(), *inputs]
    }
    for component in form_controls:
        component.input(invalidate, outputs=[draft, summary, message, submit], queue=False)
    for tab in experiment_tabs.children:
        tab.select(
            lambda selected=tab.label: (*invalidate(), gr.update(visible=selected != "任务")),
            outputs=[draft, summary, message, submit, review],
            queue=False,
        )

    def submit_task(prepared):
        if not prepared:
            return gr.update(), "请先预览有效配置。"
        try:
            identity = manager.submit(prepared["request"], prepared["token"])
            return gr.update(
                choices=[item["task_id"] for item in manager.list()], value=identity
            ), f"任务 {identity} 已入队；重复确认不会重复创建。"
        except (ValueError, OSError, RuntimeError) as error:
            return gr.update(), str(error)

    submit.click(submit_task, inputs=[draft], outputs=[task_box, message], queue=False)

    def dataset_rows(selected, existing):
        previous = {row[0]: row for row in existing or []}
        return [previous.get(label, [label, 1.0, 0]) for label in selected or []]

    datasets.change(
        lambda selected, existing: gr.update(value=dataset_rows(selected, existing)),
        inputs=[datasets, mix],
        outputs=[mix],
    )

    artifact_controls = [
        teacher,
        datasets,
        train_parent,
        eval_model,
        iteration_parent,
        history,
        files,
        stopped_batch,
        play_model,
    ]
    watch_outputs = [
        queue_table,
        detail,
        log,
        stop,
        resume,
        use_play,
        use_eval,
        use_parent,
        *compute_controls,
        *artifact_controls,
    ]

    def control_updates():
        return [
            gr.update(interactive=not state.compute_busy and not state.auto_running) for _ in compute_controls
        ]

    def artifact_updates(changed=False):
        updates = list(refresh(False)[1:10]) if changed else [gr.update() for _ in artifact_controls]
        updates[-1]["interactive"] = not state.compute_busy and not state.auto_running
        return updates

    def watch(identity, request: gr.Request = None):
        if not identity:
            yield (
                [],
                "选择任务后查看进度和结果。",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
                *[gr.update(visible=False) for _ in range(3)],
                *control_updates(),
                *artifact_updates(),
            )
            return
        previous = None
        artifact_id = None
        session = request.session_hash if request else "direct"
        token = uuid.uuid4().hex
        subscriptions[session] = token
        while True:
            if subscriptions.get(session) != token:
                return
            try:
                record = manager.detail(identity)
                rows = [
                    [
                        item["task_id"],
                        STATES.get(item["state"], item["state"]),
                        KINDS.get(read_json(Path(item["directory"]) / "request.json")["kind"], "后台任务"),
                    ]
                    for item in manager.list()
                ]
                text = record.pop("log", "")
                result = (rows, task_summary(record), text)
                fingerprint = json.dumps((result, state.compute_busy), sort_keys=True)
                if fingerprint != previous:
                    previous = fingerprint
                    produced = record.get("result", {}).get("model") or record.get("result", {}).get(
                        "batch_id"
                    )
                    refreshed = produced and produced != artifact_id
                    artifact_id = produced
                    yield (
                        *result,
                        gr.update(
                            visible=record["state"] in ("queued", "running", "stopping"),
                            value="取消排队" if record["state"] == "queued" else "停止并保存",
                        ),
                        gr.update(visible=record["state"] in ("stopped", "failed", "interrupted")),
                        *[gr.update(visible=bool(record.get("result", {}).get("model"))) for _ in range(3)],
                        *control_updates(),
                        *artifact_updates(refreshed),
                    )
                if record["state"] in TERMINAL and (
                    not state.compute_busy
                    or any(
                        item["task_id"] != identity and item["state"] == "running" for item in manager.list()
                    )
                ):
                    return
                time.sleep(1)
            except (OSError, ValueError) as error:
                yield (
                    [],
                    f"无法读取任务：{error}",
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    *[gr.update(visible=False) for _ in range(3)],
                    *control_updates(),
                    *artifact_updates(),
                )
                return

    subscription = task_box.change(
        watch,
        inputs=[task_box],
        outputs=watch_outputs,
        concurrency_limit=None,
        show_progress="hidden",
    )

    def action(identity, continuing=False):
        try:
            if continuing:
                manager.resume(identity)
            else:
                manager.stop(identity)
            return identity, "已请求继续任务。" if continuing else "已请求停止；计算会在安全边界保存退出。"
        except (ValueError, OSError) as error:
            return identity, str(error)

    stop.click(
        action, inputs=[task_box], outputs=[task_box, task_message], queue=False, concurrency_limit=None
    )
    resumed = resume.click(
        lambda identity: action(identity, True),
        inputs=[task_box],
        outputs=[task_box, task_message],
        queue=False,
        concurrency_limit=None,
    )
    resumed.then(
        watch, inputs=[task_box], outputs=watch_outputs, concurrency_limit=None, show_progress="hidden"
    )

    def refresh(select_active=True):
        current_models, current_batches, current_iterations, current_files = catalogs()
        official = model_choices(current_models, official=True)
        return (
            gr.update(
                choices=[item["task_id"] for item in manager.list()],
                **(
                    {
                        "value": next(
                            (item["task_id"] for item in manager.list() if item["state"] not in TERMINAL),
                            manager.list()[-1]["task_id"] if manager.list() else None,
                        )
                    }
                    if select_active
                    else {}
                ),
            ),
            gr.update(choices=["纯 MCTS", *official]),
            gr.update(choices=batch_choices(current_batches)),
            gr.update(choices=official),
            gr.update(choices=model_choices(current_models)),
            gr.update(choices=official),
            gr.update(choices=list(current_iterations)),
            gr.update(choices=list(current_files)),
            gr.update(choices=batch_choices(current_batches, stopped=True)),
            gr.update(choices=[entry.label() for entry in inventory.list_models()]),
            *ratings_table(),
        )

    refresh_outputs = [
        task_box,
        teacher,
        datasets,
        train_parent,
        eval_model,
        iteration_parent,
        history,
        files,
        stopped_batch,
        play_model,
        score_table,
        score_note,
    ]
    task_refresh.click(refresh, outputs=refresh_outputs)
    subscription.then(lambda: refresh(False), outputs=refresh_outputs)
    score_refresh.click(ratings_table, outputs=[score_table, score_note])

    def anchors(track):
        from r2048.workflows.rating import load_pool

        identities = sorted(
            {
                row["model_id"]
                for row in load_pool(paths.output_root() / "rating/pool.jsonl")
                if row["track"] == track
            }
        )
        return gr.update(choices=identities, value=identities[0] if identities else None)

    anchor_refresh.click(anchors, inputs=[anchor_track], outputs=[anchor_identity])

    def artifact(identity):
        record = manager.detail(identity)
        model = record.get("result", {}).get("model")
        if not model:
            raise ValueError("任务尚未产出正式模型")
        entries = inventory.list_models()
        entry = next(item for item in entries if str(item.path) == model)
        return entry, entries

    def to_play(identity):
        try:
            entry, entries = artifact(identity)
            return (
                gr.update(choices=[item.label() for item in entries], value=entry.label()),
                gr.update(selected="playworkspace"),
                "已选择产物，点击连接后使用；当前棋盘未改变。",
            )
        except (ValueError, StopIteration, OSError) as error:
            return gr.update(), gr.update(), str(error)

    def to_model(identity, father=False):
        try:
            entry, entries = artifact(identity)
            choices = model_choices(
                {item.label(): item for item in entries if item.training_target in ("dual", "joint")},
                official=father,
            )
            update = gr.update(choices=choices, value=entry.label())
            return (
                update,
                update if father else gr.update(),
                "已选择任务产物。",
                gr.update(selected="iterate" if father else "rating"),
            )
        except (ValueError, StopIteration, OSError) as error:
            return gr.update(), gr.update(), str(error), gr.update()

    use_play.click(to_play, inputs=[task_box], outputs=[play_model, workspace_tabs, message])
    use_eval.click(
        to_model, inputs=[task_box], outputs=[eval_model, iteration_parent, task_message, experiment_tabs]
    )
    use_parent.click(
        lambda identity: to_model(identity, True),
        inputs=[task_box],
        outputs=[train_parent, iteration_parent, task_message, experiment_tabs],
    )
    return refresh, refresh_outputs
