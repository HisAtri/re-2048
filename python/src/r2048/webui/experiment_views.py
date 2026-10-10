"""把工作流记录转换为界面说明，避免把机器字段直接堆给用户。"""

STATES = {
    "queued": "排队中",
    "running": "运行中",
    "stopping": "正在保存并停止",
    "stopped": "已停止",
    "finished": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
    "interrupted": "运行中断",
}
KINDS = {
    "selfplay": "跑谱",
    "train": "双头训练",
    "assessment": "评测并评分",
    "rating": "导入评测结果",
    "iterate": "完整迭代",
    "anchor": "设置评分基准",
    "resume_iteration": "继续迭代",
    "resume_batch": "继续跑谱",
}
STAGES = {"selfplay": "跑谱", "train": "训练", "publish": "发布", "evaluate": "评测", "rate": "评分"}
STAGES.update(
    {
        "generate": "跑谱",
        "selfplay-workers": "多进程跑谱",
        "eval": "评测",
        "value": "训练价值头",
        "policy": "训练策略头",
    }
)


def protocol_label(protocol, identity):
    names = {
        "nn_mcts": "模型搜索",
        "mcts": "纯 MCTS",
        "nn_policy": "策略头落子",
        "nn_value": "价值头选招",
        "random": "随机落子",
    }
    parts = [names.get(protocol.get("track"), "评测")]
    if protocol.get("simulations") is not None:
        parts.append(f"每步 {protocol['simulations']} 次")
    if protocol.get("device"):
        parts.append("显卡" if protocol["device"] == "cuda" else "处理器")
    # 摘要不能覆盖所有可比性字段，保留短编号区分同摘要的不同协议。
    return " · ".join(parts) + f"（{identity[:8]}）"


def table(rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "| 项目 | 配置 |\n| --- | --- |\n" + "\n".join(
        f"| {cell(key)} | {cell(value)} |" for key, value in rows
    )


def request_summary(request):
    kind, config = request["kind"], request["config"]
    rows = [("操作", KINDS.get(kind, kind))]
    if "checkpoint" in config:
        from pathlib import Path

        rows.append(
            (
                "教师模型" if kind in ("selfplay", "resume_batch") else "父模型 / 评测模型",
                Path(config["checkpoint"]).name
                if config["checkpoint"]
                else "从头训练"
                if kind == "train"
                else "纯 MCTS",
            )
        )
    generation = config.get("selfplay", config.get("iteration"))
    if generation:
        rows.extend(
            [
                ("目标样本数", f"{generation['samples']:,}"),
                ("每步搜索次数", generation["simulations"]),
                ("跑谱随机种子", generation["data_seed"]),
                ("同时运行的跑谱进程", generation["data_workers"]),
                ("最大推理批大小", generation["max_batch"]),
            ]
        )
        rows.extend(
            [
                ("同时进行的对局 / 每局等待请求", f"{generation['lanes']} / {generation['in_flight']}"),
                (
                    "树内局面 / 每局面续弈次数",
                    f"{generation['tree_nodes']} / {generation['tree_label_plays']}",
                ),
                (
                    "教师推理设备",
                    "处理器（纯 MCTS）"
                    if not config.get("checkpoint")
                    else "显卡（CUDA）"
                    if generation["device"] == "cuda"
                    else "处理器（CPU）",
                ),
            ]
        )
    if "recipe" in config and kind in ("train", "iterate"):
        train = config["recipe"]["train"]
        if "recipe_id" in config:
            rows.append(("训练配方 ID", config["recipe_id"]))
        if "architecture" in train:
            rows.append(("模型架构 ID", train["architecture"]))
        rows.extend(
            [
                ("每个头训练轮数", train["epochs"]),
                ("每批训练样本数", train["batch_size"]),
                ("学习率", train["lr"]),
                ("验证集比例 / 划分种子", f"{train['val_fraction']} / {train['split_seed']}"),
                ("训练设备", "显卡（CUDA）" if train["device"] == "cuda" else "处理器（CPU）"),
            ]
        )
    for spec in config.get("datasets", []):
        from pathlib import Path

        rows.append(
            (
                Path(spec["path"]).name,
                f"权重 {spec['weight']}；使用前 {spec['limit']:,} 条"
                if spec["limit"]
                else f"权重 {spec['weight']}；使用全部数据",
            )
        )
    evaluation = config.get("evaluation")
    if evaluation:
        rows.append(
            (
                "选招方式",
                {"nn_mcts": "模型辅助搜索", "nn_policy": "策略头直接落子", "nn_value": "价值头选招"}[
                    config["track"]
                ],
            )
        )
        rows.extend(
            [
                ("评测局数", evaluation["games"]),
                ("每步搜索次数", evaluation["simulations"]),
                ("评测随机种子", evaluation["seed"]),
                ("最大推理批大小", evaluation["max_batch"]),
                ("同时进行的对局 / 每局等待请求", f"{evaluation['lanes']} / {evaluation['in_flight']}"),
                ("推理设备", "显卡（CUDA）" if evaluation["device"] == "cuda" else "处理器（CPU）"),
            ]
        )
    if kind == "iterate":
        rows.extend(
            [
                ("评测局数", generation["eval_games"]),
                ("评测每步搜索次数", generation["eval_simulations"]),
                ("执行顺序", "跑谱 → 双头训练 → 发布 → 评测 → 评分"),
            ]
        )
    if "files" in config:
        rows.append(("导入评测记录", f"{len(config['files'])} 份"))
    if "resume" in config:
        rows.extend(
            [
                ("历史迭代", config["resume"]),
                ("执行阶段", "、".join(STAGES.get(s, s) for s in config["steps"]) or "恢复原阶段选择"),
                ("执行方式", "重做，并使下游记录失效" if config["force"] else "从断点继续"),
            ]
        )
    if "batch" in config:
        rows.append(("继续的数据批", config["batch"]))
    for track, identity in config.get("anchors", {}).items():
        rows.append(("搜索评分基准" if track == "search" else "直接落子评分基准", identity))
    for resolution in config.get("performance", []):
        rows.append(("性能参数来源", resolution["reason"]))
        names = {
            "workers": "进程数",
            "lanes": "并行对局数",
            "in_flight": "在途请求数",
            "max_batch": "最大推理批大小",
            "batch_size": "训练批大小",
        }
        sources = {"explicit": "手动指定", "benchmark": "本机实测", "default": "默认值"}
        rows.append(
            (
                "参数选择",
                "；".join(f"{names[key]}：{sources[value]}" for key, value in resolution["sources"].items()),
            )
        )
    return "### 确认本次操作\n\n" + table(rows)


def task_summary(record):
    rows = [
        ("任务", record["task_id"]),
        ("状态", STATES.get(record["state"], record["state"])),
        ("操作", KINDS.get(record["request"]["kind"], record["request"]["kind"])),
    ]
    progress = record.get("progress", {})
    if "done" in progress:
        units = {"samples": "条样本", "games": "局", "epochs": "轮", "steps": "批"}
        unit = units.get(progress.get("unit"), "")
        total = progress.get("total")
        rows.append(
            (
                "完成进度",
                f"{progress['done']:,}" + (f" / {total:,}" if total is not None else "") + f" {unit}",
            )
        )
        if progress.get("rate") is not None:
            rows.append(("运行速度", f"{progress['rate']:,.2f} {unit}/秒"))
        if progress.get("eta") is not None:
            rows.append(("预计剩余时间", f"{progress['eta'] / 60:.1f} 分钟"))
    for key, label in (("train_loss", "训练误差"), ("val_loss", "验证误差")):
        if key in progress.get("extra", {}):
            rows.append((label, progress["extra"][key]))
    extra = progress.get("extra", {})
    if "simulations_per_second" in extra:
        rows.append(("搜索速度", f"{extra['simulations_per_second']:,.0f} 次模拟/秒"))
        rows.append(("已完成搜索", f"{extra['simulations']:,} 次模拟"))
        if "search_share" in extra:
            rows.append(("CPU 搜索与标注耗时占比", f"{100 * extra['search_share']:.1f}%（各 worker 平均）"))
        if not progress.get("done"):
            rows.append(("样本提交", "首批棋局尚未结束；搜索进度见上方，完整棋局结束后提交样本"))
    if "epoch" in extra:
        rows.append(("训练轮次", f"{extra['epoch']} / {extra['epochs']}"))
    if "samples_per_second" in extra:
        rows.append(("样本吞吐", f"{extra['samples_per_second']:,.1f} 条/秒"))
    # 仅解释已知字段，避免把整份内部状态作为默认界面堆给用户。
    for key, label in (
        ("stage", "当前阶段"),
        ("samples", "已生成样本"),
        ("samples_per_second", "每秒样本数"),
        ("epoch", "当前训练轮次"),
        ("games", "已完成局数"),
        ("loss", "训练误差"),
    ):
        if key in progress:
            rows.append(
                (label, STAGES.get(progress[key], progress[key]) if key == "stage" else progress[key])
            )
    model = record.get("model_identity", {})
    result = record.get("result", {})
    for key, label in (
        ("batch_id", "数据批名称"),
        ("samples", "已保存样本数"),
        ("episodes", "已完成对局"),
        ("ingested", "新增评分记录"),
        ("pool_rows", "评分池总记录"),
    ):
        if key in result:
            rows.append((label, result[key]))
    for key, label in (
        ("model_id", "模型名称"),
        ("current_dataset_samples", "本轮数据规模"),
        ("inherited_dataset_samples", "父模型数据规模"),
        ("dataset_samples", "累计去重数据规模"),
    ):
        if model.get(key) is not None:
            value = model[key]
            rows.append((label, f"{value:,}" if isinstance(value, int) else value))
    if model.get("parent_models"):
        rows.append(
            (
                "父模型",
                "、".join(
                    str(parent.get("model_id", parent)) if isinstance(parent, dict) else str(parent)
                    for parent in model["parent_models"]
                ),
            )
        )
    if model.get("data_lineage"):
        rows.append(("数据来源", "完整" if model["data_lineage"].get("complete") else "存在缺口"))
        for issue in model["data_lineage"].get("issues", []):
            rows.append(("来源缺口", issue))
    if record.get("error"):
        rows.append(("失败原因", record["error"]))
    return "### 任务详情\n\n" + table(rows)
