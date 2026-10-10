"""后台任务入口；不导入 Gradio，任务请求是持久化配置，不包含可执行命令。"""

from __future__ import annotations

import argparse
import math
import os
import select
import signal
import threading
import time
from pathlib import Path

import yaml

from r2048.fileio import read_json, write_json_atomic, write_text_atomic
from r2048.identity import file_lock
from r2048.jobs import JobDir
from r2048.task_control import TaskStopped, check_stop

KINDS = {"selfplay", "train", "assessment", "rating", "iterate", "resume_iteration", "anchor", "resume_batch"}


def validate_request(request: dict, root: Path):
    if request.get("kind") not in KINDS:
        raise ValueError("不支持的实验操作")
    config = request["config"]
    for settings in (
        config,
        config.get("selfplay", {}),
        config.get("iteration", {}),
        config.get("evaluation", {}),
        config.get("recipe", {}).get("train", {}),
    ):
        for name in (
            "samples",
            "simulations",
            "games",
            "epochs",
            "batch_size",
            "workers",
            "data_workers",
            "lanes",
            "in_flight",
            "lr",
        ):
            if name in settings and (not math.isfinite(float(settings[name])) or float(settings[name]) <= 0):
                raise ValueError(f"{name} 必须大于 0")
    if request["kind"] == "assessment" and not config.get("checkpoint"):
        raise ValueError("请选择完整模型")
    for name in ("checkpoint", "resume", "batch"):
        if config.get(name):
            path = Path(config[name]).resolve()
            if not path.is_relative_to(root) or not path.exists():
                raise ValueError(f"{name} 必须从输出根的有效清单选择")
    for path in config.get("files", []):
        if not Path(path).resolve().is_relative_to(root) or not Path(path).is_file():
            raise ValueError("评测记录必须从清单选择")
    if config.get("checkpoint"):
        from r2048.nn.lineage import resolve_lineage
        from r2048.nn.model import read_checkpoint_payload

        metadata = read_checkpoint_payload(config["checkpoint"])["metadata"]
        if Path(config["checkpoint"]).suffix == ".safetensor":
            from r2048.nn.lineage import file_sha256

            manifest = read_json(Path(config["checkpoint"]).parent / "manifest.json")
            if manifest["model_id"] != metadata["model_id"] or manifest["model_sha256"] != file_sha256(
                config["checkpoint"]
            ):
                raise ValueError("发布模型身份或 SHA256 与 manifest 不一致")
        if metadata.get("training_target") not in ("dual", "joint"):
            raise ValueError("请选择双头完整模型")
        if request["kind"] in ("train", "iterate"):
            resolve_lineage(config["checkpoint"], strict=True)
    if request["kind"] == "train":
        datasets = config.get("datasets", [])
        if not datasets or not any(float(item["weight"]) > 0 for item in datasets):
            raise ValueError("至少选择一个正权重数据集")
        for item in datasets:
            path = Path(item["path"]).resolve()
            if not path.is_relative_to(root) or not (path / "manifest.json").is_file():
                raise ValueError("请选择有效数据集")
            if (
                not math.isfinite(float(item["weight"]))
                or float(item["weight"]) < 0
                or int(item["limit"]) < 0
            ):
                raise ValueError("数据权重与 limit 不能为负")
    if "recipe" in config:
        from r2048.config import train_section_payload

        params = train_section_payload("WebUI 配方", config["recipe"].get("train", {}))
        if request["kind"] in ("train", "iterate"):
            from r2048.nn.model import architecture_conflict
            from r2048.nn.models import default_params, resolve

            if config.get("checkpoint"):
                conflict = architecture_conflict(
                    params.get("architecture", ""), params.get("architecture_params"), metadata
                )
                if conflict:
                    raise ValueError(conflict)
            architecture = params.get("architecture") or (
                metadata["architecture"] if config.get("checkpoint") else "smnet-b9c64dt"
            )
            defaults = default_params(resolve(architecture))
            if not 0 < params.get("val_fraction", 0.05) < 1:
                raise ValueError("默认双头发布流程需要非空验证集，验证比例必须介于 0 与 1")
            if {**defaults, **params.get("architecture_params", {})} != defaults:
                raise ValueError("训练尺寸必须匹配已登记的正式架构，才能完成默认发布")
            if (
                config.get("checkpoint")
                and {**defaults, **metadata.get("architecture_params", {})} != defaults
            ):
                raise ValueError("父模型尺寸未登记正式代号，不能用于默认训练发布")


def run_request(directory: Path, request: dict) -> dict:
    from r2048.iterate import IterateConfig, _step_publish, _step_selfplay, run_iteration
    from r2048.nn.training import DatasetSpec
    from r2048.runtime.selfplay import pin_cpu_threads
    from r2048.workflows.assessment import evaluate_group, ingest_and_rate
    from r2048.workflows.dual_training import train_dual

    pin_cpu_threads()
    config = request["config"]
    work = directory / "work"
    work.mkdir(parents=True, exist_ok=True)
    recipe = work / "recipe.yaml"
    if "recipe" in config and not recipe.exists():
        recipe.write_text(yaml.safe_dump(config["recipe"], allow_unicode=True), encoding="utf-8")
    kind = request["kind"]
    if kind in ("iterate", "resume_iteration"):
        arguments = dict(config.get("iteration", {}))
        arguments.update(
            config_path=str(recipe) if kind == "iterate" else "",
            name="",  # 迭代使用自身的自动编号空间，队列任务身份单独保存在外层目录。
            output=str(work),
            init_checkpoint=config.get("checkpoint", ""),
        )
        if kind == "resume_iteration" or (work / "iteration.json").exists():
            # 重做只在这次提交开始时失效一次；停止再继续不能重新分配上游身份。
            if config.get("force") and not (work / "redo.json").exists():
                from r2048.iterate import STEPS

                target = Path(config.get("resume") or work)
                first = min(STEPS.index(stage) for stage in config.get("steps") or STEPS)
                for stage in STEPS[first:]:
                    (target / "steps" / f"{stage}.json").unlink(missing_ok=True)
                    (target / "requests" / f"{stage}.json").unlink(missing_ok=True)
                write_json_atomic(work / "redo.json", {"stages": list(STEPS[first:])})
            arguments = {
                "config_path": "",
                "resume": config.get("resume") or str(work),
                "steps": tuple(config["steps"]) if config.get("steps") else None,
                "force": False,
            }
        result = run_iteration(IterateConfig(**arguments))
        return {
            "iteration": result.directory,
            **result.steps.get("publish", {}),
            "rating_json": result.rating_json,
        }
    with JobDir(directory.name, kind, params=config, path=work) as job:
        check_stop()
        marker = work / "result.json"
        if marker.exists():
            return read_json(marker)
        if kind in ("selfplay", "resume_batch"):
            settings = dict(config.get("selfplay", {}))
            iteration = IterateConfig(
                config_path=str(recipe),
                name=directory.name,
                output=str(work),
                init_checkpoint=config.get("checkpoint", ""),
                **settings,
            )
            if kind == "resume_batch":
                batch = read_json(Path(config["batch"]) / "manifest.json")
                (Path(config["batch"]) / "control").unlink(missing_ok=True)
                write_json_atomic(work / "requests" / "selfplay.json", {"batch_id": batch["batch_id"]})
            iteration.on_progress = job.callback()
            result = _step_selfplay(iteration, {}, job)
            if result["state"] != "complete":
                raise TaskStopped("数据批已停止，可继续")
        elif kind == "train":
            trained = train_dual(
                work,
                directory.name,
                str(recipe),
                [DatasetSpec(**item) for item in config["datasets"]],
                config.get("checkpoint", ""),
                job.callback(),
            )
            check_stop()
            result = _step_publish(
                IterateConfig(config_path=str(recipe), output=str(work)), {"train": trained}, job
            )
        elif kind == "assessment":
            params = dict(config["evaluation"])
            track = config.get("track", "nn_mcts")
            baseline = f"mcts{params['simulations']}" if track == "nn_mcts" else "random"
            baseline_label = f"{directory.name}-{baseline}"
            baselines = [
                (
                    baseline_label,
                    {"policy": "mcts", "simulations": params["simulations"]}
                    if track == "nn_mcts"
                    else {"policy": "random"},
                )
            ]
            result = evaluate_group(
                work / "eval", [(directory.name, config["checkpoint"], track)], baselines, params, job
            )
            check_stop()
            result.update(
                ingest_and_rate(
                    work,
                    list(result["jsonl"].values()),
                    initialize={
                        "search" if track == "nn_mcts" else "pure": baseline_label,
                    },
                )
            )
        elif kind in ("rating", "anchor"):
            result = ingest_and_rate(work, config.get("files", []), anchors=config.get("anchors"))
        else:
            raise ValueError("不支持的操作")
        write_json_atomic(marker, result)
        job.finish(**result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    directory = Path(args.job).resolve()
    # pidfd 绑定具体进程实例，服务器被 SIGKILL 时也能识别退出，不受 PID 复用影响。
    server_pid = os.environ.get("R2048_SERVER_PID")
    if server_pid:
        if os.getpgrp() != os.getpid():
            raise RuntimeError("后台任务必须在独立进程组中启动，才能安全清理其子进程")
        try:
            server_fd = os.pidfd_open(int(server_pid))
        except ProcessLookupError:
            write_text_atomic(directory / "control", "stop")
            return

        def supervise_server():
            try:
                select.select([server_fd], [], [])
                write_text_atomic(directory / "control", "stop")
                print("WebUI 服务器已退出：停止并保存；60 秒后仍未退出则清理整个任务进程组", flush=True)
                # 合作式保存优先；异常关闭不能让昂贵计算无限期脱离服务器运行。
                time.sleep(60)
                with file_lock(directory / ".status.lock"):
                    write_json_atomic(
                        directory / "status.json",
                        {
                            **read_json(directory / "status.json"),
                            "state": "interrupted",
                            "error": "服务器异常退出，保存超时；已清理任务进程组，可从已提交断点继续",
                            "updated_at": time.time(),
                        },
                    )
                os.killpg(os.getpgrp(), signal.SIGKILL)
            finally:
                os.close(server_fd)

        threading.Thread(target=supervise_server, daemon=True).start()
    request = read_json(directory / "request.json")
    if request["version"] != 1 or request["submission_key"] != args.token:
        raise ValueError("任务请求身份不一致")
    # 等待父进程提交启动记录，避免极短任务的结束状态被启动状态覆盖。
    for _ in range(100):
        status = read_json(directory / "status.json")
        if status.get("pid") == os.getpid() and status["state"] in ("running", "stopping"):
            break
        time.sleep(0.05)
    else:
        return
    try:
        with file_lock(directory / ".worker.lock", blocking=False):
            result = run_request(directory, request)
            write_json_atomic(directory / "result.json", result)
            check_stop()
            state, error = "finished", ""
    except (TaskStopped, InterruptedError) as failure:
        state, error = "stopped", str(failure)
    except Exception as failure:  # noqa: BLE001 - 子进程边界必须持久化任意工作流错误。
        import traceback

        traceback.print_exc()
        state, error = "failed", str(failure)
    with file_lock(directory / ".status.lock"):
        write_json_atomic(
            directory / "status.json",
            {
                **read_json(directory / "status.json"),
                "state": state,
                "error": error,
                "updated_at": time.time(),
            },
        )


if __name__ == "__main__":
    main()
