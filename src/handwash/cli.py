"""命令行入口：``handwash <子命令> [config=... | key.sub=value]``。

设计约定（CONTRIBUTING.md R7）
-------------------------------
* 子命令与 ``Makefile`` 的 target 一一对应，两边必须同步新增。
* 位置参数**只**用来传 ``key=value`` 形式的配置覆盖，与 Makefile 写法一致::

      handwash train config=configs/experiments/smoke.yaml train.epochs=2
      handwash assess --video path/to/demo.mp4 assess.min_total_duration_s=30

* 所有可调参数都必须来自配置，CLI **不新增**任何独立参数（除了文件路径这类
  天然属于"输入输出"的东西，例如 ``--video`` / ``--checkpoint``）。
* 异常统一在这里翻译成退出码与中文提示，库代码里不许 ``sys.exit``。
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from handwash import __version__
from handwash.core.config import load_config, parse_overrides
from handwash.errors import HandwashError
from handwash.logging import get_logger, setup_logging
from handwash.paths import checkpoint_path

__all__ = ["build_parser", "main", "split_argv"]

log = get_logger(__name__)

#: 合法的配置覆盖写法：等号左边是点分键名（不允许路径分隔符/盘符冒号）
_OVERRIDE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*=")

#: 退出码约定（写进 docs/CONTRIBUTING 的 CI 章节，脚本可依赖）
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_DATA = 3
EXIT_MODEL = 4
EXIT_TRAIN = 5
EXIT_PROTOCOL = 6


def split_argv(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """把参数分成 ``(普通参数, key=value 覆盖)``。

    为什么自己做：argparse 对"位置参数里混着 key=value"支持很差，
    而 ``key=value`` 又是团队约定的统一写法（Makefile 也这么传）。

    判定规则：形如 ``^[A-Za-z_][A-Za-z0-9_.]*=`` 的才算配置覆盖
    （即等号左边必须是合法的"点分键名"）。这样 ``config=configs/x.yaml``
    会被正确识别，而作为普通参数传入的绝对路径不会被误判。
    """
    plain: list[str] = []
    overrides: list[str] = []
    for item in argv:
        if not item.startswith("-") and _OVERRIDE_RE.match(item):
            overrides.append(item)
        else:
            plain.append(item)
    return plain, overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="handwash",
        description="六步洗手动作识别与完整性评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  handwash doctor\n"
            "  handwash prepare config=configs/data/kaggle.yaml\n"
            "  handwash train config=configs/experiments/smoke.yaml train.epochs=2\n"
            "  handwash evaluate --checkpoint outputs/run/models/best.pt\n"
            "  handwash infer --video demo.mp4 --checkpoint outputs/run/models/best.pt\n"
            "  handwash assess --video demo.mp4 --checkpoint outputs/run/models/best.pt\n"
            "  handwash camera config=configs/experiments/live_yolo_frame.yaml\n"
            "\n配置覆盖统一写作 key.sub=value（与 Makefile 一致）。"
        ),
    )
    parser.add_argument("--version", action="version", version=f"handwash {__version__}")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "替换基础配置文件（默认 configs/config.yaml）。"
            "若想叠加覆盖文件，请用位置参数 config=<file>。"
        ),
    )
    parser.add_argument("--log-level", default=None, help="日志级别：DEBUG/INFO/WARNING/ERROR")

    sub = parser.add_subparsers(dest="command", required=True, metavar="<子命令>")

    p_prepare = sub.add_parser("prepare", help="数据准备：扫描 / 划分 / 抽帧 / 生成 manifest")
    p_prepare.add_argument("--stage", choices=("all", "scan", "split", "frames"), default="all")

    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument("--epochs", type=int, default=None, help="覆盖 train.epochs（冒烟测试用）")
    p_train.add_argument("--run-name", default=None, help="覆盖 runtime.run_name（输出目录名）")
    p_train.add_argument("--no-synthetic", action="store_true", help="禁止使用显式配置的合成数据")

    p_eval = sub.add_parser("evaluate", help="评估：内部测试 + 跨场景")
    p_eval.add_argument("--checkpoint", default=None, help="checkpoint 路径（默认取 best.pt）")
    p_eval.add_argument("--splits", nargs="*", default=None, help="要评估的 split，例如 val test")

    p_infer = sub.add_parser("infer", help="推理单段视频，输出逐帧预测")
    p_infer.add_argument("--video", required=True, help="视频文件路径")
    p_infer.add_argument("--checkpoint", default=None, help="checkpoint 路径（默认取 best.pt）")

    p_assess = sub.add_parser("assess", help="WHO 完整性评估（漏步/乱序/时长）")
    p_assess.add_argument("--video", default=None, help="单个视频文件")
    p_assess.add_argument("--folder", default=None, help="整个目录的视频")
    p_assess.add_argument("--checkpoint", default=None, help="checkpoint 路径（默认取 best.pt）")
    p_assess.add_argument("--json", action="store_true", help="以 JSON 输出报告（供程序调用）")

    p_camera = sub.add_parser("camera", help="启动外接摄像头的实时 YOLO 分类网页")
    p_camera.add_argument("--checkpoint", default=None, help="权重路径；正式版默认 best.pt，--demo-exp 默认根目录 exp.pt")
    p_camera.add_argument("--port", type=int, default=8765, help="本机网页端口（默认 8765）")
    p_camera.add_argument("--demo-exp", action="store_true", help="使用根目录 exp.pt 七类演示模型")
    p_camera.add_argument(
        "--temporal-head",
        default=None,
        help="exp.pt 的 GRU 时序头；默认根目录 exp_temporal_head.pt 存在时自动加载",
    )
    p_camera.add_argument("--no-temporal-head", action="store_true", help="不加载时序头，只用 exp.pt 逐帧分类")

    sub.add_parser("doctor", help="环境与配置自检")
    sub.add_parser("config", help="打印最终生效的配置（含 config_hash）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口。返回退出码（不直接 sys.exit，便于测试调用）。"""
    raw = list(sys.argv[1:] if argv is None else argv)
    plain, override_items = split_argv(raw)

    parser = build_parser()
    # 把 key=value 从主解析器里摘掉后，再允许子命令自己解析剩余参数
    args, unknown = parser.parse_known_args(plain)
    if unknown:
        parser.error(f"无法识别的参数：{unknown}")

    setup_logging(args.log_level, force=True)

    try:
        overrides = parse_overrides(override_items)
        return _dispatch(args, overrides)
    except HandwashError as exc:
        # 已知错误：打印友好提示 + 退出码，不打 traceback（读起来清爽）
        log.error("%s", exc)
        return getattr(exc, "exit_code", EXIT_ERROR)
    except KeyboardInterrupt:
        log.warning("被用户中断")
        return 130
    except Exception:
        log.exception("发生未预期的错误（这是 bug，请把下面的堆栈贴到 issue 里）")
        return EXIT_ERROR


def _load(args: argparse.Namespace, overrides: dict) -> object:
    """加载配置，支持两种写法（语义不同，不要混用）：

    * ``--config <file>``：**替换**基础配置（该文件必须自洽）；
    * 位置参数 ``config=<file>``：把该文件作为**叠加层**叠在
      ``configs/config.yaml`` 之上（推荐，也是 Makefile 的用法）。

    这样 ``configs/experiments/*.yaml`` 只需要写"与默认不同的部分"，
    不必复制几十行公共配置 —— 复制是 bug 的温床（CONTRIBUTING.md R14）。

    ``config`` 本身不是 AppConfig 的字段，因此必须在合并前摘出来，
    否则严格校验会把它当成"未知键"报错（这正是严格性在起作用）。
    """
    overrides = dict(overrides)
    overlay = overrides.pop("config", None)
    if args.config:
        base: list[str] = [str(args.config)]
    else:
        base = ["configs/config.yaml"]
    if overlay is not None:
        base.append(str(overlay))
    return load_config(base, overrides=overrides)


def _dispatch(args: argparse.Namespace, overrides: dict) -> int:
    command = args.command

    if command == "doctor":
        from handwash.cli_doctor import run_doctor

        return run_doctor(_load(args, overrides))

    if command == "config":
        import yaml

        rc = _load(args, overrides)
        print(yaml.safe_dump(rc.to_dict(), allow_unicode=True, sort_keys=False))
        return EXIT_OK

    if command == "prepare":
        from handwash.pipelines.prepare import prepare

        rc = _load(args, overrides)
        result = prepare(rc, stage=args.stage)
        print(f"准备完成：{result.num_clips} 段视频 / {result.num_frames} 帧")
        if result.manifest_path:
            print(f"manifest：{result.manifest_path}")
        for split, info in result.splits.items():
            print(f"  {split:<8} {info['num_clips']:>5} 段 / {info['num_frames']:>7} 帧")
        if args.stage == "scan":
            print("下一步：handwash prepare --stage split")
        elif args.stage == "split":
            print("下一步：handwash prepare --stage frames")
        elif result.manifest_path:
            print("下一步：handwash train")
        return EXIT_OK

    if command == "train":
        from handwash.pipelines.train import train

        if args.run_name:
            overrides = {**overrides, "runtime": {**overrides.get("runtime", {}), "run_name": args.run_name}}
        rc = _load(args, overrides)
        result = train(rc, max_epochs=args.epochs, allow_synthetic=not args.no_synthetic)
        print(f"训练完成：best val macro-F1 = {result.best_metric:.4f}")
        print(f"checkpoint：{result.best_checkpoint}")
        if result.used_synthetic_data:
            print("注意：本次使用合成数据，指标仅供链路验证，不可写入报告。")
        return EXIT_OK

    if command == "evaluate":
        from handwash.pipelines.evaluate import evaluate

        rc = _load(args, overrides)
        bundle = evaluate(rc, checkpoint=args.checkpoint, splits=args.splits)
        print(bundle.summary_table())
        return EXIT_OK

    if command == "infer":
        from handwash.pipelines.infer import predict_video, save_overlay_video, save_predictions

        rc = _load(args, overrides)
        model = _load_model(rc, checkpoint=args.checkpoint)
        output = predict_video(rc, model, args.video)
        target = rc.resolve_out_dir() / "infer"
        print(f"推理完成：{output.prediction.num_frames} 帧")
        if rc.infer.save_frame_predictions:
            path = save_predictions(output, target / f"{Path(args.video).stem}_frames.jsonl")
            print(f"逐帧结果 -> {path}")
        if rc.infer.save_overlay_video:
            overlay = save_overlay_video(
                rc, args.video, output, target / f"{Path(args.video).stem}_overlay.gif"
            )
            print(f"叠加预览 -> {overlay}")
        return EXIT_OK

    if command == "assess":
        import json

        from handwash.pipelines.assess import assess_folder, assess_video

        rc = _load(args, overrides)
        model = _load_model(rc, checkpoint=args.checkpoint)
        if args.video:
            report = assess_video(rc, model, args.video)
            if args.json:
                print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
            else:
                from handwash.pipelines.assess import report_to_markdown

                print(report_to_markdown(report))
            # A detected protocol violation is the assessment result, not a
            # command failure. EXIT_PROTOCOL is reserved for ProtocolError.
            return EXIT_OK
        if args.folder:
            reports = assess_folder(rc, model, args.folder)
            if not reports:
                raise HandwashError(
                    "目录中没有成功评估的视频",
                    hint="检查目录里是否有受支持的视频文件，以及视频解码和 checkpoint 是否正常。",
                )
            for report in reports:
                verdict = "完整" if report.is_complete else "不完整"
                print(f"{report.clip_id:<24} {verdict:<6} 得分 {report.overall_score:.2f}")
            return EXIT_OK
        raise HandwashError("assess 需要 --video 或 --folder", hint="见 handwash assess -h")

    if command == "camera":
        from handwash.camera_app import serve_camera

        if args.demo_exp:
            runtime = {"run_name": "exp_demo", **overrides.get("runtime", {})}
            overrides = {**overrides, "runtime": runtime}
        rc = _load(args, overrides)
        if args.no_temporal_head and args.temporal_head:
            raise HandwashError("--temporal-head 与 --no-temporal-head 不能同时使用")
        temporal_head = False if args.no_temporal_head else (args.temporal_head or True)
        serve_camera(
            rc,
            checkpoint=args.checkpoint,
            port=args.port,
            demo_exp=args.demo_exp,
            temporal_head=temporal_head,
        )
        return EXIT_OK

    raise HandwashError(f"未实现的子命令：{command}")


def _load_model(rc, *, checkpoint: str | None):
    """按配置加载模型（评估/推理共用）。"""
    from handwash.pipelines.common import resolve_device
    from handwash.pipelines.evaluate import _load_model_from_checkpoint

    device = resolve_device(rc.runtime.device)
    ckpt = Path(checkpoint) if checkpoint else checkpoint_path(rc.resolve_out_dir())
    if not ckpt.is_file():
        raise HandwashError(
            f"找不到已训练的 checkpoint：{ckpt}",
            hint="推理与完整性评估必须指定项目格式的训练权重；先训练或传入 --checkpoint。",
        )
    model, _ = _load_model_from_checkpoint(rc, ckpt, device=device)
    return model


def _entry() -> None:  # pragma: no cover - 供 pyproject 的 console_scripts 使用
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    _entry()
