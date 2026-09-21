#!/usr/bin/env python
"""完整性评估脚本：一段/一批视频 -> WHO 六步报告（JSON + Markdown + 汇总 CSV）。

这是最终交付物生成器：把组员自采视频跑一遍，直接得到可以贴进报告的表格。

用法::

    # 单段
    python scripts/run_assess.py --video data/external/self_recorded/full/who_correct.mp4

    # 整个目录（自采视频全部）
    python scripts/run_assess.py --folder data/external/self_recorded/full

    # 与人工答案对比（人工答案格式见 docs/SELF_RECORDING.md）
    python scripts/run_assess.py --folder <目录> --ground-truth <人工答案.csv>
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT  # noqa: E402,F401
from _common import build_config_parser  # noqa: E402

from handwash.cli import split_argv  # noqa: E402
from handwash.core.config import load_config, parse_overrides  # noqa: E402
from handwash.io.utils import read_csv, write_csv  # noqa: E402
from handwash.logging import get_logger, setup_logging  # noqa: E402

log = get_logger(__name__)


def main() -> int:
    parser = build_config_parser(__doc__ or "")
    parser.add_argument("--video", default=None, help="单个视频文件")
    parser.add_argument("--folder", default=None, help="包含视频的目录")
    parser.add_argument("--checkpoint", default=None, help="checkpoint 路径（默认取 best.pt）")
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="人工答案 CSV（列：clip_id, expected_steps 例如 step_1_palm_to_palm;step_2_palm_over_dorsum）",
    )

    plain, override_items = split_argv(sys.argv[1:])
    args, unknown = parser.parse_known_args(plain)
    if unknown:
        parser.error(f"无法识别的参数：{unknown}")
    if not args.video and not args.folder:
        parser.error("必须提供 --video 或 --folder")

    setup_logging(args.log_level, force=True)
    rc = load_config(
        list(args.config or ["configs/config.yaml"]), overrides=parse_overrides(override_items)
    )

    from handwash.cli import _load_model
    from handwash.pipelines.assess import assess_folder, assess_video

    model = _load_model(rc, checkpoint=args.checkpoint)
    reports = (
        [assess_video(rc, model, args.video)]
        if args.video
        else assess_folder(rc, model, args.folder)
    )
    if not reports:
        print("没有可评估的视频。")
        return 1

    out_dir = rc.resolve_out_dir() / "assess"
    rows = [_summary_row(report) for report in reports]
    write_csv(
        out_dir / "assess_summary.csv",
        rows,
        fieldnames=[
            "clip_id", "is_complete", "is_in_order", "overall_score",
            "total_wash_duration_s", "num_missing", "num_violations", "missing_steps",
        ],
    )

    print(f"\n共评估 {len(reports)} 段视频，逐段报告在：{out_dir}")
    print(_table(rows))

    if args.ground_truth:
        _compare_with_truth(rows, Path(args.ground_truth), out_dir)
    else:
        print(
            "\n提示：自采视频请配合 docs/SELF_RECORDING.md 的人工答案 CSV，"
            "用 --ground-truth 生成对比表（报告需要「模型结果 vs 人工答案」）"
        )
    return 0


def _summary_row(report) -> dict[str, object]:
    missing = [v for v in report.violations if v.kind == "missing"]
    return {
        "clip_id": report.clip_id,
        "is_complete": report.is_complete,
        "is_in_order": report.is_in_order,
        "overall_score": round(report.overall_score, 4),
        "total_wash_duration_s": round(report.total_wash_duration_s, 2),
        "num_missing": len(missing),
        "num_violations": len(report.violations),
        "missing_steps": ";".join(
            str(v.step.order_index) for v in missing if v.step is not None
        ),
    }


def _table(rows: list[dict[str, object]]) -> str:
    header = f"{'clip_id':<26}{'完整':<6}{'顺序':<6}{'得分':>7}{'搓洗时长':>10}{'漏步数':>8}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{str(row['clip_id']):<26}"
            f"{'是' if row['is_complete'] else '否':<6}"
            f"{'是' if row['is_in_order'] else '否':<6}"
            f"{float(row['overall_score']):>7.2f}"
            f"{float(row['total_wash_duration_s']):>10.1f}"
            f"{int(row['num_missing']):>8}"
        )
    return "\n".join(lines)


def _compare_with_truth(rows: list[dict[str, object]], truth_path: Path, out_dir: Path) -> None:
    """与人工答案对比：只对比"是否完整 / 是否乱序"，因为这是可靠的人工标注维度。"""
    if not truth_path.exists():
        log.error("人工答案文件不存在：%s", truth_path)
        return
    truth = {row["clip_id"]: row for row in read_csv(truth_path)}

    compared: list[dict[str, object]] = []
    correct_complete = 0
    correct_order = 0
    for row in rows:
        clip_id = str(row["clip_id"])
        expected = truth.get(clip_id)
        if expected is None:
            continue
        exp_complete = str(expected.get("expected_complete", "")).strip().lower() in ("1", "true", "是", "yes")
        exp_order = str(expected.get("expected_in_order", "")).strip().lower() in ("1", "true", "是", "yes")
        match_complete = bool(row["is_complete"]) == exp_complete
        match_order = bool(row["is_in_order"]) == exp_order
        correct_complete += int(match_complete)
        correct_order += int(match_order)
        compared.append(
            {
                "clip_id": clip_id,
                "expected_complete": exp_complete,
                "predicted_complete": row["is_complete"],
                "match_complete": match_complete,
                "expected_in_order": exp_order,
                "predicted_in_order": row["is_in_order"],
                "match_order": match_order,
            }
        )

    if not compared:
        log.warning("人工答案里没有任何 clip_id 与本次评估结果对应，请检查列名与命名")
        return

    write_csv(out_dir / "assess_vs_ground_truth.csv", compared, fieldnames=list(compared[0].keys()))
    total = len(compared)
    print("\n与人工答案对比（自采视频）：")
    print(f"  可比对视频数：{total}")
    print(f"  「是否完整」判断准确率：{correct_complete / total:.1%}")
    print(f"  「是否乱序」判断准确率：{correct_order / total:.1%}")
    print(f"  对比明细：{out_dir / 'assess_vs_ground_truth.csv'}")


if __name__ == "__main__":
    raise SystemExit(main())
