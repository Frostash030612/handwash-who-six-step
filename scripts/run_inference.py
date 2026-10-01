#!/usr/bin/env python
"""推理脚本：视频 -> 逐帧动作预测（JSONL + 可选 GIF 叠加预览）。

用法::

    python scripts/run_inference.py --video data/external/self_recorded/full/demo.mp4
    python scripts/run_inference.py --video demo.mp4 --checkpoint outputs/e2/models/best.pt
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT  # noqa: E402,F401
from _common import build_config_parser, load_config_from_args  # noqa: E402

from handwash.cli import split_argv  # noqa: E402
from handwash.core.config import load_config, parse_overrides  # noqa: E402
from handwash.logging import get_logger, setup_logging  # noqa: E402

log = get_logger(__name__)


def main() -> int:
    parser = build_config_parser(__doc__ or "")
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--checkpoint", default=None, help="checkpoint 路径（默认取 best.pt）")
    parser.add_argument("--overlay", action="store_true", help="同时导出一段带预测文字的预览 GIF")

    plain, override_items = split_argv(sys.argv[1:])
    args, unknown = parser.parse_known_args(plain)
    if unknown:
        parser.error(f"无法识别的参数：{unknown}")
    setup_logging(args.log_level, force=True)

    rc = load_config_from_args(args, parse_overrides(override_items))

    from handwash.cli import _load_model
    from handwash.pipelines.infer import (
        predict_video,
        save_overlay_video,
        save_predictions,
    )

    model = _load_model(rc, checkpoint=args.checkpoint)
    output = predict_video(rc, model, args.video)

    out_dir = rc.resolve_out_dir() / "infer"
    print(f"推理完成：{output.prediction.num_frames} 帧，fps={output.fps:.1f}")
    if rc.infer.save_frame_predictions:
        path = save_predictions(output, out_dir / f"{Path(args.video).stem}_frames.jsonl")
        print(f"逐帧结果：{path}")

    counts: dict[str, int] = {}
    for label in output.labels():
        counts[label.value] = counts.get(label.value, 0) + 1
    print("标签分布（帧数）：")
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<32} {count:>6}")

    if args.overlay or rc.infer.save_overlay_video:
        path = save_overlay_video(
            rc, args.video, output, out_dir / f"{Path(args.video).stem}_overlay.gif"
        )
        print(f"叠加预览：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
