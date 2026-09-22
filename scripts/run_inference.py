#!/usr/bin/env python
"""推理脚本：视频 -> 逐帧动作预测（JSONL + 可选 GIF 叠加预览）。

用法::

    python scripts/run_inference.py --video data/external/self_recorded/full/demo.mp4
    python scripts/run_inference.py --video demo.mp4 --checkpoint outputs/e2/models/best.pt
"""

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
    from handwash.pipelines.infer import predict_video, save_predictions

    model = _load_model(rc, checkpoint=args.checkpoint)
    output = predict_video(rc, model, args.video)

    out_dir = rc.resolve_out_dir() / "infer"
    path = save_predictions(output, out_dir / f"{Path(args.video).stem}_frames.jsonl")
    print(f"推理完成：{output.prediction.num_frames} 帧，fps={output.fps:.1f}")
    print(f"逐帧结果：{path}")

    counts: dict[str, int] = {}
    for label in output.labels():
        counts[label.value] = counts.get(label.value, 0) + 1
    print("标签分布（帧数）：")
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<32} {count:>6}")

    if args.overlay:
        _write_overlay(rc, args.video, output, out_dir)
    return 0


def _write_overlay(rc, video_path: str, output, out_dir: Path) -> None:
    """把预测标签画在帧上，导出一段 GIF，便于答辩时"看得见"。

    GIF 是刻意选择：不需要额外编码器即可生成，且任何电脑都能播放。
    """
    import numpy as np

    from handwash.core.labels import STEP_ZH
    from handwash.io.video import extract_frames, write_video

    out_dir.mkdir(parents=True, exist_ok=True)
    _, _, frames = extract_frames(video_path, sample_fps=rc.dataset.prep.fps)
    labels = output.labels()
    confidences = output.confidences()

    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover
        log.warning("未安装 Pillow，跳过叠加预览")
        return

    drawn = []
    for index, frame in enumerate(frames[: len(labels)]):
        image = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        step = labels[index]
        text = f"{step.order_index or '-'} {STEP_ZH.get(step, step.value)}  {confidences[index]:.2f}"
        draw.rectangle([0, 0, image.width, 26], fill=(0, 0, 0, 160))
        draw.text((6, 6), text, fill=(255, 255, 255, 255))
        drawn.append(np.asarray(image))

    target = write_video(drawn, out_dir / f"{Path(video_path).stem}_overlay.gif", fps=output.fps)
    print(f"叠加预览：{target}")


if __name__ == "__main__":
    raise SystemExit(main())
