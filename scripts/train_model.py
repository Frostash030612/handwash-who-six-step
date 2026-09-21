#!/usr/bin/env python
"""训练脚本：包一层 ``pipelines.train``，用于多步组合（准备数据 -> 训练 -> 评估）。

与 ``handwash train`` 的区别：本脚本可以在一次调用里加上自动评估，
适合"睡前挂机、早上看结果"的场景。

用法::

    python scripts/train_model.py --config configs/experiments/exp02_yolo26n_gru.yaml
    python scripts/train_model.py --config configs/experiments/smoke.yaml --evaluate
"""

#!/usr/bin/env python
"""训练脚本：包一层 ``pipelines.train``，用于多步组合（准备数据 -> 训练 -> 评估）。

与 ``handwash train`` 的区别：本脚本可以在一次调用里加上自动评估，
适合"睡前挂机、早上看结果"的场景。

用法::

    python scripts/train_model.py --config configs/experiments/exp02_yolo26n_gru.yaml
    python scripts/train_model.py --config configs/experiments/smoke.yaml --evaluate
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT  # noqa: E402,F401
from _common import build_config_parser  # noqa: E402

from handwash.cli import split_argv  # noqa: E402
from handwash.core.config import load_config, parse_overrides  # noqa: E402
from handwash.logging import get_logger, setup_logging  # noqa: E402

log = get_logger(__name__)


def main() -> int:
    parser = build_config_parser(__doc__ or "")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 train.epochs")
    parser.add_argument("--evaluate", action="store_true", help="训练结束后自动评估")
    parser.add_argument("--no-synthetic", action="store_true", help="禁止回退到合成数据")

    plain, override_items = split_argv(sys.argv[1:])
    args, unknown = parser.parse_known_args(plain)
    if unknown:
        parser.error(f"无法识别的参数：{unknown}")
    setup_logging(args.log_level, force=True)

    overrides = parse_overrides(override_items)
    if args.run_name:
        overrides["runtime"] = {**overrides.get("runtime", {}), "run_name": args.run_name}
    rc = load_config(list(args.config or ["configs/config.yaml"]), overrides=overrides)

    from handwash.pipelines.train import train

    result = train(rc, max_epochs=args.epochs)

    print("\n" + "=" * 66)
    print(f"训练完成：best val macro-F1 = {result.best_metric:.4f}")
    print(f"实验目录：{result.run_dir}")
    print(f"最佳权重：{result.best_checkpoint}")
    if result.used_synthetic_data:
        print("注意：本次使用合成数据，指标只能验证链路，不能写入报告。")
    print("=" * 66)

    if args.evaluate:
        from handwash.pipelines.evaluate import evaluate

        bundle = evaluate(rc, checkpoint=result.best_checkpoint)
        print("\n" + bundle.summary_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
