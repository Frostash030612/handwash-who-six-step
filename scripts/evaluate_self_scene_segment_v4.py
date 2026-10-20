#!/usr/bin/env python
"""Select a whole-segment decoder on validation, then run frozen regression.

Reads archived v3 probabilities. No network training, reference-label input to
decoding, class-order constraint, or confidence fabricated from one-hot labels.
The positive-boundary and boundary-free families are selected independently
with one shared setting across seeds. The same-setting edge-off run isolates
the effect of boundary evidence at inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np
import yaml

try:
    from _bootstrap import PROJECT_ROOT
except ImportError:
    from ._bootstrap import PROJECT_ROOT
from handwash.core.boundary_metrics import (
    BoundaryScore,
    predicted_boundaries,
    reference_boundaries,
    score_boundaries,
)
from handwash.core.segment_decoding import SegmentDecoderConfig, decode_action_segments
from handwash.core.segment_metrics import action_segments, segment_counts
from handwash.core.temporal_selection import replacement_decision
from handwash.pipelines.action_chain import sustained_action_chain


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_items(source, ids, cfg):
    items, hashes = {}, {}
    for seed in cfg["seeds"]:
        for ident in ids:
            path = source / "report/predictions" / f"boundary_multiscale_seed{seed}" / f"video{ident:02d}.npz"
            with np.load(path, allow_pickle=False) as data:
                item = {name: data[name].copy() for name in ("labels", "times", "probs", "edges", "decoded")}
                if not np.allclose(data["backbone_probs"], data["probs"], atol=1e-7):
                    raise ValueError("This experiment requires unchanged frozen GRU probabilities")
            item.update(seed=seed, clip=ident, continuous=ident in cfg["continuous"])
            if len(item["labels"]) != len(item["times"]) or len(item["probs"]) != len(item["labels"]):
                raise ValueError(f"Unaligned data: {path}")
            if seed != cfg["seeds"][0]:
                first = items[cfg["seeds"][0], ident]
                if not np.array_equal(first["labels"], item["labels"]) or not np.array_equal(
                    first["times"], item["times"]
                ):
                    raise ValueError("Seeds must share identical labels and timestamps")
            items[seed, ident] = item
            hashes[str(path)] = sha(path)
    return items, hashes


def chain(labels, cfg):
    settings = cfg["evaluation"]
    return sustained_action_chain(
        labels.tolist(),
        fps=cfg["decoder"]["sample_fps"],
        min_action_seconds=settings["action_chain_min_seconds"],
        smoothing_window_seconds=settings["action_chain_smoothing_seconds"],
        background_class=cfg["decoder"]["background_class"],
    )


def truth_order(labels):
    result = []
    for value in labels:
        if value > 0 and (not result or result[-1] != int(value)):
            result.append(int(value))
    return result


def errors(item, prediction, cfg):
    labels, times, probs = item["labels"], item["times"], item["probs"]
    intervals = []
    for index in np.flatnonzero((labels >= 0) & (labels != prediction)):
        key = (int(labels[index]), int(prediction[index]))
        if (
            intervals
            and intervals[-1]["last_index"] + 1 == index
            and (intervals[-1]["truth"], intervals[-1]["prediction"]) == key
        ):
            intervals[-1]["last_index"] = int(index)
        else:
            intervals.append(
                {"truth": key[0], "prediction": key[1], "first_index": int(index), "last_index": int(index)}
            )
    for row in intervals:
        first, end = row["first_index"], row["last_index"] + 1
        row.update(
            start_s=float(times[first]),
            last_sample_s=float(times[end - 1]),
            duration_s=(end - first) / cfg["decoder"]["sample_fps"],
            mean_raw_predicted_class_probability=float(probs[first:end, row["prediction"]].mean()),
            mean_raw_true_class_probability=float(probs[first:end, row["truth"]].mean()),
        )
    return intervals


def evaluate(item, prediction, cfg, detail=False):
    labels, times = item["labels"], item["times"]
    valid = labels >= 0
    confirmed = chain(prediction, cfg)
    reference = reference_boundaries(labels, times)
    predicted = predicted_boundaries(prediction, times)
    boundary, matches = score_boundaries(
        reference, predicted, tolerance_s=cfg["evaluation"]["boundary_tolerance_s"]
    )
    smooth, _ = score_boundaries(
        reference,
        predicted_boundaries(confirmed["smoothed_labels"], times),
        tolerance_s=cfg["evaluation"]["boundary_tolerance_s"],
    )
    matrix = np.zeros((7, 7), dtype=np.int64)
    np.add.at(matrix, (labels[valid], prediction[valid]), 1)
    # Same original sampled index axis; ignored transition samples break runs.
    reference_segments = action_segments(labels, valid)
    counts = segment_counts(
        reference_segments, action_segments(prediction, valid), overlap=cfg["evaluation"]["segment_iou"]
    )
    result = {
        "clip": item["clip"],
        "seed": item["seed"],
        "continuous": item["continuous"],
        "confusion": matrix,
        "boundary": boundary,
        "smoothed_boundary": smooth,
        "segment_counts": counts,
        "chain_match": confirmed["order"] == truth_order(labels),
    }
    if detail:
        matched_ref = {m.reference for m in matches}
        matched_pred = {m.predicted for m in matches}
        steps = np.flatnonzero(labels > 0)
        washing_edges = {}
        if len(steps):
            for name, mask in (
                ("opening_others", (np.arange(len(labels)) < steps[0]) & (labels == 0)),
                ("closing_others", (np.arange(len(labels)) > steps[-1]) & (labels == 0)),
            ):
                washing_edges[name] = {
                    "frames": int(mask.sum()),
                    "correct": int(np.count_nonzero(prediction[mask] == 0)),
                }
        result["detail"] = {
            "clip": item["clip"],
            "seed": item["seed"],
            "truth_order": truth_order(labels),
            "confirmed_order": confirmed["order"],
            "chain_match": result["chain_match"],
            "accuracy": float(np.mean(prediction[valid] == labels[valid])),
            "raw_chain_order": chain(item["probs"].argmax(1), cfg)["order"],
            "boundary": boundary.as_dict(),
            "smoothed_boundary": smooth.as_dict(),
            "false_boundaries": [
                {"time_s": b.time_s, "before": b.before, "after": b.after}
                for i, b in enumerate(predicted)
                if i not in matched_pred
            ],
            "missed_boundaries": [asdict(b) for i, b in enumerate(reference) if i not in matched_ref],
            "error_intervals": errors(item, prediction, cfg),
            "washing_edges": washing_edges,
            "confirmation_segments": confirmed["segments"],
            "reference_short_action_segments": sum(
                (s.end - s.start) / cfg["decoder"]["sample_fps"] < 3 for s in reference_segments
            ),
        }
    return result


def aggregate(records, cfg):
    matrix = sum((r["confusion"] for r in records), np.zeros((7, 7), dtype=np.int64))
    denominator = matrix.sum(0) + matrix.sum(1)
    per_class = np.divide(
        2 * matrix.diagonal(), denominator, out=np.zeros(7, dtype=float), where=denominator > 0
    )
    boundary, smoothed = BoundaryScore(), BoundaryScore()
    for r in records:
        boundary.add(r["boundary"])
        smoothed.add(r["smoothed_boundary"])
    tp, fp, fn = np.sum([r["segment_counts"] for r in records], axis=0).tolist()
    continuous = [r["chain_match"] for r in records if r["continuous"]]
    isolated = [r["chain_match"] for r in records if not r["continuous"]]
    result = {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "macro_f1": float(per_class.mean()),
        "per_class_f1": per_class.tolist(),
        "boundary_f1": boundary.f1,
        "boundary_precision": boundary.precision,
        "boundary_recall": boundary.recall,
        "boundary_mean_abs_error_s": boundary.mean_abs_error_s,
        "boundary_counts": boundary.as_dict(),
        "boundary_f1_after_chain_smoothing": smoothed.f1,
        "segment_f1_50": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "segment_counts": {"true_positive": tp, "false_positive": fp, "false_negative": fn},
        "continuous_chain_match_rate": float(np.mean(continuous)),
        "continuous_chain_matches": f"{sum(continuous)}/{len(continuous)}",
        "isolated_chain_matches": f"{sum(isolated)}/{len(isolated)}",
    }
    result["composite"] = sum(result[k] * w for k, w in cfg["selection"]["score"].items())
    return result


MEAN_KEYS = (
    "accuracy",
    "macro_f1",
    "boundary_f1",
    "boundary_precision",
    "boundary_recall",
    "boundary_mean_abs_error_s",
    "boundary_f1_after_chain_smoothing",
    "segment_f1_50",
    "continuous_chain_match_rate",
    "composite",
)


def mean_seeds(rows):
    return {
        key: float(np.mean([r[key] for r in rows])) if all(r[key] is not None for r in rows) else None
        for key in MEAN_KEYS
    }


def run_setting(items, setting, cfg):
    by_seed = {}
    for seed in cfg["seeds"]:
        records = []
        for (s, _), item in items.items():
            if s == seed:
                decoded = decode_action_segments(item["probs"], item["edges"], setting)
                records.append(evaluate(item, decoded.labels, cfg))
        by_seed[str(seed)] = aggregate(records, cfg)
    return {"config": asdict(setting), "mean": mean_seeds(list(by_seed.values())), "per_seed": by_seed}


def selection_key(row):
    m, c = row["mean"], row["config"]
    return (
        round(m["composite"], 12),
        round(m["macro_f1"], 12),
        round(m["boundary_f1"], 12),
        -c["boundary_weight"],
        -c["short_action_penalty_per_second"],
        -c["change_penalty"],
    )


def render(summary, details, output):
    lines = [
        "# 整段动作解码 v4：固定模型复测",
        "",
        "仅修改解码，使用 v3 冻结 GRU 与多尺度边界头的原始概率。"
        "27 组预定设置仅用 8 段验证视频选择；每个候选族跨三个种子共享一组参数。"
        "选择结果写盘后才读取回归视频 4、8。现有视频已经反复检查，本轮仍是内部验证。",
        "",
        "切换代价由固定代价及低边界概率的额外代价组成；高边界概率不能奖励凭空切段。"
        "内部不足 3 秒的动作只受软惩罚，强证据仍可保留；others 和视频两端片段豁免。"
        "所有类别顺序均允许。短动作确认规则仍为 1 秒居中平滑、连续 3 秒确认。",
        "",
        "输出离散标签与原始模型证据；未把解码标签的 one-hot 值当置信度。"
        "WHO 片段 F1@50 在 5 fps 原采样索引轴上计，未知标签打断片段。",
        "",
        "## 三个种子均值",
        "",
        "| 方案 | 数据 | 准确率 | 宏 F1 | 边界 F1 | 边界精确率 | 边界召回率 | 时间误差/秒 | 片段 F1@50 | 动作链 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, splits in summary["mean"].items():
        for split, m in splits.items():
            timing = (
                "—" if m["boundary_mean_abs_error_s"] is None else f"{m['boundary_mean_abs_error_s']:.4f}"
            )
            lines.append(
                f"| {name} | {split} | {m['accuracy']:.2%} | {m['macro_f1']:.2%} | "
                f"{m['boundary_f1']:.2%} | {m['boundary_precision']:.2%} | "
                f"{m['boundary_recall']:.2%} | {timing} | {m['segment_f1_50']:.2%} | "
                f"{m['continuous_chain_match_rate']:.2%} |"
            )
    lines += [
        "",
        "## 验证选择的参数",
        "",
        "```json",
        json.dumps(summary["selection"]["selected"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 三种子明细",
        "",
        "| 方案 | 数据 | 种子 | 准确率 | 宏 F1 | 边界 TP/FP/FN | 边界 F1 | 动作链 |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ]
    for r in summary["per_run"]:
        c = r["boundary_counts"]
        lines.append(
            f"| {r['mode']} | {r['set']} | {r['seed']} | {r['accuracy']:.2%} | "
            f"{r['macro_f1']:.2%} | {c['true_positive']}/{c['false_positive']}/{c['false_negative']} | "
            f"{r['boundary_f1']:.2%} | {r['continuous_chain_matches']} |"
        )
    lines += [
        "",
        "## 逐视频动作链",
        "",
        "| 方案 | 数据 | 种子 | 视频 | 真实 | 确认预测 | 匹配 |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for r in details:

        def fmt(values):
            return "→".join(map(str, values)) or "空"

        lines.append(
            f"| {r['mode']} | {r['set']} | {r['seed']} | {r['clip']} | "
            f"{fmt(r['truth_order'])} | {fmt(r['confirmed_order'])} | {r['chain_match']} |"
        )
    lines += [
        "",
        "替换门槛见 summary.json 的 replacement_decisions；本脚本不更换部署模型。",
        "原始概率、逐帧标签和解码片段证据在 predictions/；错误时间段和原始置信度在 per_video.json。",
        "selection.json 与 validation_grid.json 保存选择过程；新采集的独立测试视频未使用。",
        "",
    ]
    (output / "REPORT.zh-CN.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/research/self_scene_segment_v4.yaml"
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if set(cfg["validation"]) & set(cfg["regression"]):
        raise ValueError("Validation and regression must be disjoint")
    args.output.mkdir(parents=True, exist_ok=True)
    config_hash = sha(args.config)
    started = time.time()
    items, hashes = load_items(args.source, cfg["validation"], cfg)
    selected_path = args.output / "selection.json"
    if selected_path.exists():
        selection = json.loads(selected_path.read_text())
        if selection["config_sha256"] != config_hash or selection["validation_inputs"] != hashes:
            raise ValueError("Existing selection inputs differ; use a new output directory")
    else:
        grid = []
        for penalty, short, edge in product(
            cfg["grid"]["change_penalty"],
            cfg["grid"]["short_action_penalty_per_second"],
            cfg["grid"]["boundary_weight"],
        ):
            setting = SegmentDecoderConfig(
                **cfg["decoder"],
                change_penalty=penalty,
                short_action_penalty_per_second=short,
                boundary_weight=edge,
            )
            row = run_setting(items, setting, cfg)
            grid.append(row)
            print(
                f"grid {len(grid)}/27 {penalty=} {short=} {edge=} score={row['mean']['composite']:.6f}",
                flush=True,
            )
        selected = {}
        for family, use_edges in (("without_boundary", False), ("with_boundary", True)):
            rows = [r for r in grid if (r["config"]["boundary_weight"] > 0) == use_edges]
            selected[family] = max(rows, key=selection_key)["config"]
        selection = {
            "config_sha256": config_hash,
            "validation_inputs": hashes,
            "selected": selected,
            "selected_at_unix": time.time(),
            "regression_opened": False,
            "selection_rule": cfg["selection"],
        }
        write_json(args.output / "validation_grid.json", grid)
        write_json(selected_path, selection)
    # No regression arrays have been opened above this line.
    regression, regression_hashes = load_items(args.source, cfg["regression"], cfg)
    all_hashes = {**hashes, **regression_hashes}
    datasets = {"validation": items, "regression": regression}
    with_edge = SegmentDecoderConfig(**selection["selected"]["with_boundary"])
    without_edge = SegmentDecoderConfig(**selection["selected"]["without_boundary"])
    methods = {
        "raw_gru": None,
        "v3_saved": None,
        "duration_control": without_edge,
        "boundary_duration": with_edge,
        "same_decoder_edge_off": replace(with_edge, boundary_weight=0),
    }
    per_run, details, means = [], [], {}
    for mode, setting in methods.items():
        means[mode] = {}
        for split, data in datasets.items():
            rows = []
            for seed in cfg["seeds"]:
                records = []
                for (s, ident), item in data.items():
                    if s != seed:
                        continue
                    decoded = (
                        None
                        if setting is None
                        else decode_action_segments(item["probs"], item["edges"], setting)
                    )
                    prediction = (
                        (item["probs"] if mode == "raw_gru" else item["decoded"]).argmax(1)
                        if decoded is None
                        else decoded.labels
                    )
                    record = evaluate(item, prediction, cfg, detail=True)
                    records.append(record)
                    detail = dict(mode=mode, set=split, **record["detail"])
                    if decoded:
                        detail["decoder_objective"] = decoded.objective
                        detail["decoded_segments"] = decoded.segments
                    details.append(detail)
                    destination = args.output / "predictions" / f"{mode}_seed{seed}" / f"video{ident:02d}.npz"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        destination,
                        labels=item["labels"],
                        times=item["times"],
                        raw_probs=item["probs"],
                        boundary_probs=item["edges"],
                        decoded_labels=prediction,
                    )
                row = dict(mode=mode, set=split, seed=seed, **aggregate(records, cfg))
                per_run.append(row)
                rows.append(row)
            means[mode][split] = mean_seeds(rows)
    decisions = {
        name: {split: replacement_decision(means[name][split], means["raw_gru"][split]) for split in datasets}
        for name in ("duration_control", "boundary_duration")
    }
    checkpoints = {str(p): sha(p) for p in (args.source / "runs").glob("boundary_multiscale_seed*/head.pt")}
    summary = {
        "format": cfg["format"],
        "selection": selection,
        "mean": means,
        "per_run": per_run,
        "replacement_decisions": decisions,
        "inputs": all_hashes,
        "checkpoints": checkpoints,
        "elapsed_seconds": time.time() - started,
        "new_independent_test_evaluated": False,
        "training_performed": False,
        "deployment_replaced": False,
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "per_video.json", details)
    (args.output / "config.yaml").write_text(args.config.read_text())
    render(summary, details, args.output)
    print(json.dumps(means, indent=2), flush=True)
    (args.output / "COMPLETE").write_text("Frozen-model evaluation complete\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
