#!/usr/bin/env python3
"""Compare YOLO, YOLO + GRU, and YOLO + Adaptive Boundary-Aware GRU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from handwash.core.action_chain import adaptive_boundary_gru_chain
from handwash.core.boundary_metrics import (
    BoundaryScore,
    predicted_boundaries,
    reference_boundaries,
    score_boundaries,
)

MODELS = ("yolo", "yolo_gru", "boundary_gru")
MODEL_NAMES = {
    "yolo": "YOLO",
    "yolo_gru": "YOLO+GRU",
    "boundary_gru": "YOLO+自适应边界感知 GRU",
}


def collapse(values):
    result = []
    for value in values:
        value = int(value)
        if value > 0 and (not result or result[-1] != value):
            result.append(value)
    return result


def frame_metrics(y, prediction):
    y = np.asarray(y)
    prediction = np.asarray(prediction)
    accuracy = float(np.mean(y == prediction))
    class_f1 = []
    for class_id in range(7):
        tp = int(np.sum((y == class_id) & (prediction == class_id)))
        fp = int(np.sum((y != class_id) & (prediction == class_id)))
        fn = int(np.sum((y == class_id) & (prediction != class_id)))
        class_f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
    return accuracy, float(np.mean(class_f1))


def text_order(values):
    return "→".join(map(str, values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_hash = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    videos = defaultdict(list)
    with args.predictions.open(newline="") as handle:
        for row in csv.DictReader(handle):
            videos[row["video_id"]].append(row)

    result = {
        "method": "adaptive_boundary_gru/1",
        "applied_to": "yolo_adaptive_boundary_gru_only",
        "analysis_type": "retrospective reanalysis of previously inspected videos",
        "frame_predictions_changed": False,
        "input_sha256": source_hash,
        "models": {},
        "videos": {},
    }
    aggregate_y = {model: [] for model in MODELS}
    aggregate_pred = {model: [] for model in MODELS}
    aggregate_boundaries = {model: BoundaryScore() for model in MODELS}
    markdown_rows = []

    for video_id, rows in videos.items():
        times = np.asarray([float(row["actual_time_s"]) for row in rows])
        y_all = np.asarray([
            int(row["ground_truth"]) if row["valid_for_evaluation"] == "True" else -100
            for row in rows
        ])
        valid = y_all >= 0
        gt_order = collapse(y_all[valid])
        ref = reference_boundaries(y_all, times)
        video_result = {"truth_order": gt_order, "models": {}}
        for model in MODELS:
            prediction = np.asarray([int(row[model]) for row in rows])
            confidence = np.asarray([float(row[model+"_confidence"]) for row in rows])
            raw_order = collapse(prediction)
            aggregate_y[model].extend(y_all[valid].tolist())
            aggregate_pred[model].extend(prediction[valid].tolist())
            boundary_score, _ = score_boundaries(
                ref, predicted_boundaries(prediction, times), tolerance_s=0.4
            )
            aggregate_boundaries[model].add(boundary_score)
            item = {"raw_order": raw_order, "raw_order_match": raw_order == gt_order}
            if model == "boundary_gru":
                chain = adaptive_boundary_gru_chain(
                    prediction, fps=5.0, smoothing_window_seconds=1.0,
                    background_class=0, winner_confidences=confidence,
                    outlier_z=-2.0, minimum_comparable_segments=4,
                )
                item.update(
                    confirmed_order=chain["order"],
                    confirmed_order_match=chain["order"] == gt_order,
                    duration_comparison_available=chain["duration_comparison_available"],
                    unconfirmed_segments=[{
                        "class": segment["class"],
                        "start_s": segment["start_index"]/5.0,
                        "end_s": segment["end_index_exclusive"]/5.0,
                        "duration_s": segment["duration_s"],
                        "mean_class_probability": segment.get("mean_class_probability"),
                        "duration_outlier_z": segment["duration_outlier_z"],
                    } for segment in chain["segments"] if segment["status"] == "unconfirmed_outlier"],
                )
            video_result["models"][model] = item

        boundary = video_result["models"]["boundary_gru"]
        rejected = "; ".join(
            f"{seg['class']} {seg['start_s']:.1f}–{seg['end_s']:.1f}s "
            f"(均值置信度 {seg['mean_class_probability']:.2f})"
            for seg in boundary["unconfirmed_segments"]
        ) or "无"
        markdown_rows.append(
            f"| {video_id} | {text_order(gt_order)} | {text_order(boundary['raw_order'])} | "
            f"{text_order(boundary['confirmed_order'])} | "
            f"{'✓' if boundary['confirmed_order_match'] else '✗'} | {rejected} |"
        )
        result["videos"][video_id] = video_result

    for model in MODELS:
        accuracy, macro_f1 = frame_metrics(aggregate_y[model], aggregate_pred[model])
        result["models"][model] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "boundary_f1": aggregate_boundaries[model].f1,
            "raw_order_matches": sum(
                result["videos"][video]["models"][model]["raw_order_match"]
                for video in videos
            ),
        }
    result["models"]["boundary_gru"]["adaptive_chain_order_matches"] = sum(
        result["videos"][video]["models"]["boundary_gru"]["confirmed_order_match"]
        for video in videos
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    lines = [
        "# YOLO＋自适应边界感知 GRU：十段测试集结果",
        "",
        "YOLO＋自适应边界感知 GRU 把 YOLO 特征、边界感知 GRU 分类与边界预测、"
        "以及自适应动作片段确认作为一个完整方案。确认步骤将 1 秒多数平滑与视频内时长、"
        "置信度判断结合；YOLO 与 YOLO+GRU 不使用这一步。模型输出和逐帧指标未改变。",
        "",
        "这十段及错误区间此前已经检查，当前结果属于回顾性复算，不是新的独立测试。",
        "",
        "| 模型 | 帧准确率 | 宏 F1 | 边界 F1 | 原始顺序正确 | 最终动作链正确 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        item = result["models"][model]
        confirmed = f"{item['adaptive_chain_order_matches']}/{len(videos)}" if model == "boundary_gru" else "—"
        lines.append(
            f"| {MODEL_NAMES[model]} | {item['accuracy']:.2%} | {item['macro_f1']:.2%} | "
            f"{item['boundary_f1']:.2%} | {item['raw_order_matches']}/{len(videos)} | {confirmed} |"
        )
    lines += [
        "",
        f"YOLO＋自适应边界感知 GRU 的最终动作链为 {result['models']['boundary_gru']['adaptive_chain_order_matches']}/{len(videos)}；"
        f"原始顺序为 {result['models']['boundary_gru']['raw_order_matches']}/{len(videos)}。",
        "自适应片段确认不改写逐帧分类，因此不会改变帧准确率、宏 F1 或边界 F1。",
        "",
        "| 视频 | 标注 | 模型原始顺序 | 最终动作链 | 匹配 | 未确认片段 |",
        "|---|---|---|---|:---:|---|",
        *markdown_rows,
        "",
        "短片段异常依据该视频动作时长的中位数和 MAD；只有时长明显偏短且置信度低于视频中位数时才标为未确认。"
        "不足四个动作段时跳过时长异常判断。阈值仍需在未查看的新视频上验证。",
    ]
    (args.output/"REPORT.zh-CN.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result["models"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
