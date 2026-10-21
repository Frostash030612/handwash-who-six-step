"""Time accounting for the selected model. No per-step duration requirement."""

from math import isfinite

MINIMUM_WASH_SECONDS = 40.0


def make_segments(labels, starts, end_s, *, confidence=None, statuses=None):
    """Turn a timeline into disjoint intervals, clipped to the recording end.

    -1 means no prediction, 0 means others. All six actions contribute time,
    including short segments that the boundary model did not admit to its chain.
    """
    if len(labels) != len(starts):
        raise ValueError("时间和标签数量不一致")
    if not isfinite(end_s) or end_s < 0:
        raise ValueError("无效录像时长")
    result = []
    for i, value in enumerate(labels):
        left = max(0.0, float(starts[i]))
        right = min(float(end_s), float(starts[i + 1]) if i + 1 < len(starts) else float(end_s))
        if right <= left:
            continue
        label = int(value)
        if label not in range(-1, 7):
            raise ValueError("未知动作类别")
        status = (
            statuses[i]
            if statuses is not None
            else ("unknown" if label == -1 else "background" if label == 0 else "predicted")
        )
        p = float(confidence[i]) if confidence is not None and label >= 0 else None
        duration = right - left
        if (
            result
            and result[-1]["class"] == label
            and result[-1]["status"] == status
            and abs(result[-1]["end_s"] - left) < 1e-6
        ):
            item = result[-1]
            if p is not None:
                item["mean_class_probability"] = (
                    item["mean_class_probability"] * item["duration_s"] + p * duration
                ) / (item["duration_s"] + duration)
            item["end_s"] = right
            item["duration_s"] += duration
        else:
            result.append(
                {
                    "class": label,
                    "start_s": left,
                    "end_s": right,
                    "duration_s": duration,
                    "status": status,
                    "mean_class_probability": p,
                }
            )
    return result


def collapse_order(labels):
    order = []
    for label in labels:
        label = int(label)
        if 1 <= label <= 6 and (not order or order[-1] != label):
            order.append(label)
    return order


def assess(segments, order, recorded_seconds, minimum_seconds=MINIMUM_WASH_SECONDS):
    if not isfinite(minimum_seconds) or minimum_seconds <= 0:
        raise ValueError("无效最低洗手时长")
    washing = sum(s["duration_s"] for s in segments if 1 <= s["class"] <= 6)
    background = sum(s["duration_s"] for s in segments if s["class"] == 0)
    unknown = max(0.0, recorded_seconds - washing - background)
    enough = washing + 1e-9 >= minimum_seconds
    duration_status = (
        "met"
        if enough
        else ("insufficient_evidence" if washing + unknown + 1e-9 >= minimum_seconds else "not_met")
    )
    missing = sorted(set(range(1, 7)) - set(order))
    in_order = list(order) == [1, 2, 3, 4, 5, 6]
    passed = enough and not missing and in_order
    reasons = []
    if duration_status == "not_met":
        reasons.append(f"洗手总时长不足：{washing:.1f} 秒，要求至少 {minimum_seconds:g} 秒")
    elif duration_status == "insufficient_evidence":
        reasons.append(f"已识别洗手 {washing:.1f} 秒，另有 {unknown:.1f} 秒未得到预测，暂不能确认时长达标")
    if missing:
        reasons.append("未识别到动作 " + "、".join(map(str, missing)))
    elif not in_order:
        reasons.append("动作顺序不符合 1 → 2 → 3 → 4 → 5 → 6")
    return {
        "wash_seconds": washing,
        "background_seconds": background,
        "unknown_seconds": unknown,
        "minimum_wash_seconds": minimum_seconds,
        "remaining_wash_seconds": max(0.0, minimum_seconds - washing),
        "duration_met": enough,
        "duration_status": duration_status,
        "steps_complete": not missing,
        "missing_steps": missing,
        "order_correct": in_order,
        "passed": passed,
        "verdict": "达标"
        if passed
        else ("待确认" if duration_status == "insufficient_evidence" and in_order else "未达标"),
        "reasons": reasons,
        "per_step_duration_required": False,
    }
