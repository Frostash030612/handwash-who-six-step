"""Time accounting for the selected model. No per-step duration requirement."""

from math import isfinite

MINIMUM_WASH_SECONDS = 40.0


def make_segments(labels, starts, end_s, *, confidence=None, statuses=None):
    """Turn a timeline into disjoint intervals, clipped to the recording end.

    -1 means no prediction, 0 means others. All six actions contribute time,
    including short segments that the boundary model did not admit to its chain.
    """
    if len(labels) != len(starts):
        raise ValueError("The timestamp and label counts do not match")
    if not isfinite(end_s) or end_s < 0:
        raise ValueError("Invalid recording duration")
    result = []
    for i, value in enumerate(labels):
        left = max(0.0, float(starts[i]))
        right = min(float(end_s), float(starts[i + 1]) if i + 1 < len(starts) else float(end_s))
        if right <= left:
            continue
        label = int(value)
        if label not in range(-1, 7):
            raise ValueError("Unknown action class")
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
        raise ValueError("Invalid minimum washing duration")
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
        reasons.append(
            f"Insufficient washing time: {washing:.1f} seconds; required minimum: {minimum_seconds:g} s"
        )
    elif duration_status == "insufficient_evidence":
        reasons.append(
            f"Recognized washing: {washing:.1f} seconds; {unknown:.1f} seconds have no prediction, so the duration requirement cannot yet be confirmed"
        )
    if missing:
        reasons.append("Missing steps: " + ", ".join(map(str, missing)))
    elif not in_order:
        reasons.append("The detected steps are out of order; expected 1 -> 2 -> 3 -> 4 -> 5 -> 6")
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
        "verdict": "Complete"
        if passed
        else ("Needs review" if duration_status == "insufficient_evidence" and in_order else "Incomplete"),
        "reasons": reasons,
        "per_step_duration_required": False,
    }
