> **English** | [中文](/docs/zh/PROTOCOL.md)

# Completeness Judgement Criteria (PROTOCOL)

This document defines the **precise meaning** of "missed step / out-of-order /
insufficient duration".
Every number in the report must be traceable to a definition and a threshold origin found here.

Implementation: `../src/handwash/core/protocol.py`
Thresholds: the `assess` section of `configs/config.yaml` (**do not write thresholds in code**)

---

## 1. Input and output

```
Input:  per-frame label sequence labels[] (Step), per-frame confidence confidences[],
        effective frame rate fps
Output: ProtocolReport {
          is_complete, is_in_order,
          step_sequence, statistics[], violations[],
          total_wash_duration_s, total_duration_s, overall_score, notes[]
        }
```

Processing chain: **smoothing → segmentation → three classes of judgement → aggregate scoring**

---

## 2. Smoothing (eliminating single-frame jumps)

`assess.smooth_window` (default 9 frames, about 1.8 seconds at 5 fps)

- A **majority-vote** sliding window centred on the current frame; the window shrinks
  automatically at both ends of the sequence (no zero padding, otherwise the boundary
  would be "voted" into some class on no grounds at all).
- On a tie, keep the "previous frame's result" (temporal coherence takes precedence over
  instantaneous confidence).
- Frames whose confidence is below `assess.min_confidence` (default 0.4) **do not take
  part in the vote**; when the whole window is untrustworthy, the previous frame's label
  is carried over.

> Why not "plain majority + fixed window": in our measurements step 4 and step 2 (both
> involving the back of the hand) tend to jump back and forth into each other, and the
> temporal-coherence constraint markedly reduces that kind of jitter.

---

## 3. Segmentation (cutting the frame sequence into action segments)

`assess.min_segment_frames` (default 5 frames) + `assess.min_segment_s` (default 1.0 second)

- Consecutive frames with the same label form one segment.
- **Both** conditions must hold for a segment to count as valid.
- Segments that are too short are **not discarded**; they are merged into the nearest
  neighbouring segment in time (keeping the timeline free of holes).
  If every segment is too short, fall back to "one dominant label for the whole clip",
  so that later statistics never come out empty.

> Why not simply delete short segments: deleting them would shorten the total duration
> out of thin air, and the "insufficient duration" conclusion would be manufactured by
> that very deletion.

---

## 4. Three classes of judgement

### 4.1 Missed step (missing)

```
detected(step) := cumulative duration of that step >= assess.min_segment_s
is_complete    := |{detected among the six steps}| >= 6 - assess.missing_tolerance
```

- Default `missing_tolerance: 0` — missing one step already counts as incomplete.
- Every missing step produces one violation with `severity="error"`.

**Note**: `detected` uses a **duration threshold**, not "appeared in at least one frame".
Rationale: a momentary misclassification must not be taken as "this step was performed".

### 4.2 Out-of-order (out_of_order)

```
action sequence := collapse_repeats(segment label sequence)   # drop consecutive repeats
for the six-step subsequence, count all inversion pairs (earlier, later):
    if order(later) < order(earlier)  → one out_of_order
```

- Only the WHO six steps are considered; `faucet_on/off`, `other` and `unknown` take no
  part in the order judgement.
- **1 → 2 → 1** (going back to redo a step) is judged as out-of-order and explained
  separately rather than silently ignored — out-of-order behaviour is itself one of the
  targets this project is meant to detect.
- `assess.order_check: false` turns the order judgement off (for comparison experiments).

### 4.3 Insufficient duration (insufficient_duration)

`assess.duration_check` takes one of three values:

| Value | Judgement condition | When it applies |
| --- | --- | --- |
| `seconds` | Step duration ≥ `min_step_duration_s` (default 3.0 s) | Datasets whose action tempo is known |
| `ratio` (default) | Step duration ≥ `step_duration_ratio` × (total wash time ÷ number of steps required) | **Recommended**: WHO does not require the six steps to share time equally |
| `none` | No judgement | When only missed steps / order matter |

Plus one **whole-procedure** judgement:

```
if total_wash_duration_s < assess.min_total_duration_s  → one error
(default 40.0 s, based on the WHO recommendation of 40—60 seconds for a complete wash)
```

> **Why `ratio` is the default**: the WHO guideline only requires the whole procedure to
> last 40—60 seconds; it does **not** require each step to get an equal share. Using
> absolute seconds would misjudge "a step done quickly while the procedure is complete"
> as failing. The ratio criterion is anchored on the "fair share" each step is entitled
> to, which is closer to the intent of the guideline.
> The report must state which criterion was used.

### 4.4 Repeated step (repeated)

A step that occurs more than once in the action sequence → `severity="info"`.
With `assess.allow_repeats: true` it is no longer reported.

### 4.5 Optional faucet events

When `assess.require_faucet_events` is true, the smoothed sequence must contain both
`faucet_on` and `faucet_off`; each missing event produces a warning. These supplemental
warnings do **not** change `is_complete` or `overall_score`, which describe the WHO six
hand-rubbing steps. The configured model label space must contain both event classes.
PSKUS has no `faucet_on` annotation, so its configuration keeps this option false.

---

## 5. Overall score (overall_score, 0—1)

```
score = 0.5 × six-step coverage
      + 0.3 × (order correct ? 1 : 0)
      + 0.2 × min(1, total wash time / reference_total_duration_s)
```

- The weights are **fixed in code** (`core/protocol.py::_overall_score`) and are not put
  into the configuration — otherwise team members using different weights would obtain
  incomparable scores.
- The denominator `reference_total_duration_s` is configurable (default 50 s).
- The score is only for "cross-sectional comparison under one and the same set of
  thresholds"; it is **not a clinical metric**, and the report must say so.

---

## 6. The correct way to adjust thresholds

**Change the configuration, not the code.**

```bash
# a one-off experiment
python -m handwash.cli assess --video demo.mp4 assess.min_total_duration_s=30

# a long-term setting for one dataset -> write it into the assess section of configs/data/<name>.yaml
```

If what you want to change is the **judgement rule itself** (for example "allow going
back to redo a step without calling it out-of-order"), that is an L1 change: write an RFC
first (see section 5 of `ARCHITECTURE.md`), then change `core/protocol.py`, and update
this document and `CHANGELOG.md` along with it.

---

## 7. Four things the report must disclose

1. **Criteria**: whether `duration_check` uses `seconds` or `ratio`, and what each
   threshold is;
2. **Frame rate**: the completeness judgement is based on the effective frame rate after
   frame extraction (`data.prep.fps`), not on the original fps;
3. **Model**: which checkpoint was used (`config_hash` + epoch), and whether inference
   was per-frame or temporal;
4. **Failure cases**: at least 2 examples where the model judged wrongly, saying whether
   the difficulty was visual (occlusion / similar actions) or procedural (out-of-order /
   too fast). This section usually shows the depth of the work better than any other.

---

## 8. Known limitations (writing them into the report proactively beats being asked)

- **Normative analysis only, no medical judgement**: we do not assess whether the washing
  meets a disinfection standard.
- **Camera-position dependent**: when the hands are not clearly visible, the
  discriminability of the six steps drops sharply (especially when the camera position
  differs greatly from that of public datasets).
- **Ambiguity in the order judgement**: real hand-washing includes "going back to redo",
  which the framework judges as out-of-order by default; if you consider that
  unreasonable, change the rule through an RFC and explain it in the report.
- **The total-duration threshold is a reference value**: the WHO figure of 40—60 seconds
  is a recommendation for a complete procedure, not a hard standard, so the framework
  treats it as an `error`-level warning rather than a "failing" judgement.
