> **English** | [中文](/docs/zh/PROJECT_PLAN_4_WEEK.md)

# Four-Week Team Plan (4 members) — PRS Practice Module

Derived from the official brief, kept in this repo as
[`docs/PRS-Practice-Module-brief.pdf`](PRS-Practice-Module-brief.pdf) (NUS-ISS, © 2026 National
University of Singapore — internal course material, not for redistribution).
**Every task below is anchored to a graded item.** Nothing is busywork.

---

## 0. The grading contract (read this first)

### 0.1 Where the marks are

| Assessment component | Weight |
| --- | --- |
| Practice Module project work (documentation + MVP deliverables) | 50% |
| Written examination (3 h, open book) | 50% |
| **Pass mark: 50% overall** | |

Breakdown of the 50% project work — **this is the plan's backbone**:

| Item | Weight | Delivered in | Covers |
| --- | --- | --- | --- |
| First presentation | 5% | Week 1 | goals, data resources, techniques/tools, progress |
| Final presentation | 10% | Week 4 | 10–15 min recorded video presentation |
| Final report | 15% | Week 4 | tools/techniques, system design, performance, findings |
| Final system | 15% | Week 4 | **runnable** pattern recognition system + datasets + code |
| Peer review | 5% | Week 4 | submitted through the peer review system |

### 0.2 Hard requirements from the brief

1. Team of **max 5 members** — you have 4.
2. The project must **develop, integrate and demonstrate at least 3 of these 4 aspects**:

   | # | Aspect | How this project satisfies it |
   | --- | --- | --- |
   | A1 | Supervised / unsupervised learning | **Supervised**: frame-level 6-class classification, video-level train/val/test split |
   | A2 | Machine learning / deep learning | **Deep learning**: YOLO26n-cls + MobileNetV2 (transfer learning) |
   | A3 | Hybrid ML / ensemble approach | **Hybrid + ensemble**: frame model × temporal model (GRU/TCN) fused by probability smoothing and majority voting |
   | A4 | Intelligent sensing / sense making | **Sense making**: video as a sensor stream; temporal segmentation turns per-frame labels into an action sequence and a compliance verdict |

   **Plan for all four, guarantee three.** A4 is the one that most often collapses under deadline
   pressure, so it gets its own dedicated owner (see §2) and its deliverable is the demo — which you
   must have anyway for the final video.
3. Deliverables: runnable system, datasets, final report, **10–15 min** video presentation,
   slides for both presentations, code/model files.
4. **Additional:** 1–2 page individual report **per member** (personal contribution; what you
   learnt that is most useful; how you will apply it elsewhere) + peer review via the system.
5. Submission: 31 Oct 2026, via Canvas → Assignments → Practice Module.
   **One ZIP per team**, except the individual reports.

### 0.3 Calendar anchors

| Course date | What |
| --- | --- |
| 15 Sep 2026 | Project proposal deadline *(already past — confirm it was submitted)* |
| **30 Sep 2026, 18:30–22:30** | **First presentation, via Zoom** (5%) |
| **31 Oct 2026** | All deliverables (10% + 15% + 15% + 5%) + individual reports |

**This plan runs Weeks 1–4. Pin it to real dates by setting your own Week-1 Monday**, e.g.
Week 1 = 21–27 Sep, Week 2 = 28 Sep–4 Oct, Week 3 = 5–11 Oct, Week 4 = 12–18 Oct.
That leaves **~2 buffer weeks before 31 Oct** — use them for the exam, re-shoots, and the
final polish. Do not let the buffer tempt you into sliding the plan.

> The brief estimates **10 days of effort** for the module. Across 4 people × 4 weeks that is
> **~2.5 focused days per person per week**. If a week is running long, cut scope — never cut the
> final system, the report, or the video.

---

## 1. Shared working rules (non-negotiable)

These exist so that four people can work in parallel without breaking each other's work.

| Rule | Why | Where it is enforced |
| --- | --- | --- |
| Every change goes through a pull request | Two people editing one file = lost work | `CONTRIBUTING.md` §5 |
| Never touch `src/handwash/core/` without an RFC | It is the contract layer; changing it invalidates everyone's results | `CODEOWNERS`, `scripts/check_structure.py` |
| Split data by **source video**, never by frame | Frame-level random splitting leaks data and inflates accuracy | `io/split.py`, 3 guard checks |
| Every experiment records its `config_hash` | Otherwise results are not comparable | `docs/EXPERIMENTS.md` |
| Run `make check` before every PR | CI and local must agree | `Makefile`, `.github/workflows/ci.yml` |

**Meeting rhythm:** 30-min sync every Monday (plan the week, assign PRs) and a 15-min
checkpoint every Thursday (unblock). Written decision log in `docs/EXPERIMENTS.md` — if it is not
written down, it did not happen.

---

## 2. Four roles (one owner per workstream, no shared ownership)

Shared ownership means nobody owns it. Each workstream has exactly one owner; others review.

| Role | Owner (fill in) | Owns | Primary modules / files | Graded item it feeds |
| --- | --- | --- | --- | --- |
| **R1 — Data & Evaluation Lead** | | datasets, splits, metrics, evaluation protocol, statistical claims | `src/handwash/io/`, `src/handwash/data/`, `scripts/prepare_data.py`, `docs/DATA.md`, `docs/EXPERIMENTS.md` | Final system, final report (performance) |
| **R2 — Model & Training Lead** | | frame classifier, training pipeline, baselines, hyper-parameters | `src/handwash/models/`, `src/handwash/pipelines/train.py`, `configs/models/`, `configs/experiments/` | Final system, final report (design) |
| **R3 — Temporal & Assessment Lead** | | temporal fusion, completeness rules, cross-scenario robustness | `src/handwash/models/temporal.py`, `src/handwash/core/protocol.py`, `configs/config.yaml` (`assess`) | Final system (the differentiator) |
| **R4 — Product, Demo & Delivery Lead** | | runnable demo, video, slides, report integration, ZIP, peer review logistics | `scripts/run_assess.py`, `docs/REPORT_OUTLINE.md`, `deliverables/` | First presentation, final presentation, final report, peer review |

**Why this split:** R1 and R2 unblock each other on day one (splits → first baseline).
R3 depends on R2's checkpoint but can build and unit-test the temporal head against synthetic
data in Week 1. R4 has the earliest deadline (30 Sep presentation) and the latest (31 Oct ZIP),
which is exactly why the demo owner must also own delivery — one throat to choke.

**Load balance check:** each role is ~2.5 days/week. If any role exceeds 4 days in a week,
that is a planning bug — raise it in the Monday sync.

---

## 3. Week-by-week plan

Legend: **[R1]**…**[R4]** = the role that owns the task. Tasks without a bracket are whole-team.

### Week 1 — Foundations: data flows, first baseline runs, first presentation

**Goal:** by Sunday, a real dataset trains end to end on one GPU box, and everyone can run
`make check` green. First presentation is delivered on 30 Sep, so Week 1 must also produce slides.

| # | Task | Owner | Deliverable (evidence) |
| --- | --- | --- | --- |
| 1.1 | Download the small Kaggle dataset; run `prepare_data.py --inspect`; verify label mapping | [R1] | `data/processed/kaggle/split_report.json` |
| 1.2 | Start the PSKUS subset download (300–500 clips) in the background — **do not wait for it** | [R1] | download log |
| 1.3 | Environment check: `python scripts/doctor.py` green on every member's machine | all | 4 screenshots in the team channel |
| 1.4 | Run the smoke test, then the first real baseline (`exp01`) | [R2] | `outputs/e1_*/eval/eval_val.json` |
| 1.5 | Build and unit-test the temporal head on synthetic data (no checkpoint needed yet) | [R3] | `pytest tests/unit -k temporal` green |
| 1.6 | Draft the first-presentation deck: goals, data resources, techniques/tools, progress | [R4] | `deliverables/slides_first_presentation.pdf` |
| 1.7 | Freeze the label space and the assessment criteria; write them down | [R3] | `docs/PROTOCOL.md` reviewed & merged |
| 1.8 | Weekly sync + fill `docs/EXPERIMENTS.md` with the baseline row (incl. `config_hash`) | all | experiment log entry |

**Week-1 exit criteria**
- [ ] `make check` green on all four machines
- [ ] One baseline row in `docs/EXPERIMENTS.md` with a real Accuracy **and** Macro-F1
- [ ] The six classes are confirmed identical across datasets (no label misalignment)
- [ ] First-presentation slides ready **3 days before** 30 Sep (so there is time to rehearse)

> **Risk to watch:** PSKUS is 18.4 GB. If the download is not done by Week 1 end, do not block —
> proceed on Kaggle and swap the dataset in Week 2. Say so explicitly in the experiment log.

### Week 2 — Real data at scale: full training, temporal integration, cross-scenario setup

**Goal:** the primary model (YOLO26n-cls + temporal) trains on real hospital data, and the
evaluation harness produces every number the report needs.

| # | Task | Owner | Deliverable (evidence) |
| --- | --- | --- | --- |
| 2.1 | Prepare the PSKUS subset: split by source video, extract frames, rebuild manifest | [R1] | `split_report.json` showing train/val/test clip counts |
| 2.2 | Leakage audit: run the guard checks, confirm no source video crosses splits | [R1] | audit note in `docs/EXPERIMENTS.md` |
| 2.3 | Train the primary model (`exp02`: YOLO26n-cls + GRU); log all curves | [R2] | `outputs/e2_*/models/best.pt` + `history.json` |
| 2.4 | Baseline matrix: MobileNetV2 (frame) and YOLOv8n-cls (frame) on the same split | [R2] | 3 comparable rows in the experiment log |
| 2.5 | Integrate the temporal head with the trained checkpoint (freeze backbone → train head) | [R3] | `outputs/e2_temporal/` |
| 2.6 | Wire `assess` end-to-end on a real clip: report JSON + Markdown | [R3] | `outputs/*/assess/<clip>.md` |
| 2.7 | Evaluate on METC as the cross-scenario split | [R1] | `eval/eval_external.json` |
| 2.8 | Record a first rough demo capture (phone, one correct + one deliberate mistake) | [R4] | raw `.mp4` files in `data/external/self_recorded/` |
| 2.9 | Draft the report skeleton: tools/techniques, system design, performance, findings | [R4] | `deliverables/report_draft_v1.pdf` |

**Week-2 exit criteria**
- [ ] Same-split comparison table exists: ≥ 3 models, Accuracy + Macro-F1 + confusion matrix
- [ ] The temporal-vs-frame ablation is running (not necessarily finished)
- [ ] `assess` produces a human-readable report for at least one real video
- [ ] Cross-scenario number exists (even if poor — a poor number is a *finding*, not a failure)

### Week 3 — The differentiator: temporal ablation, completeness evaluation, robustness

**Goal:** produce the evidence that answers the research questions. This is the week that turns
"we trained a classifier" into "we built a pattern recognition **system**".

| # | Task | Owner | Deliverable (evidence) |
| --- | --- | --- | --- |
| 3.1 | Run the four-way temporal ablation: none / probability smoothing / mean-pool / GRU or TCN | [R3] | 4 rows + a chart of macro-F1 and label-flip count |
| 3.2 | Quantify "fewer single-frame misjudgements": count label transitions before vs after | [R3] | `deliverables/flicker_reduction.csv` |
| 3.3 | Cross-scenario analysis: how much does Macro-F1 drop PSKUS → METC, and on which classes | [R1] | per-class drop table |
| 3.4 | Bootstrap confidence intervals so the claimed improvement is not sampling noise | [R1] | CI values in the experiment log |
| 3.5 | Hyper-parameter sweep for the primary model (LR, window, class weights); keep it bounded | [R2] | sweep table + chosen config |
| 3.6 | Error analysis: pick 5 failure cases, classify each as visual (occlusion/similar action) or procedural (order/speed) | [R2] | `deliverables/error_analysis.md` |
| 3.7 | Self-recorded validation set: 4–6 clips per member covering correct / missing / swapped / too-short | all | clips + `docs/SELF_RECORDING.md` table |
| 3.8 | Ground-truth comparison: model verdict vs human verdict, completeness + order accuracy | [R4] | `assess_vs_ground_truth.csv` |
| 3.9 | Assemble the demo so a stranger can run it from the README in under 5 minutes | [R4] | demo run script + `docs/DEMO.md` |

**Week-3 exit criteria**
- [ ] The report's central table (the ablation) is complete and reproducible
- [ ] Every number in it traces to a `config_hash` in `docs/EXPERIMENTS.md`
- [ ] At least 2 honest failure cases documented with a cause
- [ ] Cross-scenario performance drop quantified and discussed, not hidden

> **This week decides the grade quality.** The brief rewards "findings and discussions".
> A 0.84 Macro-F1 with a clean ablation, an honest error analysis and a cross-scenario study
> scores better than 0.91 with no analysis — and the second is also more likely to be leakage.

### Week 4 — Consolidate: final system, video, report, submissions

**Goal:** ship. No new experiments this week — **freeze features at the start of Week 4.**

| # | Task | Owner | Deliverable (evidence) |
| --- | --- | --- | --- |
| 4.1 | **Feature freeze** at Monday sync; only bug fixes afterwards | all | decision recorded in experiment log |
| 4.2 | Freeze the runnable system: one command installs, one command runs the demo | [R4] | verified on a **clean machine / fresh env** |
| 4.3 | Package datasets + code + model files into `deliverables/` | [R1] | dataset README + download instructions |
| 4.4 | Final report: all sections written, figures finalised, references complete | [R4] + all | `deliverables/final_report.pdf` |
| 4.5 | Each member writes their **1–2 page individual report** | all | 4 separate PDFs |
| 4.6 | Record the **10–15 min** video presentation; check the timer | [R4] | `deliverables/final_presentation.mp4` |
| 4.7 | Final slides | [R4] | `deliverables/slides_final_presentation.pdf` |
| 4.8 | Peer review submitted by every member | all | submission receipts |
| 4.9 | Build the single team ZIP; verify it opens and the demo runs from it | [R4] | `deliverables/PRS_GroupX_submission.zip` |
| 4.10 | Dry-run the whole submission on a second machine, 3 days before the deadline | all | checklist signed off |

**Week-4 exit criteria**
- [ ] ZIP submitted to Canvas (**one per team**) — before 31 Oct
- [ ] 4 individual reports submitted separately
- [ ] 4 peer reviews submitted
- [ ] Video is 10–15 min and plays from the ZIP
- [ ] The system runs from the ZIP on a machine that has never seen the repo

---

## 4. Requirement traceability matrix

Fill this table in Week 4 and paste it into the report's appendix. It is the fastest way for a
grader to confirm you met every requirement.

| Requirement (from the brief) | Evidence artefact | Owner | Status |
| --- | --- | --- | --- |
| Runnable pattern recognition system | `deliverables/…zip` → one-command demo | R4 | |
| Datasets | `docs/DATA.md` + dataset README in ZIP | R1 | |
| Final report: tools/techniques | report §Method | R4 | |
| Final report: system design / models | report §System design + `docs/ARCHITECTURE.md` | R2 | |
| Final report: system performance | report §Results + `docs/EXPERIMENTS.md` | R1 | |
| Final report: findings and discussions | report §Discussion + error analysis | R3 | |
| Code + model files | `src/`, `configs/`, `models/best.pt` | R2 | |
| Video presentation 10–15 min | `final_presentation.mp4` | R4 | |
| Slides for two presentations | `slides_first_*.pdf`, `slides_final_*.pdf` | R4 | |
| Individual report per member (1–2 pages) | 4 PDFs | all | |
| Peer review | peer review system receipts | all | |
| Aspect A1 supervised/unsupervised | `docs/PROTOCOL.md` + split design | R1 | |
| Aspect A2 ML/DL techniques | `docs/ARCHITECTURE.md` + model configs | R2 | |
| Aspect A3 hybrid/ensemble | ablation table (frame × temporal fusion) | R3 | |
| Aspect A4 intelligent sensing / sense making | `assess` pipeline: video → verdict | R3 | |

**Rule:** if an artefact is empty in this table at the Week-4 Monday sync, that requirement is
at risk. Fix it that week or cut scope elsewhere — do not leave it to the last day.

---

## 5. Risk register

| Risk | Likelihood | Impact | Mitigation (owner) |
| --- | --- | --- | --- |
| PSKUS (18.4 GB) download/prep not finished | High | High | Start in Week 1; proceed on Kaggle; treat "small-data results" as a legitimate first milestone (R1) |
| yolo26 weights unavailable | High | Medium | Adapter already supports any YOLO classification weight; switch by config and **state the actual weights used** in the report (R2) |
| Two members edit the same module | Medium | High | PR-only + CODEOWNERS; Monday sync assigns files before work starts (all) |
| Results not reproducible on another machine | Medium | High | `config_hash` + fixed seed + `environment.yml`; clean-machine dry run in Week 4 (R4) |
| Accuracy looks great but is leakage | Medium | Fatal | Three guard checks; report the split design explicitly (R1) |
| Video/report left to the last days | High | High | Draft skeleton in Week 2, feature freeze Week 4, dry run 3 days early (R4) |
| One member overloaded / unavailable | Medium | Medium | Roles are documented, so a task can be reassigned in one sync; each artefact has a written spec (all) |
| Exam collides with the final week | High | Medium | This is exactly what the 2 buffer weeks before 31 Oct are for — do not spend them early |

---

## 6. How the grading components map to repo artefacts

| Graded item | Weight | Where it lives in this repo |
| --- | --- | --- |
| First presentation | 5% | `deliverables/slides_first_presentation.pdf`; content sourced from `README.md` + `docs/DATA.md` |
| Final presentation | 10% | `deliverables/final_presentation.mp4` (10–15 min) |
| Final report | 15% | `deliverables/final_report.pdf`; every number traced to `docs/EXPERIMENTS.md` |
| Final system | 15% | the repo itself; `docs/DEMO.md` is the runnable proof |
| Peer review | 5% | submitted externally; self-assessment evidence in each individual report |

---

## 7. Definition of done (per person, Week 4)

- [ ] I have at least one **merged PR** for every week of the project
- [ ] Every artefact I own appears in the traceability matrix with a real path
- [ ] My individual report (1–2 pages) covers: my contribution / what I learnt that is most
      useful / how I will apply it elsewhere
- [ ] I submitted my peer review
- [ ] I can explain, without notes, **any** number in the final report
- [ ] I can run the demo from the ZIP on a machine that has never seen this repo
