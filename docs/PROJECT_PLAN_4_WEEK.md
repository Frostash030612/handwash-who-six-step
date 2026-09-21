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
| First presentation | 5% | Week 2 | goals, data resources required/available, techniques/tools used, progress |
| Final presentation | 10% | Week 4 | 10–15 min recorded video presentation |
| Final report | 15% | Week 4 | tools/techniques, system design/models, system performance, findings and discussions |
| Final system | 15% | Week 4 | **runnable** pattern recognition system + datasets + code + model files |
| Peer review | 5% | Week 4 | submitted through the peer review system |

### 0.2 Hard requirements from the brief

1. Team of **max 5 members**, and **enrol in Canvas project groups** — you have 4; confirm the
   Canvas group ID (you need it on the proposal cover).
2. The project must **develop, integrate and demonstrate at least 3 of these 4 aspects**:

   | # | Aspect | How this project satisfies it |
   | --- | --- | --- |
   | A1 | Supervised / unsupervised learning | **Supervised**: frame-level 6-class classification; video-level train/val/test split |
   | A2 | Machine learning / deep learning | **Deep learning**: YOLO26n-cls + MobileNetV2 (transfer learning) |
   | A3 | Hybrid ML / ensemble approach | **Hybrid + ensemble**: frame model × temporal model (GRU/TCN) fused by probability smoothing and majority voting |
   | A4 | Intelligent sensing / sense making | **Sense making**: video as a sensor stream; temporal segmentation turns per-frame labels into an action sequence and a compliance verdict |

   **Plan for all four, guarantee three.** A4 is the one that most often collapses under deadline
   pressure, so it gets its own dedicated owner (see §2) and its deliverable is the demo — which you
   must have anyway for the final video.
3. Deliverables: runnable system, datasets, final report, **10–15 min** video presentation
   (`.mp4/.mov/.wmv`), slides for **both** presentations, code + model files.
4. **Additional submissions:** 1–2 page individual report **per member** (personal contribution;
   what you learnt that is most useful; how you will apply it elsewhere) + peer review via the
   peer review system.
5. Submission: **31 Oct 2026**, Canvas → Assignments → Practice Module.
   **One ZIP per team**, except the individual reports, which are submitted separately.

### 0.3 Calendar anchors — **all dates on the right are Week 2 and later**

| Date | What | Owner |
| --- | --- | --- |
| **30 Sep 2026** | **Project Proposal submission deadline** (template: `docs/Project_Proposal_Template.docx`) | R4 (lead) + all |
| **6 Oct 2026, 09:00–18:00** | **First presentation, via Zoom** (5%) | R4 (lead) + all |
| **31 Oct 2026** | All deliverables (10% + 15% + 15% + 5%) + individual reports + peer review | all |

**The four weeks are counted backwards from 31 Oct.** Set your Week-1 Monday so that Week 4 ends
comfortably before the deadline — the recommended mapping is:

| Week | Dates | Focus | Course event |
| --- | --- | --- | --- |
| Week 1 | 22–28 Sep | Proposal + data pipeline + smoke runs | **Proposal due 30 Sep** |
| Week 2 | 29 Sep–5 Oct | First presentation + real-data baseline | **First presentation 6 Oct** |
| Week 3 | 6–12 Oct | Primary model, temporal integration, cross-scenario | — |
| Week 4 | 13–19 Oct | Ablation, error analysis, demo, report skeleton | — |
| Buffer | 20–31 Oct | Final system freeze, video, report, ZIP, peer review, **exam** | **Deliverables 31 Oct** |

> The brief estimates **10 days of effort** for the module. Across 4 people × 4 weeks that is
> **~2.5 focused days per person per week**. If a week runs long, cut scope — never cut the final
> system, the report, or the video.
>
> **Do not treat the buffer as slack.** Weeks 1–4 produce a submission that is at least 80%
> complete; the buffer weeks exist for the exam, re-shoots, and the clean-machine rehearsal.
> If you let the plan slide into the buffer, you lose that safety margin entirely.

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

> **This week's exception:** with the proposal due 30 Sep, hold a 45-min **kick-off** on day one
> to agree the project title, the 3-of-4 aspect story, and who writes which proposal section.
> Everything else in Week 1 is downstream of that 45 minutes.

---

## 2. Four roles (one owner per workstream, no shared ownership)

Shared ownership means nobody owns it. Each workstream has exactly one owner; others review.

| Role | Owner | Owns | Primary modules / files | Graded item it feeds |
| --- | --- | --- | --- | --- |
| **R1 — Data & Evaluation Lead** | **Shen Ziyi** (A0350940J) | datasets, splits, metrics, evaluation protocol, statistical claims | `src/handwash/io/`, `src/handwash/data/`, `scripts/prepare_data.py`, `docs/DATA.md`, `docs/EXPERIMENTS.md` | Final system, final report (performance) |
| **R2 — Model & Training Lead** | **Wang Lepeng** (A0357864L) | frame classifier, training pipeline, baselines, hyper-parameters | `src/handwash/models/`, `src/handwash/pipelines/train.py`, `configs/models/`, `configs/experiments/` | Final system, final report (design) |
| **R3 — Temporal & Assessment Lead** | **Zhu Jianyu** (A0353769L) | temporal fusion, completeness rules, cross-scenario robustness | `src/handwash/models/temporal.py`, `src/handwash/core/protocol.py`, `configs/config.yaml` (`assess`) | Final system (the differentiator) |
| **R4 — Product, Demo & Delivery Lead** | **Xu Wenzhe** (A0328771W) | proposal, runnable demo, video, slides, report integration, ZIP, peer review logistics | `docs/PROJECT_PROPOSAL.md`, `scripts/run_assess.py`, `deliverables/` | **Proposal, first presentation**, final presentation, final report, peer review |

Group ID (Canvas): **43**. Watch the proposal deadline: **30 Sep 2026** — see
[`PROJECT_PROPOSAL.md`](PROJECT_PROPOSAL.md), which already contains the filled proposal content.

> **Suggested assignment — confirm or swap it at the kick-off meeting.** It is aligned with what
> each person would touch anyway, and it front-loads the critical path: R1 owns the foundations
> every other result depends on (the split must be correct before any accuracy number means
> anything), R2 owns the model, R3 owns the differentiator, R4 owns the two submission gates
> (30 Sep proposal, 6 Oct presentation) plus the final delivery.
> **If you swap, swap the whole workstream**, including the files listed above.
>
> **R4 is the only role with a hard deadline inside Week 1.** Whoever holds it must be reachable
> in the days before 30 Sep.

**Why this split:** R1 and R2 unblock each other on day one (splits → first baseline).
R3 depends on R2's checkpoint but can build and unit-test the temporal head against synthetic
data in Week 1. R4 owns the two submission gates (30 Sep proposal, 6 Oct presentation) and the
two final ones (video, ZIP) — one throat to choke.

**Load balance check:** each role is ~2.5 days/week. If any role exceeds 4 days in a week,
that is a planning bug — raise it in the Monday sync.

---

## 3. Week-by-week plan

Legend: **[R1]**…**[R4]** = the role that owns the task. Tasks without a bracket are whole-team.

### Week 1 (22–28 Sep) — Proposal + foundations + first runs

**Goal:** the proposal is submitted, a real dataset trains end to end on one GPU box, and
everyone can run `make check` green.

| # | Task | Owner | Deliverable (evidence) |
| --- | --- | --- | --- |
| 1.1 | **Kick-off (45 min): confirm or swap the §2 role assignment, agree the 3-of-4 aspect story** | all | §2 table confirmed |
| 1.2 | Confirm the Canvas group ID (**43**) and that all 4 members are enrolled | [R4] | Canvas group screenshot |
| 1.3 | **Draft the proposal into `docs/Project_Proposal_Template.docx`** — see `docs/PROJECT_PROPOSAL.md` for ready-to-paste content | [R4] + all | `deliverables/project_proposal.pdf` |
| 1.4 | **Submit the proposal to Canvas — deadline 30 Sep. Do not leave this to the 30th** | [R4] | submission receipt |
| 1.5 | Download the small Kaggle dataset; run `prepare_data.py --inspect`; verify label mapping | [R1] | `data/processed/kaggle/split_report.json` |
| 1.6 | Start the PSKUS subset download (300–500 clips) in the background — **do not wait for it** | [R1] | download log |
| 1.7 | Environment check: `python scripts/doctor.py` green on every member's machine | all | 4 screenshots in the team channel |
| 1.8 | Run the smoke test, then the first real baseline (`exp01`) | [R2] | `outputs/e1_*/eval/eval_val.json` |
| 1.9 | Build and unit-test the temporal head on synthetic data (no checkpoint needed yet) | [R3] | `pytest tests/unit -k temporal` green |
| 1.10 | **Draft the first-presentation deck** (goals, data required/available, techniques/tools, progress) — the proposal content is 70% of it | [R4] | `deliverables/slides_first_presentation.pdf` |
| 1.11 | Freeze the label space and the assessment criteria; write them down | [R3] | `docs/PROTOCOL.md` reviewed & merged |
| 1.12 | Weekly sync + fill `docs/EXPERIMENTS.md` with the baseline row (incl. `config_hash`) | all | experiment log entry |

**Week-1 exit criteria**
- [ ] Proposal **submitted** (not merely drafted) — this is the date that cannot slip
- [ ] Canvas group ID recorded and all 4 members enrolled
- [ ] `make check` green on all four machines
- [ ] One baseline row in `docs/EXPERIMENTS.md` with a real Accuracy **and** Macro-F1
- [ ] The six classes are confirmed identical across datasets (no label misalignment)
- [ ] First-presentation slides at draft-2 stage (only polish left for Week 2)

> **Risk to watch:** PSKUS is 18.4 GB. If the download is not done by Week 1 end, do not block —
> proceed on Kaggle and swap the dataset in Week 3. Say so explicitly in the experiment log.

### Week 2 (29 Sep–5 Oct) — First presentation + real-data baseline

**Goal:** deliver the first presentation on 6 Oct and start training the primary model on real
hospital data.

| # | Task | Owner | Deliverable (evidence) |
| --- | --- | --- | --- |
| 2.1 | Rehearse the presentation twice; time it; agree who speaks to which slide | all | rehearsal notes |
| 2.2 | **Deliver the first presentation — 6 Oct, 09:00–18:00, Zoom** (5%) | [R4] lead | slides + attendance |
| 2.3 | Capture supervisor feedback verbatim and turn it into tasks | [R4] | feedback note in `docs/EXPERIMENTS.md` |
| 2.4 | Prepare the PSKUS subset: split by source video, extract frames, rebuild manifest | [R1] | `split_report.json` showing train/val/test clip counts |
| 2.5 | Leakage audit: run the guard checks, confirm no source video crosses splits | [R1] | audit note in `docs/EXPERIMENTS.md` |
| 2.6 | Train the primary model (`exp02`: YOLO26n-cls + GRU); log all curves | [R2] | `outputs/e2_*/models/best.pt` + `history.json` |
| 2.7 | Baseline matrix: MobileNetV2 (frame) and YOLOv8n-cls (frame) on the same split | [R2] | 3 comparable rows in the experiment log |
| 2.8 | Integrate the temporal head with the trained checkpoint (freeze backbone → train head) | [R3] | `outputs/e2_temporal/` |
| 2.9 | Wire `assess` end-to-end on a real clip: report JSON + Markdown | [R3] | `outputs/*/assess/<clip>.md` |
| 2.10 | Evaluate on METC as the cross-scenario split | [R1] | `eval/eval_external.json` |
| 2.11 | Record a first rough demo capture (phone, one correct + one deliberate mistake) | [R4] | raw `.mp4` files in `data/external/self_recorded/` |

**Week-2 exit criteria**
- [ ] First presentation delivered; feedback captured as tasks
- [ ] Same-split comparison table exists: ≥ 3 models, Accuracy + Macro-F1 + confusion matrix
- [ ] The temporal-vs-frame ablation is running (not necessarily finished)
- [ ] `assess` produces a human-readable report for at least one real video
- [ ] Cross-scenario number exists (even if poor — a poor number is a *finding*, not a failure)

### Week 3 (6–12 Oct) — The differentiator: temporal ablation, completeness evaluation, robustness

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
| 3.10 | Draft the report: tools/techniques, system design, performance, findings | [R4] + all | `deliverables/report_draft_v1.pdf` |

**Week-3 exit criteria**
- [ ] The report's central table (the ablation) is complete and reproducible
- [ ] Every number in it traces to a `config_hash` in `docs/EXPERIMENTS.md`
- [ ] At least 2 honest failure cases documented with a cause
- [ ] Cross-scenario performance drop quantified and discussed, not hidden

> **This week decides the grade quality.** The brief rewards "findings and discussions".
> A 0.84 Macro-F1 with a clean ablation, an honest error analysis and a cross-scenario study
> scores better than 0.91 with no analysis — and the second is also more likely to be leakage.

### Week 4 (13–19 Oct) — Consolidate: final system, video, report, submissions

**Goal:** reach a submission-ready state. No new experiments this week — **freeze features at the
start of Week 4.**

| # | Task | Owner | Deliverable (evidence) |
| --- | --- | --- | --- |
| 4.1 | **Feature freeze** at Monday sync; only bug fixes afterwards | all | decision recorded in experiment log |
| 4.2 | Freeze the runnable system: one command installs, one command runs the demo | [R4] | verified on a **clean machine / fresh env** |
| 4.3 | Package datasets + code + model files into `deliverables/` | [R1] | dataset README + download instructions |
| 4.4 | Final report: all sections written, figures finalised, references complete | [R4] + all | `deliverables/final_report.pdf` (draft-final) |
| 4.5 | Each member drafts their **1–2 page individual report** | all | 4 drafts |
| 4.6 | Record the **10–15 min** video presentation; check the timer | [R4] | `deliverables/final_presentation.mp4` |
| 4.7 | Final slides | [R4] | `deliverables/slides_final_presentation.pdf` |
| 4.8 | Build the single team ZIP; verify it opens and the demo runs from it | [R4] | `deliverables/PRS_GroupX_submission.zip` |

**Week-4 exit criteria**
- [ ] End-to-end submission exists and runs from the ZIP
- [ ] Report is draft-final, video is recorded, slides are final
- [ ] Everything is reproducible from `docs/EXPERIMENTS.md` + `config_hash`

### Buffer weeks (20–31 Oct) — Submission, exam, and the clean-machine rehearsal

Not new work — verification, polish and the exam.

| # | Task | Owner | Deliverable |
| --- | --- | --- | --- |
| B.1 | Dry-run the whole submission on **a second machine that has never seen the repo** | all | signed checklist |
| B.2 | Finalise the 4 individual reports | all | 4 PDFs |
| B.3 | Submit peer review (every member) | all | receipts |
| B.4 | **Submit the ZIP to Canvas (one per team) + individual reports — before 31 Oct** | [R4] | submission receipt |
| B.5 | Sit the written examination (3 h, open book) | all | — |

---

## 4. Requirement traceability matrix

Fill this table during Week 4 and paste it into the report's appendix. It is the fastest way for a
grader to confirm you met every requirement.

| Requirement (from the brief) | Evidence artefact | Owner | Status |
| --- | --- | --- | --- |
| Project proposal (deadline 30 Sep) | `deliverables/project_proposal.pdf` (from `docs/Project_Proposal_Template.docx`) | R4 | |
| Enrolled in Canvas project groups | Canvas group screenshot | R4 | |
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
| **Proposal (30 Sep) missed while everyone focuses on modelling** | Medium | **Fatal** | It is task 1.3/1.4 in Week 1 and it is R4's only Week-1 priority; the technical work runs in parallel (R4) |
| PSKUS (18.4 GB) download/prep not finished | High | High | Start in Week 1; proceed on Kaggle; treat "small-data results" as a legitimate first milestone (R1) |
| yolo26 weights unavailable | High | Medium | Adapter already supports any YOLO classification weight; switch by config and **state the actual weights used** in the report (R2) |
| Two members edit the same module | Medium | High | PR-only + CODEOWNERS; Monday sync assigns files before work starts (all) |
| Results not reproducible on another machine | Medium | High | `config_hash` + fixed seed + `environment.yml`; clean-machine dry run in the buffer weeks (R4) |
| Accuracy looks great but is leakage | Medium | Fatal | Three guard checks; report the split design explicitly (R1) |
| Video/report left to the last days | High | High | Report skeleton in Week 3, feature freeze Week 4, buffer weeks for polish (R4) |
| Exam collides with deliverable week | High | Medium | Buffer weeks 20–31 Oct exist for exactly this; do not spend them early (all) |
| One member overloaded / unavailable | Medium | Medium | Roles are documented, so a task can be reassigned in one sync; each artefact has a written spec (all) |

---

## 6. How the grading components map to repo artefacts

| Graded item | Weight | Where it lives in this repo |
| --- | --- | --- |
| Project proposal | gate | `deliverables/project_proposal.pdf`; content staged in `docs/PROJECT_PROPOSAL.md` |
| First presentation | 5% | `deliverables/slides_first_presentation.pdf`; content sourced from the proposal + `docs/DATA.md` |
| Final presentation | 10% | `deliverables/final_presentation.mp4` (10–15 min) |
| Final report | 15% | `deliverables/final_report.pdf`; every number traced to `docs/EXPERIMENTS.md` |
| Final system | 15% | the repo itself; `docs/DEMO.md` is the runnable proof |
| Peer review | 5% | submitted externally; self-assessment evidence in each individual report |

---

## 7. Definition of done (per person, by 31 Oct)

- [ ] I have at least one **merged PR** for every week of the project
- [ ] Every artefact I own appears in the traceability matrix with a real path
- [ ] My individual report (1–2 pages) covers: my contribution / what I learnt that is most
      useful / how I will apply it elsewhere
- [ ] I submitted my peer review
- [ ] I can explain, without notes, **any** number in the final report
- [ ] I can run the demo from the ZIP on a machine that has never seen this repo
