# Camera-Based Handwashing Assessment

Group 43: Shen Ziyi, Wang Lepeng, Xu Wenzhe, Zhu Jianyu

**Recording plan:** 8:00 silent PPT video. Insert the separately recorded 5:00 system demonstration at PPT video time 04:20 to make the 13:00 final presentation.

Use the timings below as narration targets. The silent PPT video contains slides 1–6 and 8–11. Cut at 04:20, insert the demonstration, then resume with slide 8. In the assembled presentation, slide 8 begins at 09:20. The animation cues are measured from the start of the 8:00 PPT video. The existing video stays intact; each speaker records audio for the assigned sections.

| Presenter | Slides | PPT video time | Speaking time |
| --- | --- | --- | ---: |
| Shen Ziyi | 1–4 | 00:00–02:30 | 2:30 |
| Wang Lepeng | 5–6, 8 | 02:30–04:20; 04:20–05:25 | 2:55 |
| Zhu Jianyu | Separate system demonstration | Insert after PPT video 04:20 | 5:00 |
| Xu Wenzhe | 9–11 | 05:25–08:00 | 2:35 |

In the assembled 13:00 video, Wang resumes at 09:20 and Xu begins at 10:25. Speaker changes coincide with slide changes, leaving a natural visual pause for the next voice to enter. Record the spoken text only; presenter and video cues are production notes.

## Slide 1: Camera-Based Handwashing Assessment

**Time: 00:00-00:25 (25 seconds)**

**Speaker: Shen Ziyi**

**Video cue:** Hold the cover for the introduction; no object reveal.

Hello, we are Group forty-three. Our project uses a local camera to assess six-step handwashing. It recognizes hand gestures, reconstructs their order, and measures recognized washing time. We will explain the model, demonstrate the application, and discuss the results and deployment trade-offs.

*Presenter cue: Start the presentation timer. Introduce the team without reading every name.*

Source: Final project report, Abstract and Section 1.

## Slide 2: What the system assesses

**Time: 00:25-01:05 (40 seconds)**

**Speaker: Shen Ziyi**

**Video cues:** 00:35 reveal the six-step order; 00:49 reveal the 40-second requirement.

A correct gesture prediction is only one part of the task. A person can perform recognizable actions while skipping a step, changing the order, or stopping too early. Our application therefore evaluates the complete procedure. It recognizes six actions and a background class, builds the observed action sequence, and checks total recognized washing time. Completion requires all six steps in the standard order and at least forty seconds of recognized actions. Background time is excluded. This is an assessment of visible procedure, rather than a direct measurement of cleanliness.

*Presenter cue: Point to the action examples, then to the order and duration requirements.*

Source: Final project report, Sections 1, 2.2 and 6. The 40-second criterion is the application's configured recognized-action threshold.

## Slide 3: The self-recorded dataset

**Time: 01:05-01:55 (50 seconds)**

**Speaker: Shen Ziyi**

**Video cue:** 01:09 reveal the 6,987-frame count.

Our target-scene dataset contains six thousand nine hundred and eighty-seven sampled frames from thirty-two recordings. Sampling at about five frames per second captures motion variation and transitions. The recordings include standard sequences, reordered or missing actions, and isolated examples of steps four and five. We split by video, so neighboring frames stay in the same partition. Fourteen videos support training, eight support validation, and ten form the reported test set. We distinguish definite action labels from uncertain transition frames. Those uncertain frames remain explicit in the annotations instead of becoming background labels by default.

*Presenter cue: Emphasize frame annotation and the split by video. Avoid describing adjacent frames as independent samples.*

Source: Final project report, Tables 2 and 3. dataset/README.md.

## Slide 4: Why we model temporal boundaries

**Time: 01:55-02:30 (35 seconds)**

**Speaker: Shen Ziyi**

**Video cues:** 02:00 reveal the PSKUS chart; 02:15 reveal the +7.74 pp callout.

We first studied the public PSKUS dataset to check feasibility. On that preliminary comparison, macro F1 near transitions increased from eighty-three point three for YOLO to ninety point one with GRU, and ninety-one point zero with early boundary awareness. This supported using temporal context and transition supervision. We then trained the final systems on our self-recorded data. The final model set contains no PSKUS-trained weights. These findings motivate the temporal architecture that Wang Lepeng will explain next.

*Presenter cue: The plot shows preliminary transition-region macro F1, which differs from the final output-boundary F1 metric.*

Source: Final project report, Section 2.1 and Table 1. PSKUS study: 615 videos, 5,599 valid centers, 1,068 transition-region centers.

## Slide 5: YOLO + Adaptive Boundary-Aware GRU

**Time: 02:30-03:30 (60 seconds)**

**Speaker: Wang Lepeng**

**Video cue:** 02:33 reveal the model architecture diagram.

The architecture follows from that feasibility study. YOLO extracts visual features from each sampled frame. A unidirectional GRU carries information forward through the sequence. Two heads share this temporal representation. The classification head estimates the current action, while the boundary head estimates where an action changes. Boundary-guided decoding produces candidate segments, and adaptive confirmation decides which segments belong in the final chain. This separates frame recognition from confirmation of an action instance. The recurrent state also supports incremental processing during a camera session. We compare this complete method with standalone YOLO, ordinary YOLO plus GRU, and VideoMAE. Each baseline answers a different question about visual recognition, temporal context, or video-window modeling.

*Presenter cue: Trace the diagram from the frame thumbnail through the two heads to the final chain. Explain the flow without reading layer dimensions.*

Source: Final project report, Section 3 and Figure 1. docs/MODEL_CARD.md.

## Slide 6: Adaptive action confirmation

**Time: 03:30-04:20 (50 seconds)**

**Speaker: Wang Lepeng**

**Video cues:** 03:34 reveal smoothing; 03:48 reveal candidate evaluation; 04:02 reveal order preservation.

Brief prediction changes can create extra actions even when most frames are correct. We smooth labels over about one second, then group adjacent labels into segments. Adaptive confirmation compares each segment's duration and confidence with the recording's own statistics. An unusually short and uncertain segment can remain unconfirmed. This replaces the fixed three-second rule. With fewer than four comparable segments, duration rejection is skipped. The example removes an extra step four while preserving the observed order. Zhu Jianyu will now demonstrate the application in a complete session.

*Presenter cue: Finish the introduction before 04:20, then start the prepared demonstration video.*

Source: Final project report, Section 3.3 and Algorithm 1. Original Video 8: common-center raw chain 1245346, dense-stream final chain 124536. This is a pipeline comparison, not an isolated ablation.

## Slide 7: Recorded system demonstration

**Time: separate 05:00 demonstration (excluded from the PPT video)**

**Speaker: Zhu Jianyu**

**Video cue:** The five-minute demonstration is supplied as a separate clip.



*Editing cue: Insert Zhu's separately recorded, narrated demonstration at PPT video time 04:20. Resume the PPT video at slide 8 and Wang's narration. Zhu prepares the demonstration narration separately.*

Source: Final project report, Section 6. src/handwash/self_scene_camera.py and src/handwash/self_scene_assessment.py.

## Slide 8: Recognition and boundary results

**Time: 04:20-05:25 (65 seconds)**

**Speaker: Wang Lepeng**

**Video cues:** 04:26 reveal the model table; 05:03 reveal the boundary F1 improvement.

The demonstration showed the complete camera workflow. We now turn to the quantitative results. The comparison uses six hundred and thirty-two aligned valid time centers from ten videos. Accuracy and macro F1 use the same reference labels for every model. Boundary F1 measures predicted changes using one-to-one matching with transition tolerance. Our method reaches ninety-six point five two percent accuracy, ninety-three point seven one macro F1, and ninety-four point four boundary F1. Relative to ordinary GRU, the largest improvement is boundary F1, which rises by six point eight one percentage points. VideoMAE achieves the highest values on all three measures. These results compare complete systems under their actual training protocols. Xu Wenzhe will now examine whether their frame scores translate into correct action chains.

*Presenter cue: Read the proposed row and the boundary improvement. Do not read every table cell.*

Source: evaluation/metrics.csv and final project report, Table 4. Boundary tolerance: annotation transition interval plus 0.4 seconds. Final temporal heads refit on 22 development videos after selection on the 14/8 split.

## Slide 9: Complete action chains

**Time: 05:25-06:10 (45 seconds)**

**Speaker: Xu Wenzhe**

**Video cues:** 05:42 reveal the missing-step chain; 05:53 reveal the reordered chain.

Frame scores alone do not describe complete action chains. YOLO matches only two of the ten complete reference chains, ordinary GRU matches six, and both our final pipeline and VideoMAE match all ten. The examples show that a correct prediction can describe an incorrect procedure. Our model preserves a missing step and a reordered sequence instead of repairing them into the standard order. Only our pipeline uses adaptive confirmation on the dense five-frame-per-second stream. These results therefore compare complete methods, including their decoding, rather than an isolated change to the neural network.

*Presenter cue: Point to the four bars, then distinguish prediction correctness from procedure correctness.*

Source: evaluation/action_chains.csv and final project report, Tables 4 and 5. Proposed examples: test_error_12356 and test_error_645321.

## Slide 10: The local deployment choice

**Time: 06:10-06:55 (45 seconds)**

**Speaker: Xu Wenzhe**

**Video cues:** 06:20 reveal incremental input; 06:35 reveal the VideoMAE comparison.

We retain our model for the local camera application because deployment also depends on size and temporal input requirements. It has about ten point six four million parameters, compared with eighty-six point two three million for VideoMAE. Its recurrent path consumes a frame and an existing state. VideoMAE uses a sixteen-frame window, requiring about three seconds of initial context at our sampling rate. Centered predictions also use future frames. VideoMAE can run locally and stream delayed predictions, but its workflow differs from immediate incremental feedback. The parameter comparison supports our choice without claiming a measured speed or energy advantage.

*Presenter cue: Present VideoMAE as the accuracy reference. Do not claim it is impossible to deploy or that parameter ratio equals speed ratio.*

Source: provenance/deployment/architecture_timing.json and final project report, Table 6. Exact parameters: 10,642,639 and 86,232,583. Initial VideoMAE window approximately 3 seconds, centered future context approximately 1.4 seconds.

## Slide 11: Conclusions and next evaluation

**Time: 06:55-08:00 (65 seconds)**

**Speaker: Xu Wenzhe**

**Video cues:** 07:03 reveal the delivered system summary; 07:29 reveal the next-evaluation items.

The project delivers a working camera application, annotated data, four comparison models, and reproducible evaluation records. Our selected pipeline combines visual recognition with learned boundaries and adaptive action confirmation. It achieves all ten reference chains in the reviewed evaluation and supports local feedback on both order and recognized duration. The scope of that evidence matters. We have inspected these recordings repeatedly, including correcting labels after reviewing predictions. The results therefore describe internal validation in the recorded scene. The next evaluation should freeze labels and settings, then use new people, camera positions, and lighting. It should also measure sustained throughput on the same hardware. Our contribution is a complete, inspectable assessment system with a clear deployment rationale. Thank you.

*Presenter cue: Finish at PPT video time 08:00 (assembled presentation time 13:00). Keep the final slide visible as the recording ends.*

Source: Final project report, Section 7. docs/FINAL_RELEASE.md and provenance/README.md.
