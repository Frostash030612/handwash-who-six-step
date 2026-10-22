# Self-Recorded Dataset

The local final technical ZIP contains 32 original self-recorded videos and 6,987
sampled frames with frame-level annotation CSVs. The development partition has
14 training clips and eight validation clips. Original videos 4 and 8 plus eight
additional continuous videos form the ten-video reported test set. The final temporal
heads were refit on all 22 development clips after epoch/decoder selection.

Model classes are `other` (0) and handwashing steps 1–6. Annotated transitions with
`valid_for_training=false` are ignored, not converted to background. Raw exports use
slightly different timestamp column names; use their actual units and validity flags.

Original video 8 has the reviewed chain 1 -> 2 -> 4 -> 5 -> 3 -> 6. Step 4 occupies
21.8–28.6 s, the 28.6–29.0 s transition is ignored, and step 5 starts at 29.0 s.
Its correction was made after predictions had been viewed, so the evaluation is
retrospective internal evidence. The local ZIP includes the full manifest, split table,
raw videos and annotations. They are not published in this source repository.
