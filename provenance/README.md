# Training and Deployment Provenance

The original frozen version is tag `self-scene-final-20261022` at main commit
`87f02a69bc26075730a798119a554f2ca40e6201`. This English source release is
a derived distribution. The local technical ZIP contains the complete weights,
raw videos, labels, predictions, English documentation and the source-change inventory.

YOLO and VideoMAE use a 14-training/8-validation development partition. The retained
GRU script selects epochs/decoding on that partition, then initializes fresh temporal
heads and refits them on all 22 development clips with recomputed feature normalization.
The selected epoch counts are five for GRU and three for boundary GRU. The ten reported
test clips are excluded from the final-fit list. This is not an equal-training-quantity
architecture-only ablation.

`training_source/` contains original cloud training scripts unchanged, with historical
paths and comments. They document model origin and are not application entry points.
The VideoMAE training source is at `../scripts/train_videomae_self_scene.py`.
`deployment/` contains historical synthetic architecture timing, incremental-equivalence
and duration records. Timing uses different inference units, so it is not a validated
sustained-throughput, power or isolated memory comparison.

No new model training or inference was performed for this English source release.
