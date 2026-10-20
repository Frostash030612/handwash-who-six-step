#!/usr/bin/env python
"""Replay frozen v3 networks, verify v4 decisions, and package inference assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

try:
    from _bootstrap import PROJECT_ROOT
except ImportError:
    from ._bootstrap import PROJECT_ROOT
from handwash.core.segment_decoding import SegmentDecoderConfig, decode_action_segments
from handwash.pipelines.multiscale_boundary_gru import STRUCTURE_FORMAT, StructuralBoundaryGRU


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    selection = json.loads((args.result / "selection.json").read_text())
    summary = json.loads((args.result / "summary.json").read_text())
    setting = SegmentDecoderConfig(**selection["selected"]["with_boundary"])
    yolo_hash = sha(args.yolo)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    rows = []
    bundle = args.result / "inference_bundle"
    bundle.mkdir(exist_ok=True)
    model_records = []
    for checkpoint in sorted((args.source / "runs").glob("boundary_multiscale_seed*/head.pt")):
        original_hash = sha(checkpoint)
        if summary["checkpoints"][str(checkpoint)] != original_hash:
            raise RuntimeError("A source checkpoint changed after decoder evaluation")
        seed = int(checkpoint.parent.name.split("seed")[-1])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["format"] != STRUCTURE_FORMAT or payload["exp_sha256"] != yolo_hash:
            raise RuntimeError("Checkpoint format or YOLO dependency mismatch")
        model = StructuralBoundaryGRU(**payload["hparams"]).to(args.device).eval()
        model.load_state_dict(payload["state_dict"])
        files = sorted((args.result / "predictions" / f"boundary_duration_seed{seed}").glob("video*.npz"))
        for path in files:
            ident = int(path.stem.removeprefix("video"))
            cache = args.feature_cache / f"clip_{ident:02d}.npz"
            with np.load(cache, allow_pickle=False) as data:
                if str(data["yolo_sha256"]) != yolo_hash:
                    raise RuntimeError("Feature cache belongs to a different YOLO")
                x = torch.from_numpy(data["features"].astype(np.float32)).unsqueeze(0).to(args.device)
            with torch.inference_mode():
                logits, edge_logits = model(x)
                probs = logits[0].softmax(-1).cpu().numpy()
                edges = edge_logits[0].sigmoid().cpu().numpy()
            with np.load(path, allow_pickle=False) as result:
                probability_delta = float(np.max(np.abs(probs - result["raw_probs"])))
                boundary_delta = float(np.max(np.abs(edges - result["boundary_probs"])))
                if probability_delta > 2e-5 or boundary_delta > 2e-5:
                    raise RuntimeError(f"Network replay differs: {path}")
                fresh = decode_action_segments(probs, edges, setting)
                changed = int(np.count_nonzero(fresh.labels != result["decoded_labels"]))
                if changed:
                    raise RuntimeError(f"Replayed network changes {changed} decoded samples: {path}")
            rows.append(
                {
                    "seed": seed,
                    "clip": ident,
                    "max_probability_delta": probability_delta,
                    "max_boundary_delta": boundary_delta,
                    "decoded_label_difference": changed,
                    "feature_cache_sha256": sha(cache),
                }
            )
        if sha(checkpoint) != original_hash:
            raise RuntimeError("Network replay modified its source")
        target = bundle / f"head_seed{seed}.pt"
        shutil.copy2(checkpoint, target)
        model_records.append({"seed": seed, "file": target.name, "sha256": sha(target)})
    if len(model_records) != 3 or len(rows) != 30:
        raise RuntimeError("Expected three frozen networks on ten videos each")
    shutil.copy2(args.yolo, bundle / "yolo_best.pt")
    manifest = {
        "format": "self-scene-segment-inference/1",
        "name": "YOLO + frozen GRU + multiscale boundary + whole-segment decoder",
        "decoder": selection["selected"]["with_boundary"],
        "heads": model_records,
        "yolo": {"file": "yolo_best.pt", "sha256": yolo_hash, "input_size": 1088},
        "sample_fps": 5,
        "classes": ["others", "step1", "step2", "step3", "step4", "step5", "step6"],
        "action_chain": {"smoothing_window_seconds": 1, "min_action_seconds": 3},
        "offline": True,
        "independent_test_evaluated": False,
        "deployed": False,
        "note": "Three seeds are independent runs, not an ensemble. Decode raw outputs with this manifest, "
        "not the archived decoder embedded in the unchanged v3 checkpoint.",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verification = {
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if str(args.device).startswith("cuda") else None,
        "torch_version": str(torch.__version__),
        "yolo_sha256": yolo_hash,
        "networks_unchanged": True,
        "all_decoded_labels_identical": True,
        "runs": rows,
        "code_sha256": {
            str(p.relative_to(PROJECT_ROOT)): sha(p)
            for p in [
                PROJECT_ROOT / "src/handwash/core/segment_decoding.py",
                PROJECT_ROOT / "scripts/evaluate_self_scene_segment_v4.py",
                PROJECT_ROOT / "scripts/verify_self_scene_segment_v4.py",
            ]
        },
    }
    (args.result / "network_replay.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({k: v for k, v in verification.items() if k not in {"runs", "code_sha256"}}, indent=2))
    print("Verified 30 network/video pairs and packaged three seeds with frozen YOLO.")


if __name__ == "__main__":
    main()
