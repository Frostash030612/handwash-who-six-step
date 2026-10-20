#!/usr/bin/env python3
"""Validation-only experiment: increase class-6 loss weight for the self-scene GRU.

Train clips and validation clips follow self_scene_boundary_v2.yaml. Regression
clips are opened only after the class-weight factor has been selected and saved.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from sklearn.metrics import classification_report
from torch.nn import functional as F

try:
    from _bootstrap import PROJECT_ROOT
except ImportError:
    from ._bootstrap import PROJECT_ROOT
from scripts import train_self_scene_boundary_v2 as base
from handwash.core.seeding import seed_everything


def score(head, clips, ids, cfg, device):
    preds = {int(i): base.predict(head, clips[int(i)], device) for i in ids}
    records = [base.evaluate_clip(clips[int(i)], *preds[int(i)], {"method": "raw"}, cfg,
                                  labels=clips[int(i)]["labels"], detail=False) for i in ids]
    summary = base.summarize(records, cfg)
    y = np.concatenate([clips[int(i)]["labels"][clips[int(i)]["labels"] >= 0] for i in ids])
    p = np.concatenate([preds[int(i)][0].argmax(1)[clips[int(i)]["labels"] >= 0] for i in ids])
    report = classification_report(y, p, labels=list(range(7)), output_dict=True, zero_division=0)
    summary["class6_recall"] = report["6"]["recall"]
    summary["class6_precision"] = report["6"]["precision"]
    summary["class6_f1"] = report["6"]["f1-score"]
    return summary, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = base.yaml.safe_load(args.config.read_text())
    clips = base.load_clips(args.scene, cfg, args.cache)
    fit = [int(x) for x in cfg["splits"]["fit"]]
    val = [int(x) for x in cfg["splits"]["validation"]]
    regression = [int(x) for x in cfg["splits"]["regression"]]
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device
    result = {"protocol": "fixed 14 fit / 8 validation / 2 regression; regression opened after selection",
              "factors": [1.0, 1.5, 2.0, 3.0], "runs": [], "regression": None}
    per_factor = {factor: [] for factor in result["factors"]}
    settings = cfg["training"]

    for factor in result["factors"]:
        for seed in settings["seeds"]:
            seed = int(seed)
            seed_everything(seed, deterministic=True)
            torch.use_deterministic_algorithms(True, warn_only=True)
            head = base.ExpTemporalHead(feature_dim=clips[fit[0]]["features"].shape[1],
                                        hidden=int(settings["hidden"]), num_classes=7,
                                        dropout=float(settings["dropout"]))
            fit_features = np.concatenate([clips[i]["features"] for i in fit])
            head.feature_mean.copy_(torch.from_numpy(fit_features.mean(0)))
            head.feature_std.copy_(torch.from_numpy(fit_features.std(0).clip(min=1e-4)))
            head = head.to(device)
            items = [{"x": torch.from_numpy(clips[i]["features"]).unsqueeze(0).to(device),
                      "y": torch.from_numpy(clips[i]["labels"]).unsqueeze(0).to(device)} for i in fit]
            class_weights = torch.ones(7, device=device)
            class_weights[6] = float(factor)
            optimizer = torch.optim.AdamW(head.parameters(), lr=float(settings["learning_rate"]),
                                          weight_decay=float(settings["weight_decay"]))
            rng = random.Random(seed)
            order = list(range(len(items)))
            best, stale, started = None, 0, time.perf_counter()
            run_name = f"class6_weight_{factor:g}_seed{seed}"
            for epoch in range(1, int(settings["max_epochs"]) + 1):
                rng.shuffle(order)
                head.train()
                for ix in order:
                    item = items[ix]
                    optimizer.zero_grad(set_to_none=True)
                    logits = head(item["x"])[0]
                    loss = F.cross_entropy(logits.reshape(-1, 7), item["y"].reshape(-1),
                                           weight=class_weights, ignore_index=base.IGNORE)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), float(settings["grad_clip_norm"]))
                    optimizer.step()
                metrics, _ = score(head, clips, val, cfg, device)
                value = float(metrics["composite"])
                print(json.dumps({"run": run_name, "epoch": epoch, "val_composite": value,
                                  "val_accuracy": metrics["accuracy"], "val_macro_f1": metrics["macro_f1"],
                                  "val_class6_recall": metrics["class6_recall"]}), flush=True)
                if best is None or value > best["composite"] + 1e-9:
                    best = {"epoch": epoch, "composite": value, "metrics": metrics,
                            "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
                    stale = 0
                else:
                    stale += 1
                    if stale >= int(settings["patience"]):
                        break
            head.load_state_dict(best["state"])
            run_dir = args.out / "runs" / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            payload = {"format": "self-scene-class6-weighting/1", "seed": seed,
                       "class6_weight": factor, "selected_epoch": best["epoch"],
                       "fit_clips": fit, "validation_clips": val,
                       "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                       "hparams": dict(head.hparams), "exp_sha256": clips[fit[0]]["yolo_sha256"],
                       "fps": float(cfg["features"]["sample_fps"]), "metrics": best["metrics"]}
            torch.save(payload, run_dir / "head.pt")
            run = {"factor": factor, "seed": seed, "epoch": best["epoch"],
                   "metrics": best["metrics"], "elapsed_s": time.perf_counter() - started}
            (run_dir / "selection.json").write_text(json.dumps(run, indent=2) + "\n")
            result["runs"].append(run)
            per_factor[factor].append(run)

    # Select the weight using validation summaries only, then persist before opening regression.
    means = {}
    for factor, runs in per_factor.items():
        means[str(factor)] = {key: float(np.mean([r["metrics"][key] for r in runs]))
                              for key in ("composite", "accuracy", "macro_f1", "class6_recall", "class6_f1",
                                          "boundary_f1", "continuous_chain_match_rate")}
    chosen = max((f for f in result["factors"] if f > 1.0),
                 key=lambda f: (means[str(f)]["composite"], means[str(f)]["class6_recall"], -f))
    result["validation_means"] = means
    result["selected_class6_weight"] = chosen
    (args.out / "validation_selection.json").write_text(json.dumps(result, indent=2) + "\n")

    # The selected weight was fixed above. Compare it and the matched unweighted control on regression.
    reg = {}
    for factor in (1.0, chosen):
        entries = []
        for seed in settings["seeds"]:
            run_dir = args.out / "runs" / f"class6_weight_{factor:g}_seed{int(seed)}"
            payload = torch.load(run_dir / "head.pt", map_location="cpu", weights_only=True)
            head = base.ExpTemporalHead(**payload["hparams"])
            head.load_state_dict(payload["state_dict"])
            head.to(device).eval()
            metric, preds = score(head, clips, regression, cfg, device)
            entries.append({"seed": int(seed), "metrics": metric})
            for clip_id in regression:
                np.savez_compressed(run_dir / f"video{clip_id:02d}.npz",
                                    labels=clips[clip_id]["labels"], times=clips[clip_id]["times"],
                                    raw_probs=preds[clip_id][0])
        reg[str(factor)] = entries
    result["regression"] = reg
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected_class6_weight": chosen, "validation_means": means,
                      "regression": {k: [x["metrics"] for x in v] for k, v in reg.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
