#!/usr/bin/env python
"""Sequential, controlled structural experiments; all selection is validation-only.

Stage 1: old-shape MLP vs causal multi-scale boundary heads on identical frozen GRUs.
Stage 2: equal-size residual refiners, with vs without boundary probabilities.
Stage 3: repeat the refiner pair with within-action consistency loss.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from torch.nn import functional as F

try:
    from _bootstrap import PROJECT_ROOT
except ImportError:
    from ._bootstrap import PROJECT_ROOT
from scripts import train_self_scene_boundary_v2 as base
from handwash.core.boundary_decoding import boundary_peaks
from handwash.core.boundary_metrics import BoundaryScore, PredictedBoundary, reference_boundaries, score_boundaries
from handwash.core.seeding import seed_everything
from handwash.core.temporal_selection import replacement_decision
from handwash.errors import DataError
from handwash.pipelines.boundary_gru import boundary_targets_from_consensus, weak_boundary_intervals_from_consensus, weak_interval_loss
from handwash.pipelines.multiscale_boundary_gru import STRUCTURE_FORMAT, StructuralBoundaryGRU, within_action_consistency

LOG = logging.getLogger("handwash.structure_v3")
REFINERS = ("refine_no_boundary", "refine_boundary", "refine_no_boundary_smooth", "refine_boundary_smooth")


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def setup_model(kind: str, seed: int, v2: Path, output: Path, cfg: dict, device: str):
    seed_everything(seed, deterministic=True)
    # The shared seed helper uses warn_only; this experiment requires strict mode.
    torch.use_deterministic_algorithms(True)
    backbone_path = v2 / "runs" / f"gru_seed{seed}" / "head.pt"
    backbone = torch.load(backbone_path, map_location="cpu", weights_only=True)
    refinement = kind in REFINERS
    model = StructuralBoundaryGRU(
        backbone_hparams=backbone["hparams"], boundary_kind="mlp" if kind == "boundary_mlp" else "multiscale",
        **cfg["architecture"], refinement=refinement,
        use_boundary=kind in {"refine_boundary", "refine_boundary_smooth"},
    )
    model.backbone.load_state_dict(backbone["state_dict"])
    dependencies = {str(backbone_path): base.file_sha256(backbone_path)}
    peak_setting = None
    if refinement:
        boundary_path = output / "runs" / f"boundary_multiscale_seed{seed}" / "head.pt"
        boundary = torch.load(boundary_path, map_location="cpu", weights_only=True)
        model.boundary.load_state_dict({k.removeprefix("boundary."):v for k,v in boundary["state_dict"].items()
                                       if k.startswith("boundary.")})
        dependencies[str(boundary_path)] = base.file_sha256(boundary_path)
        peak_setting = boundary["head_peak_setting"]
    return model.to(device), dependencies, backbone["exp_sha256"], peak_setting


@torch.no_grad()
def encode_clips(model, clips, ids, cfg, device):
    items = {}
    for ident in ids:
        clip = clips[ident]
        x = torch.from_numpy(clip["features"]).unsqueeze(0).to(device)
        hidden, logits = model.encode(x)
        y = torch.from_numpy(clip["labels"]).unsqueeze(0).to(device)
        targets, mask = boundary_targets_from_consensus(clip["labels"])
        intervals = weak_boundary_intervals_from_consensus(clip["labels"], max_gap=cfg["training"]["interval_max_gap"])
        items[ident] = dict(hidden=hidden, logits=logits, y=y, targets=torch.from_numpy(targets).to(device),
                            mask=torch.from_numpy(mask).to(device), intervals=[(0,a,b) for a,b in intervals])
    return items


def boundary_loss(edges, item, positive_weight, settings):
    mask = item["mask"]
    direct = (F.binary_cross_entropy_with_logits(edges[0][mask], item["targets"][mask], pos_weight=positive_weight)
              if bool(mask.any()) else edges.sum()*0.0)
    interval = weak_interval_loss(edges,item["intervals"])
    return settings["direct_boundary_weight"]*direct + settings["interval_boundary_weight"]*interval


@torch.no_grad()
def get_predictions(model,items,ids,*,ablate_boundary=False):
    model.eval()
    predictions = {}
    for ident in ids:
        logits, edges = model.forward_encoded(items[ident]["hidden"], items[ident]["logits"],
                                              ablate_boundary=ablate_boundary)
        predictions[ident] = (logits[0].softmax(-1).cpu().numpy(),edges[0].sigmoid().cpu().numpy())
    return predictions


def head_boundary_score(clips, ids, predictions, setting, protocol):
    pooled = BoundaryScore()
    for ident in ids:
        clip = clips[ident]
        edges = predictions[ident][1]
        peaks = boundary_peaks(edges,threshold=setting["threshold"],min_gap=setting["min_gap"])
        proposed = [PredictedBoundary(float((clip["times"][i-1]+clip["times"][i])/2),-1,-1,i) for i in peaks]
        score,_ = score_boundaries(reference_boundaries(clip["labels"],clip["times"]),proposed,
                                    tolerance_s=protocol["evaluation"]["boundary_tolerance_s"])
        pooled.add(score)
    return pooled.as_dict()


def metrics(clips,ids,predictions,decoder,protocol,detail=False):
    records = [base.evaluate_clip(clips[i],*predictions[i],decoder,protocol,labels=clips[i]["labels"],detail=detail)
               for i in ids]
    return base.summarize(records,protocol),records


def train_run(kind,seed,clips,items,v2,output,cfg,protocol,device,protocol_hash):
    run_dir = output/"runs"/f"{kind}_seed{seed}"
    if (run_dir/"selection.json").exists():
        old = json.loads((run_dir/"selection.json").read_text())
        if old["protocol_sha256"] != protocol_hash:
            raise DataError(f"{run_dir}: existing protocol differs; use a new output directory")
        LOG.info("keep %s",run_dir.name)
        return
    model,dependencies,yolo_sha,peak_setting = setup_model(kind,seed,v2,output,cfg,device)
    fit, val = protocol["splits"]["fit"],protocol["splits"]["validation"]
    settings = cfg["training"]
    positive = sum(int(items[i]["targets"].sum()) for i in fit)
    negative = sum(int(items[i]["mask"].sum()-items[i]["targets"].sum()) for i in fit)
    pw = min(settings["positive_weight_cap"],max(1.,math.sqrt(negative/max(positive,1))))
    weight = torch.tensor(pw,device=device)
    refinement = kind in REFINERS
    coefficient = settings["consistency_weight"] if kind.endswith("_smooth") else 0.
    trainable = [p for p in model.parameters() if p.requires_grad]
    init_hash = state_hash((model.refiner if refinement else model.boundary).state_dict())
    frozen_hash = state_hash(model.backbone.state_dict())
    frozen_boundary_hash = state_hash(model.boundary.state_dict()) if refinement else None
    optimizer = torch.optim.AdamW(trainable,lr=settings["learning_rate"],weight_decay=settings["weight_decay"])
    rng = random.Random(seed)
    order = list(fit)
    history, best, stale = [],None,0
    started = time.perf_counter()
    peak_grid = [dict(threshold=float(t),min_gap=int(g)) for t in cfg["boundary_peak_grid"]["thresholds"]
                 for g in cfg["boundary_peak_grid"]["min_gaps"]]
    # Refiner epoch 0 is the exact baseline, a valid no-improvement fallback.
    for epoch in range(0 if refinement else 1,settings["max_epochs"]+1):
        losses = []
        if epoch > 0:
            model.train()
            rng.shuffle(order)
            for ident in order:
                item = items[ident]
                optimizer.zero_grad(set_to_none=True)
                logits, edges = model.forward_encoded(item["hidden"],item["logits"])
                if refinement:
                    loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),item["y"].reshape(-1),ignore_index=base.IGNORE)
                    loss = loss + coefficient*within_action_consistency(logits,item["y"],truncate=settings["consistency_truncate"])
                else:
                    loss = boundary_loss(edges,item,weight,settings)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable,settings["grad_clip_norm"])
                optimizer.step()
                losses.append(float(loss.detach()))
        predictions = get_predictions(model,items,val)
        validation,_ = metrics(clips,val,predictions,{"method":"raw"},protocol)
        if refinement:
            key = (validation["composite"],)
            direct = head_boundary_score(clips,val,predictions,peak_setting,protocol)
            val_loss = None
        else:
            candidates = [(head_boundary_score(clips,val,predictions,p,protocol),p) for p in peak_grid]
            direct,peak_setting = max(candidates,key=lambda x:(x[0]["f1"],x[0]["recall"]))
            with torch.no_grad():
                val_loss = float(np.mean([float(boundary_loss(model.boundary(items[i]["hidden"]),items[i],weight,settings)) for i in val]))
            key = (direct["f1"],-val_loss)
        eligible_improvement = best is None or key > best["key"]
        row = dict(epoch=epoch,train_loss=float(np.mean(losses)) if losses else None,
                   validation=validation,validation_boundary_loss=val_loss,
                   direct_head=direct,head_peak_setting=peak_setting,improved=eligible_improvement)
        history.append(row)
        LOG.info("%s epoch=%d cls=%.4f head=%.4f chain=%s",run_dir.name,epoch,validation["accuracy"],direct["f1"],validation["continuous_chain_matches"])
        if eligible_improvement:
            best=dict(key=key,epoch=epoch,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                      validation=validation,direct_head=direct,head_peak_setting=copy.deepcopy(peak_setting))
            stale=0
        else:
            stale+=1
        if stale >= settings["patience"]:
            break
    model.load_state_dict(best["state"])
    assert frozen_hash == state_hash(model.backbone.state_dict())
    if refinement:
        assert frozen_boundary_hash == state_hash(model.boundary.state_dict())
    predictions = get_predictions(model,items,val)
    raw_metrics,_ = metrics(clips,val,predictions,{"method":"raw"},protocol)
    assert abs(raw_metrics["composite"]-best["validation"]["composite"]) < 1e-8
    # Boundary heads are selected by direct boundary F1, then the legacy blend
    # grid is chosen on validation only. Refiners always output their own logits.
    decoder = {"method":"raw"}
    if not refinement:
        options = [(d,metrics(clips,val,predictions,d,protocol)[0]) for d in base.decoder_grid(protocol)]
        decoder,_ = max(options,key=base._selection_key)
    selected_metrics,_ = metrics(clips,val,predictions,decoder,protocol)
    selection=dict(kind=kind,seed=seed,selected_epoch=best["epoch"],epochs_run=len(history),
                   protocol_sha256=protocol_hash,head_peak_setting=best["head_peak_setting"],
                   direct_head=best["direct_head"],decoder=decoder,validation=selected_metrics,
                   dependencies=dependencies,yolo_sha256=yolo_sha,initial_trainable_state_sha256=init_hash,
                   frozen_backbone_state_sha256=frozen_hash,frozen_boundary_state_sha256=frozen_boundary_hash,
                   fit_ids=fit,validation_ids=val,regression_used_for_selection=False,
                   trainable_parameters=sum(p.numel() for p in trainable),
                   total_parameters=sum(p.numel() for p in model.parameters()),consistency_weight=coefficient,
                   boundary_positive_weight=pw,elapsed_s=time.perf_counter()-started)
    run_dir.mkdir(parents=True,exist_ok=True)
    torch.save(dict(format=STRUCTURE_FORMAT,hparams=model.hparams,state_dict=best["state"],
                    decoder=decoder,head_peak_setting=best["head_peak_setting"],
                    exp_sha256=yolo_sha,action_chain=protocol["action_chain"],metrics=selection),run_dir/"head.pt")
    base._write_json(run_dir/"selection.json",selection)
    base._write_json(run_dir/"history.json",history)
    LOG.info("SAVED %s epoch=%d",run_dir.name,best["epoch"])


def load_trained(path,device):
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload["format"] != STRUCTURE_FORMAT:
        raise DataError(f"{path}: expected {STRUCTURE_FORMAT}")
    model=StructuralBoundaryGRU(**payload["hparams"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(),payload


def evaluate_all(scene,v2,output,cfg,protocol,device):
    # Regression data is opened only after all model/decoder selections finish.
    clips=base.load_clips(scene,protocol,v2)
    per_run,details=[],[]
    report=output/"report"
    for seed in cfg["seeds"]:
        baseline,_=base._load_head(v2/"runs"/f"gru_seed{seed}"/"head.pt",device)
        encoded_model,_,_,_=setup_model("boundary_mlp",seed,v2,output,cfg,device)
        items=encode_clips(encoded_model,clips,list(clips),cfg,device)
        for kind in ["gru",*cfg["stages"]]:
            if kind=="gru":
                model,payload=None,{"decoder":{"method":"raw"},"metrics":{"selected_epoch":None}}
            else:
                model,payload=load_trained(output/"runs"/f"{kind}_seed{seed}"/"head.pt",device)
            tag=f"{kind}_seed{seed}"
            for set_name in ("validation","regression"):
                ids=protocol["splits"][set_name]
                if model is None:
                    predictions={i:base.predict(baseline,clips[i],device) for i in ids}
                else:
                    predictions=get_predictions(model,items,ids)
                modes=[("saved",predictions,payload["decoder"])]
                if kind in {"refine_boundary","refine_boundary_smooth"}:
                    modes.append(("zero_boundary",get_predictions(model,items,ids,ablate_boundary=True),payload["decoder"]))
                if kind in {"boundary_mlp","boundary_multiscale"}:
                    modes.append(("zero_boundary",{i:(p,np.zeros_like(e)) for i,(p,e) in predictions.items()},payload["decoder"]))
                for mode,preds,decoder in modes:
                    summary,records=metrics(clips,ids,preds,decoder,protocol,detail=True)
                    direct = head_boundary_score(clips,ids,predictions,payload["head_peak_setting"],protocol) if model else None
                    per_run.append(dict(kind=kind,seed=seed,set=set_name,mode=mode,selected_epoch=payload["metrics"]["selected_epoch"],
                                        head_boundary=direct,decoder=decoder,**summary))
                    for ident,record in zip(ids,records):
                        arrays=record["arrays"]
                        filtered={k:v for k,v in record.items() if k not in {"kind","y","pred","raw_pred","arrays","boundary","smoothed_boundary"}}
                        details.append(dict(kind=kind,seed=seed,set=set_name,mode=mode,**filtered,
                                            clip_kind=record["kind"],
                                            boundary=record["boundary"].as_dict(),
                                            smoothed_boundary=record["smoothed_boundary"].as_dict()))
                        if mode=="saved":
                            dest=report/"predictions"/tag/f"video{ident:02d}.csv"
                            base._write_predictions(dest,clips[ident],record)
                            backbone_probs=items[ident]["logits"][0].softmax(-1).cpu().numpy()
                            np.savez_compressed(dest.with_suffix(".npz"),labels=clips[ident]["labels"],times=clips[ident]["times"],
                                                backbone_probs=backbone_probs,probs=arrays["probs"],decoded=arrays["decoded"],
                                                edges=arrays["edges"] if arrays["edges"] is not None else np.empty(0))
                            if kind=="gru":
                                old=list(csv.DictReader((v2/"report/predictions"/tag/f"video{ident:02d}.csv").open()))
                                assert len(old)==len(arrays["decoded"])
                                assert all(int(r["decoded_label"])==int(p) for r,p in zip(old,arrays["decoded"].argmax(1)))
    aggregate={}
    for kind in ["gru",*cfg["stages"]]:
        aggregate[kind]={}
        for set_name in ("validation","regression"):
            rows=[r for r in per_run if r["kind"]==kind and r["set"]==set_name and r["mode"]=="saved"]
            aggregate[kind][set_name]=base._mean_std(rows)
            aggregate[kind][set_name]["head_boundary_f1"]=float(np.mean([r["head_boundary"]["f1"] for r in rows])) if kind!="gru" else None
    chosen=max(REFINERS,key=lambda k:aggregate[k]["validation"]["composite"])
    decisions={k:{split:replacement_decision(aggregate[k][split],aggregate["gru"][split])
                  for split in ("validation","regression")} for k in REFINERS}
    paired_init={}
    for seed in cfg["seeds"]:
        selections={k:json.loads((output/"runs"/f"{k}_seed{seed}"/"selection.json").read_text()) for k in REFINERS}
        hashes={x["initial_trainable_state_sha256"] for x in selections.values()}
        assert len(hashes)==1
        paired_init[str(seed)]=dict(initial_refiner_state_sha256=next(iter(hashes)),all_four_refiners_identical=True)
    result=dict(format="self-scene-structure-v3-report/1",per_run=per_run,mean=aggregate,
                selected_by_validation=chosen,replacement_decisions=decisions,
                provisional_internal_replacement_pass=all(d["recommend_replacement"] for d in decisions[chosen].values()),
                new_independent_test_evaluated=False,paired_initialization=paired_init,
                notes=["Validation selects heads, decoders and checkpoints; regression videos 4/8 are repeatedly inspected, not independent tests.",
                       "Frozen GRU remains identical across matched seeds; new modules alone are trained.",
                       "Direct head boundary F1 is distinct from decoded label-change boundary F1.",
                       "No deployment replacement is performed by this script."])
    base._write_json(report/"summary.json",result)
    base._write_json(report/"per_video.json",details)
    render_report(result,details,output)
    return result


def render_report(summary,details,output):
    names={"gru":"普通 GRU（冻结基线）","boundary_mlp":"原形状边界头（匹配对照）", "boundary_multiscale":"多尺度边界头",
           "refine_no_boundary":"细化，无边界", "refine_boundary":"细化，有边界",
           "refine_no_boundary_smooth":"细化，无边界＋一致性", "refine_boundary_smooth":"细化，有边界＋一致性"}
    lines=["# 边界感知 GRU 结构优化 v3", "", "## 实验协议", "",
           "沿用自采 14 段训练、8 段验证，5 fps、冻结的 1088 YOLO 特征，三个种子。每个种子使用对应 v2 普通 GRU 最佳检查点并冻结，仅新增模块随机初始化训练。不会把验证集并入训练。原视频 4、8 在所有训练与选择完成后才作回归检查；新采集的 10 段尚未使用。", "",
           "第一阶段对比 MLP 和三层因果卷积边界头（卷积核 3、膨胀率 1/2/4、64 通道，15 个采样点感受野）。该阶段按验证集直接边界头 F1 选权重，再按原验证综合分数选解码。第二阶段比较参数量完全相同、初始权重完全相同的分类细化模块（输入边界概率或零值），边界头与 GRU 均冻结；第三阶段只增加 0.1 倍同动作内部一致性损失。细化模型不再做分段平均，按原综合分数选实际检查点；零初始化残差、epoch 0 为可选的原分类退路。", "",
           "动作链仍为 1 秒居中多数平滑＋连续 3 秒确认，整体系统属于离线评估；因果卷积不代表整套后处理零延迟。", "",
           "## 三个种子均值", "",
           "| 模型 | 数据 | 帧准确率 | 宏 F1 | 动作链匹配率 | 解码边界 F1 | 平滑后边界 F1 | 边界头自身 F1 |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for kind,by_set in summary["mean"].items():
        for split,row in by_set.items():
            vals=[row[k] for k in ("accuracy","macro_f1","continuous_chain_match_rate","boundary_f1","boundary_f1_after_chain_smoothing","head_boundary_f1")]
            lines.append(f"| {names[kind]} | {split} | "+" | ".join("—" if v is None else f"{100*v:.2f}%" for v in vals)+" |")
    selected=summary["selected_by_validation"]
    lines += ["",f"验证集综合分数选择的细化候选：**{names[selected]}**。新独立测试尚未进行，不据内部结果直接部署替换。", "", "## 同一检查点关闭边界输入", "",
              "| 模型 | 数据 | 正常准确率 | 关闭边界准确率 | 正常平滑边界 F1 | 关闭边界平滑 F1 |", "|---|---|---:|---:|---:|---:|"]
    for kind in ("refine_boundary","refine_boundary_smooth"):
        for split in ("validation","regression"):
            r=[x for x in summary["per_run"] if x["kind"]==kind and x["set"]==split]
            values=[np.mean([x[key] for x in r if x["mode"]==mode]) for key in ("accuracy","boundary_f1_after_chain_smoothing") for mode in ("saved","zero_boundary")]
            lines.append(f"| {names[kind]} | {split} | "+" | ".join(f"{v*100:.2f}%" for v in values)+" |")
    lines += ["", "## 逐视频确认动作链", "", "| 模型 | 种子 | 视频 | 真实 | 预测 | 匹配 |", "|---|---|---:|---|---|---|"]
    for r in details:
        if r["mode"]=="saved":
            lines.append(f"| {names[r['kind']]} | {r['seed']} | {r['clip']} | "+"→".join(map(str,r["truth_order"]))+" | "+"→".join(map(str,r["chain_order"]))+f" | {r['chain_match']} |")
    lines += ["", "## 可复现材料", "", "config.yaml、protocol_v2.yaml、integrity.json 记录配置和输入指纹；runs/ 保存 18 个实际检查点与训练历史；report/summary.json 保存逐种子指标与预定替换门槛；report/per_video.json 保存错误时间段和置信度；predictions/ 保存逐帧 CSV 及概率 NPZ。", ""]
    (output/"report/report.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root",type=Path,required=True)
    parser.add_argument("--v2-output",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs/research/self_scene_structure_v3.yaml")
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    cfg=yaml.safe_load(args.config.read_text())
    protocol=yaml.safe_load((PROJECT_ROOT/cfg["base_protocol"]).read_text())
    protocol_hash=hashlib.sha256(json.dumps({"cfg":cfg,"base":protocol},sort_keys=True).encode()).hexdigest()
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    if (output/"config.yaml").exists() and yaml.safe_load((output/"config.yaml").read_text()) != cfg:
        raise DataError(f"{output}: configuration differs from existing run")
    (output/"config.yaml").write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    (output/"protocol_v2.yaml").write_text(yaml.safe_dump(protocol,allow_unicode=True,sort_keys=False))
    protected=[args.scene_root/protocol["features"]["yolo_checkpoint"],
               *[args.v2_output/"runs"/f"gru_seed{s}"/"head.pt" for s in cfg["seeds"]]]
    before={str(p):base.file_sha256(p) for p in protected}
    source_files=[Path(__file__),PROJECT_ROOT/"src/handwash/pipelines/multiscale_boundary_gru.py",
                  PROJECT_ROOT/"scripts/train_self_scene_boundary_v2.py"]
    source_hashes={str(p):base.file_sha256(p) for p in source_files}
    train_protocol=copy.deepcopy(protocol)
    train_protocol["splits"]["regression"]=[]
    train_protocol["splits"]["reviewed_labels"]={}
    clips=base.load_clips(args.scene_root,train_protocol,args.v2_output)
    assert set(clips)==set(protocol["splits"]["fit"]+protocol["splits"]["validation"])
    encoded={}
    for seed in cfg["seeds"]:
        model,_,sha,_=setup_model("boundary_mlp",seed,args.v2_output,output,cfg,args.device)
        assert all(c["yolo_sha256"]==sha for c in clips.values())
        encoded[seed]=encode_clips(model,clips,list(clips),cfg,args.device)
    for kind in cfg["stages"]:
        LOG.info("STAGE_START %s",kind)
        for seed in cfg["seeds"]:
            train_run(kind,seed,clips,encoded[seed],args.v2_output,output,cfg,protocol,args.device,protocol_hash)
        LOG.info("STAGE_COMPLETE %s",kind)
    del encoded
    result=evaluate_all(args.scene_root,args.v2_output,output,cfg,protocol,args.device)
    after={str(p):base.file_sha256(p) for p in protected}
    assert before==after
    base._write_json(output/"integrity.json",dict(protocol_sha256=protocol_hash,source_sha256=source_hashes,
                     preserved_checkpoints=before,all_preserved=True,new_test_used=False,regression_used_for_selection=False,
                     environment=dict(torch=torch.__version__,device=args.device)))
    (output/"COMPLETE").write_text("All 18 new modules trained, selected, and evaluated; no deployment changed.\n")
    LOG.info("STRUCTURE_V3_COMPLETE selected=%s",result["selected_by_validation"])


if __name__=="__main__":
    main()
