"""Single selected self-scene model, streaming inference and cached final report."""

import gc
import hashlib
import io
import json
import queue
import struct
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
import torch
from PIL import Image

from handwash.core.action_chain import adaptive_boundary_gru_chain
from handwash.logging import get_logger
from handwash.pipelines.boundary_gru import BoundaryAwareGRU, decode_boundaries
from handwash.pipelines.exp_demo import ExpFeatureExtractor, ExpTemporalHead
from handwash.self_scene_assessment import MINIMUM_WASH_SECONDS, assess, collapse_order, make_segments

log = get_logger(__name__)
ROOT = Path("outputs/self_scene_camera")
STATIC = Path(__file__).resolve().parent / "static"

torch.set_num_threads(4)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAMES = {
    "yolo": "YOLO",
    "yolo_gru": "YOLO + GRU",
    "boundary_gru": "YOLO + 自适应边界感知 GRU",
    "videomae": "VideoMAE",
}
LOCK = threading.RLock()
WORK = queue.Queue()
STATE = {"phase": "ready", "message": "选择模型并开启摄像头后开始。", "device": DEVICE, "frames": 0}
SESSION = None


def synchronize():
    if DEVICE == "mps":
        torch.mps.synchronize()


def vm_ready():
    folder = ROOT / "models/videomae"
    try:
        # Both configuration files must be readable; a partially downloaded weight is unavailable.
        for name in ("config.json", "preprocessor_config.json"):
            json.loads((folder / name).read_text())
        with (folder / "model.safetensors").open("rb") as handle:
            n = struct.unpack("<Q", handle.read(8))[0]
            if n > 10_000_000:
                return False
            header = json.loads(handle.read(n))
        end = max(
            v["data_offsets"][1] for v in header.values() if isinstance(v, dict) and "data_offsets" in v
        )
        return (folder / "model.safetensors").stat().st_size == 8 + n + end
    except (OSError, ValueError, KeyError, struct.error):
        return False


def catalog():
    return [
        {
            "id": key,
            "name": name,
            "available": vm_ready()
            if key == "videomae"
            else (ROOT / "models/yolo_best.pt").is_file()
            and (key == "yolo" or (ROOT / "models" / f"{key}.pt").is_file()),
        }
        for key, name in MODEL_NAMES.items()
    ]


def prepared(raw):
    im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("无效图像")
    h, w = im.shape[:2]
    scale = 1080 / max(h, w)
    nw, nh = round(w * scale), round(h * scale)
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    top, left = (1080 - nh) // 2, (1080 - nw) // 2
    return cv2.copyMakeBorder(im, top, 1080 - nh - top, left, 1080 - nw - left, cv2.BORDER_REPLICATE)


class SelectedModel:
    """Only the selected model and its required backbone are loaded/executed."""

    def __init__(self, key):
        self.key = key
        self.ext = self.head = self.vm = None
        self.sha = {}
        modelroot = ROOT / "models"
        if key == "videomae":
            if not vm_ready():
                raise ValueError("VideoMAE 权重或预处理配置尚未完整就绪")
            from torchvision.transforms import Compose, Normalize, Resize, ToTensor
            from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

            self.vm = (
                VideoMAEForVideoClassification.from_pretrained(modelroot / "videomae", local_files_only=True)
                .eval()
                .to(DEVICE)
            )
            if self.vm.config.num_frames != 16 or self.vm.config.num_labels != 7:
                raise ValueError("VideoMAE 窗口或类别数量与自采模型不一致")
            expected = {0: "other", **{i: f"step{i}" for i in range(1, 7)}}
            if self.vm.config.id2label != expected:
                raise ValueError("VideoMAE 类别映射不一致")
            processor = VideoMAEImageProcessor.from_pretrained(modelroot / "videomae", local_files_only=True)
            # Same direct 224 resize and normalization as self-scene training/evaluation.
            self.transform = Compose(
                [Resize((224, 224)), ToTensor(), Normalize(processor.image_mean, processor.image_std)]
            )
            self.sha["videomae"] = hashlib.sha256(
                (modelroot / "videomae/model.safetensors").read_bytes()
            ).hexdigest()
            return
        self.sha["yolo"] = hashlib.sha256((modelroot / "yolo_best.pt").read_bytes()).hexdigest()
        self.ext = ExpFeatureExtractor(modelroot / "yolo_best.pt", device=DEVICE, cache_dir=ROOT / "cache")
        if key != "yolo":
            checkpoint = modelroot / f"{key}.pt"
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if ck["exp_sha256"] != self.sha["yolo"]:
                raise ValueError("YOLO 与时序模型权重不匹配")
            cls = BoundaryAwareGRU if key == "boundary_gru" else ExpTemporalHead
            self.head = cls(**ck["hparams"]).eval().to(DEVICE)
            self.head.load_state_dict(ck["state_dict"])
            self.sha[key] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if key == "boundary_gru":
                self.decoder = ck["metrics"]["decoder"]

    def frame(self, session, row):
        if self.key == "videomae":
            # Resample only received frames; no future images or tail padding are invented.
            with Image.open(row["path"]) as image:
                session["pixels"][row["index"]] = self.transform(image.convert("RGB"))
            rows = session["rows"][: row["index"] + 1]
            times = np.array([r["time_s"] for r in rows])
            latest = None
            while session["next_window"] + 3.0 <= row["time_s"] + 1e-6:
                start = session["next_window"]
                session["next_window"] += 0.8
                targets = start + np.arange(16) / 5
                indices = abs(times[:, None] - targets[None, :]).argmin(axis=0)
                if np.max(abs(times[indices] - targets)) > 0.3:
                    continue
                tensor = torch.stack([session["pixels"][int(i)] for i in indices])[None].to(DEVICE)
                with torch.inference_mode():
                    p = self.vm(pixel_values=tensor).logits[0].softmax(-1).cpu().numpy()
                synchronize()
                latest = {"time_s": float(start + 1.6), "probabilities": p}
                session["windows"].append(latest)
                if p.argmax() > 0:
                    session["live_wash_seconds"] += 0.8
            # Retain only pixels that can appear in the next window.
            keep_from = session["next_window"] - 0.4
            for index in list(session["pixels"]):
                if rows[index]["time_s"] < keep_from:
                    del session["pixels"][index]
            return latest
        with torch.inference_mode():
            features, probabilities = self.ext.forward_bgr([prepared(row["path"].read_bytes())])
            if self.key == "yolo":
                p = np.asarray(probabilities[0]).reshape(-1)
                edge = None
            else:
                features = torch.as_tensor(features, device=DEVICE).reshape(1, 1, -1)
                normalized = (features - self.head.feature_mean) / self.head.feature_std
                encoded, session["hidden"] = self.head.rnn(self.head.encoder(normalized), session["hidden"])
                p = self.head.classifier(encoded)[0, 0].softmax(-1).cpu().numpy()
                edge = None
                if self.key == "boundary_gru":
                    previous = session["previous_encoded"]
                    if previous is None:
                        previous = encoded
                    change = abs(encoded - previous)
                    edge = float(
                        self.head.boundary_classifier(torch.cat((encoded, change), dim=-1))
                        .sigmoid()
                        .reshape(-1)[0]
                        .item()
                    )
                    session["previous_encoded"] = encoded.detach()
            synchronize()
        last = session.get("last_prediction")
        if last is not None:
            gap = row["time_s"] - last["time_s"]
            if 0 < gap <= 0.6 and int(last["probabilities"].argmax()) > 0:
                session["live_wash_seconds"] += gap
        record = {"time_s": row["time_s"], "probabilities": p, "edge": edge}
        session["cache"][row["index"]] = record
        session["last_prediction"] = record
        return record


def finish_session(session, engine):
    began = time.perf_counter()
    rows = session["rows"]
    if not rows:
        raise ValueError("本次没有收到画面，请重新开始")
    end_s = session["end_s"]
    grid = np.arange(0, end_s, 0.2)
    if not len(grid):
        raise ValueError("录像时长不足")
    key = session["model"]
    confidence = np.zeros(len(grid))
    labels = np.full(len(grid), -1, dtype=int)
    statuses = ["unknown"] * len(grid)
    probabilities = np.zeros((len(grid), 7), dtype=np.float32)
    edges = np.zeros(len(grid), dtype=np.float32)
    if key == "videomae":
        windows = session["windows"]
        if windows:
            times = np.array([w["time_s"] for w in windows])
            centers = grid + np.minimum(0.2, end_s - grid) / 2
            idx = abs(times[:, None] - centers[None, :]).argmin(axis=0)
            valid = abs(times[idx] - centers) <= 0.400001
            probabilities = np.stack([windows[int(i)]["probabilities"] for i in idx])
        else:
            valid = np.zeros(len(grid), dtype=bool)
    else:
        if len(session["cache"]) != len(rows):
            raise ValueError("尚有帧未完成推理，不能生成完整结果")
        times = np.array([r["time_s"] for r in rows])
        idx = abs(times[:, None] - grid[None, :]).argmin(axis=0)
        valid = abs(times[idx] - grid) <= 0.3
        picked = [session["cache"][int(i)] for i in idx]
        probabilities = np.stack([r["probabilities"] for r in picked])
        if key == "boundary_gru":
            edges = np.array([r["edge"] for r in picked], dtype=np.float32)
            probabilities[~valid] = np.eye(7, dtype=np.float32)[0]
            edges[~valid] = 0
            raw = probabilities.copy()
            probabilities = decode_boundaries(
                probabilities,
                edges,
                threshold=float(engine.decoder["threshold"]),
                min_gap=int(engine.decoder["min_gap"]),
            )
            np.savez_compressed(
                session["dir"] / "boundary_raw.npz", times=grid, probabilities=raw, edges=edges
            )
    labels[valid] = probabilities.argmax(axis=1)[valid]
    confidence[valid] = probabilities.max(axis=1)[valid]
    if key == "boundary_gru":
        chain = adaptive_boundary_gru_chain(
            np.maximum(labels, 0),
            fps=5,
            smoothing_window_seconds=1,
            background_class=0,
            winner_confidences=confidence,
            outlier_z=-2,
            minimum_comparable_segments=4,
        )
        for part in chain["segments"]:
            a, b = part["start_index"], part["end_index_exclusive"]
            labels[a:b] = part["class"]
            statuses[a:b] = [part["status"]] * (b - a)
        labels[~valid] = -1
        confidence[valid] = probabilities[np.flatnonzero(valid), labels[valid]]
        for i in np.flatnonzero(~valid):
            statuses[int(i)] = "unknown"
        order = collapse_order([labels[i] for i in range(len(labels)) if statuses[i] == "confirmed"])
    else:
        statuses = ["unknown" if x < 0 else "background" if x == 0 else "predicted" for x in labels]
        order = collapse_order(labels)
    segments = make_segments(labels, grid, end_s, confidence=confidence, statuses=statuses)
    assessment = assess(segments, order, end_s, minimum_seconds=MINIMUM_WASH_SECONDS)
    result = {
        "schema_version": 2,
        "session": session["id"],
        "model": key,
        "model_name": MODEL_NAMES[key],
        "device": DEVICE,
        "recorded_seconds": end_s,
        "captured_frames": len(rows),
        "capture_fps": (len(rows) - 1) / max(0.01, rows[-1]["time_s"] - rows[0]["time_s"]),
        "weights_sha256": engine.sha,
        "assessment": assessment,
        "prediction": {"order": order, "segments": segments, "total_compute_s": session["compute_s"]},
        "ground_truth_available": False,
        "accuracy": None,
    }
    if key == "videomae":
        result["prediction"].update(windows=len(session["windows"]), temporal_lookahead_s=1.4)
    np.savez_compressed(
        session["dir"] / "predictions.npz",
        times=grid,
        probabilities=probabilities,
        labels=labels,
        valid=valid,
    )
    result["finalization_ms"] = (time.perf_counter() - began) * 1000
    (session["dir"] / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def worker():
    engine = None
    while True:
        kind, sid, row = WORK.get()
        try:
            with LOCK:
                session = SESSION
                if session is None or session["id"] != sid:
                    continue
                if STATE["phase"] == "error":
                    continue
            if kind == "prepare":
                if engine is None or engine.key != session["model"]:
                    engine = None
                    gc.collect()
                    if DEVICE == "mps":
                        torch.mps.empty_cache()
                    engine = SelectedModel(session["model"])
                with LOCK:
                    STATE.update(phase="recording", message="模型已就绪，正在采集。")
            elif kind == "frame":
                began = time.perf_counter()
                record = engine.frame(session, row)
                elapsed = time.perf_counter() - began
                session["compute_s"] += elapsed
                with LOCK:
                    STATE.update(processed_index=row["index"], live_wash_seconds=session["live_wash_seconds"])
                    if record is not None:
                        p = record["probabilities"]
                        STATE.update(
                            label=int(p.argmax()),
                            confidence=float(p.max()),
                            prediction_time_s=record["time_s"],
                            inference_ms=elapsed * 1000,
                            capture_to_result_ms=(time.time() - row["received"]) * 1000,
                            message="正在识别，洗手计时为实时估计。",
                        )
                    elif session["model"] == "videomae" and not session["windows"]:
                        STATE["message"] = "VideoMAE 正在收集首个 16 帧窗口（约 3 秒）。"
            elif kind == "finish":
                with LOCK:
                    STATE.update(phase="analyzing", message="正在汇总已缓存的预测…")
                result = finish_session(session, engine)
                with LOCK:
                    STATE.update(
                        phase="done",
                        message="识别完成，洗手时长和动作顺序见下方。",
                        result=result,
                        live_wash_seconds=result["assessment"]["wash_seconds"],
                    )
        except Exception as error:
            log.exception("摄像头推理失败")
            if kind == "prepare":
                engine = None
            with LOCK:
                STATE.update(phase="error", message=str(error))
        finally:
            WORK.task_done()


class Handler(BaseHTTPRequestHandler):
    def reply(self, obj, code=200, kind="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            return self.reply(
                (STATIC / "self_scene_camera.html").read_bytes(), kind="text/html; charset=utf-8"
            )
        if path == "/camera.js":
            return self.reply(
                (STATIC / "self_scene_camera.js").read_bytes(), kind="application/javascript; charset=utf-8"
            )
        if path == "/status":
            with LOCK:
                state = dict(STATE)
            state.update(api_version=2, available_models=catalog(), minimum_wash_seconds=MINIMUM_WASH_SECONDS)
            return self.reply(state)
        self.reply({"error": "not found"}, 404)

    def do_POST(self):
        global SESSION
        try:
            host = self.headers.get("Host", "")
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if host not in allowed or self.headers.get("Origin") not in (None, f"http://{host}"):
                return self.reply({"error": "仅接受本机页面请求"}, 403)
            path = urlsplit(self.path)
            query = parse_qs(path.query)
            n = int(self.headers.get("Content-Length", "0"))
            if n < 0 or n > 4 * 1024 * 1024:
                raise ValueError("请求过大")
            body = self.rfile.read(n)
            if len(body) != n:
                raise ValueError("数据接收不完整")
            if path.path == "/start":
                request = json.loads(body or b"{}")
                key = request.get("model")
                if key not in MODEL_NAMES:
                    raise ValueError("请先选择一个模型，再开始")
                if not next(m["available"] for m in catalog() if m["id"] == key):
                    raise ValueError("所选模型权重尚未完整就绪")
                with LOCK:
                    if STATE["phase"] not in ("ready", "done", "error") or WORK.unfinished_tasks:
                        raise ValueError("上一段识别还未结束")
                    sid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
                    folder = ROOT / "sessions" / sid
                    folder.mkdir(parents=True)
                    SESSION = {
                        "id": sid,
                        "dir": folder,
                        "model": key,
                        "rows": [],
                        "cache": {},
                        "hidden": None,
                        "previous_encoded": None,
                        "pixels": {},
                        "windows": [],
                        "next_window": 0.0,
                        "compute_s": 0.0,
                        "live_wash_seconds": 0.0,
                    }
                    (folder / "session.json").write_text(
                        json.dumps({"model": key, "minimum_wash_seconds": MINIMUM_WASH_SECONDS})
                    )
                    STATE.clear()
                    STATE.update(
                        phase="preparing",
                        session=sid,
                        model=key,
                        model_name=MODEL_NAMES[key],
                        device=DEVICE,
                        frames=0,
                        live_wash_seconds=0.0,
                        message="正在加载所选模型…",
                    )
                    WORK.put(("prepare", sid, None))
                return self.reply({"session": sid, "model": key, "phase": "preparing"})
            if path.path == "/frame":
                t, sid = float(query["t"][0]), query["session"][0]
                with Image.open(io.BytesIO(body)) as image:
                    if max(image.size) > 4096:
                        raise ValueError("图像尺寸过大")
                    image.verify()
                with LOCK:
                    if SESSION is None or SESSION["id"] != sid or STATE["phase"] != "recording":
                        raise ValueError("当前没有对应的录像会话")
                    if not np.isfinite(t) or not 0 <= t <= 125:
                        raise ValueError("超出测试时长")
                    if SESSION["rows"] and t <= SESSION["rows"][-1]["time_s"]:
                        raise ValueError("采样时间未递增")
                    index = len(SESSION["rows"])
                    dest = SESSION["dir"] / f"frame_{index:05d}.jpg"
                    dest.write_bytes(body)
                    row = {"index": index, "time_s": t, "path": dest, "received": time.time()}
                    SESSION["rows"].append(row)
                    STATE.update(frames=index + 1, elapsed_s=t)
                    with (SESSION["dir"] / "capture.jsonl").open("a") as handle:
                        handle.write(json.dumps({"index": index, "time_s": t, "image": dest.name}) + "\n")
                    WORK.put(("frame", sid, row))
                return self.reply({"saved": index})
            if path.path == "/stop":
                request = json.loads(body or b"{}")
                with LOCK:
                    if (
                        SESSION is None
                        or STATE["phase"] != "recording"
                        or request.get("session") != SESSION["id"]
                    ):
                        raise ValueError("当前没有对应的录像会话")
                    last = SESSION["rows"][-1]["time_s"] if SESSION["rows"] else 0
                    end_s = float(request.get("elapsed_s", last))
                    if not np.isfinite(end_s) or not last <= end_s <= 125:
                        raise ValueError("无效结束时间")
                    SESSION["end_s"] = end_s
                    STATE.update(phase="flushing", elapsed_s=end_s, message="正在处理最后几帧…")
                    WORK.put(("finish", SESSION["id"], None))
                return self.reply({"ok": True})
            self.reply({"error": "not found"}, 404)
        except (ValueError, KeyError, OSError, TypeError) as error:
            self.reply({"error": str(error)}, 400)

    def log_message(self, *args):
        pass


def serve_self_scene_camera(runtime_dir, *, port=8876, minimum_wash_seconds=40.0):
    """Serve one local camera session with weights supplied in runtime_dir/models.

    The 1088/224 input sizes and temporal windows are the trained checkpoint
    contract. This entry point does not retrain or change model decoding.
    """
    global ROOT, MINIMUM_WASH_SECONDS
    import math

    from handwash.errors import HandwashError

    if not math.isfinite(minimum_wash_seconds) or minimum_wash_seconds <= 0:
        raise HandwashError("assess.min_total_duration_s 必须为正数")
    ROOT = Path(runtime_dir).expanduser().resolve()
    MINIMUM_WASH_SECONDS = float(minimum_wash_seconds)
    ROOT.mkdir(parents=True, exist_ok=True)
    # Bind first: a port conflict must not leave an orphan inference worker.
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        threading.Thread(target=worker, daemon=True).start()
        log.info("摄像头页面：http://127.0.0.1:%s", server.server_port)
        server.serve_forever()
