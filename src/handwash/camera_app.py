"""本机摄像头网页与逐帧推理 HTTP 入口。"""

from __future__ import annotations

import io
import json
import math
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import BinaryIO
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image, UnidentifiedImageError

from handwash.core.config import AssessConfig, ResolvedConfig
from handwash.core.labels import CANONICAL_STEPS, STEP_EN, STEP_ORDER, Step
from handwash.core.protocol import check_order, check_repeats, collapse_repeats
from handwash.core.schema import ProtocolReport
from handwash.errors import DataError, HandwashError
from handwash.io.utils import write_jsonl
from handwash.logging import get_logger
from handwash.paths import checkpoint_path, ensure_dir, project_path
from handwash.pipelines.assess import save_report
from handwash.pipelines.common import resolve_device
from handwash.pipelines.evaluate import _load_model_from_checkpoint
from handwash.pipelines.exp_demo import DEFAULT_TEMPORAL_HEAD, ExpDemoClassifier
from handwash.pipelines.live import LiveClassifier, LiveSession

__all__ = ["CameraApp", "serve_camera"]

log = get_logger(__name__)
# 网页界面为英文：步骤名称取 core/labels.py 的 STEP_EN，页面加载时注入。
_LABELS_JSON = json.dumps(
    {step.value: {"name": STEP_EN[step], "step_no": step.order_index} for step in Step},
    ensure_ascii=False,
).encode("utf-8")
_MAX_JPEG_BYTES = 4 * 1024 * 1024
_MAX_IMAGE_SIDE = 4096
_MAX_VIDEO_BYTES = 128 * 1024 * 1024


def _ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        executable = shutil.which("ffmpeg")
        if executable:
            return executable
        raise DataError("The local video decoder is unavailable. Install the project's video dependencies.") from None


def _step_en(step: Step) -> str:
    return f"Step {step.order_index} ({STEP_EN[step]})" if step.order_index else STEP_EN[step]


def _english_violation_details(report: ProtocolReport, cfg: AssessConfig) -> list[str]:
    """为英文网页逐条生成问题说明，与 ``report.violations`` 一一对应。

    core/protocol.py 只生成中文说明（L1，不在此改动）。这里用同一批公开判定函数
    在同一份分段结果上重算逆序对与重复次数，顺序与 build_report 产出的违规条目一致；
    保存到磁盘的报告仍是中文。
    """
    actions = collapse_repeats(list(report.step_sequence))
    step_actions = collapse_repeats([step for step in actions if step in STEP_ORDER])
    inversions = iter(check_order(step_actions))
    repeats = dict(check_repeats(step_actions))
    durations = {stat.step: stat.duration_s for stat in report.statistics}
    fair_share = report.total_wash_duration_s / len(CANONICAL_STEPS)

    details: list[str] = []
    for violation in report.violations:
        step = violation.step
        if violation.kind == "missing" and step is not None:
            text = f"{_step_en(step)} was not detected"
        elif violation.kind == "out_of_order" and step is not None:
            earlier = next(inversions, None)
            text = (
                f"{_step_en(step)} came after {_step_en(earlier[0])}"
                if earlier is not None
                else f"{_step_en(step)} is out of order"
            )
        elif violation.kind == "insufficient_duration" and step is None:
            text = (
                f"Total rubbing time is {report.total_wash_duration_s:.1f} s, below the "
                f"WHO-recommended minimum of {cfg.min_total_duration_s:.0f} s"
            )
        elif violation.kind == "insufficient_duration":
            duration = durations.get(step, 0.0)
            if cfg.duration_check == "seconds":
                reason = f"only {duration:.1f} s, below the {cfg.min_step_duration_s:.1f} s minimum"
            else:
                threshold = fair_share * cfg.step_duration_ratio
                reason = (
                    f"only {duration:.1f} s, below {cfg.step_duration_ratio:.0%} of the average share "
                    f"of {fair_share:.1f} s (threshold {threshold:.1f} s)"
                )
            text = f"{_step_en(step)} is too short: {reason}"
        elif violation.kind == "repeated" and step is not None:
            text = f"{_step_en(step)} was performed {repeats.get(step, 2)} times"
        elif violation.kind == "missing_faucet_event" and step is not None:
            text = f"Faucet event not detected: {STEP_EN[step]}"
        else:
            text = violation.kind.replace("_", " ").capitalize() + (f": {_step_en(step)}" if step else "")
        details.append(text)
    return details


class CameraApp:
    """一个本机浏览器摄像头会话；模型只加载一次。"""

    def __init__(
        self,
        rc: ResolvedConfig,
        *,
        checkpoint: str | Path | None = None,
        demo_exp: bool = False,
        temporal_head: str | Path | bool = True,
    ) -> None:
        """``temporal_head``：True 表示根目录有 exp_temporal_head.pt 时自动加载，
        False 表示只用 exp.pt 逐帧分类，路径表示加载指定的时序头。"""
        if checkpoint is not None:
            target = Path(checkpoint)
        elif demo_exp:
            target = project_path("exp.pt")
        else:
            target = checkpoint_path(rc.resolve_out_dir())
        if not target.is_file():
            raise DataError(
                f"找不到模型权重：{target}",
                hint="演示版需要项目根目录的 exp.pt；正式版需要云端训练后复制的项目 best.pt。",
            )
        if demo_exp:
            head_path: Path | None
            if temporal_head is True:
                default_head = project_path(DEFAULT_TEMPORAL_HEAD)
                head_path = default_head if default_head.is_file() else None
            elif temporal_head is False:
                head_path = None
            else:
                head_path = Path(temporal_head)
            self.classifier = ExpDemoClassifier(rc, target, temporal_head=head_path)
            if head_path is None:
                log.info("未加载时序头：exp.pt 逐帧分类")
            else:
                log.info("已加载时序头：%s", head_path)
        elif temporal_head not in (True, False):
            raise DataError("时序头目前只适用于 --demo-exp 的 exp.pt")
        else:
            device = resolve_device(rc.runtime.device)
            model, _ = _load_model_from_checkpoint(rc, target, device=device)
            self.classifier = LiveClassifier(rc, model)
        self.rc = rc
        self.checkpoint = target
        self.demo_exp = demo_exp
        self._lock = threading.Lock()
        self._session: LiveSession | None = None
        self._preview_dir = tempfile.TemporaryDirectory(prefix="handwash-preview-")
        self._preview_id: str | None = None
        self._preview_file: Path | None = None

    def prepare_video_preview(self, source: BinaryIO, length: int) -> str:
        """临时生成浏览器可播放预览；不改动用户原片，也不写入项目数据目录。"""
        with self._lock:
            if self._session is not None:
                raise DataError("Stop the current session before loading another video")
            preview_id = uuid.uuid4().hex
            temporary_root = Path(self._preview_dir.name)
            original = temporary_root / f"{preview_id}.source"
            converted = temporary_root / f"{preview_id}.mp4"
            try:
                remaining = length
                with original.open("wb") as target:
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise DataError("The selected video was not received completely")
                        target.write(chunk)
                        remaining -= len(chunk)
                result = subprocess.run(
                    [
                        _ffmpeg_executable(),
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(original),
                        "-map",
                        "0:v:0",
                        "-map",
                        "0:a:0?",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "22",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-movflags",
                        "+faststart",
                        str(converted),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0 or not converted.is_file():
                    log.warning("无法生成本机视频预览：%s", result.stderr.strip())
                    raise DataError("The local service could not decode this video; see the terminal log")
                self._preview_id = preview_id
                self._preview_file = converted
                return f"/api/video/{preview_id}"
            finally:
                original.unlink(missing_ok=True)

    def preview_file(self, preview_id: str) -> Path | None:
        with self._lock:
            return self._preview_file if preview_id == self._preview_id else None

    def close(self) -> None:
        self._preview_dir.cleanup()

    def start(self) -> dict[str, object]:
        with self._lock:
            if self._session is not None:
                log.warning("旧摄像头会话被新会话替换：%s", self._session.session_id)
                if self._session.num_frames:
                    self._save_session(self._session)
            session_id = uuid.uuid4().hex
            self._session = LiveSession(self.classifier, session_id=session_id)
            return {
                "session_id": session_id,
                "sample_fps": self.rc.dataset.prep.fps,
                "model_name": self.classifier.model_name,
                "demo": self.demo_exp,
            }

    def frame(self, session_id: str, image: np.ndarray, *, capture_time_s: float) -> dict[str, object]:
        with self._lock:
            session = self._require_session(session_id)
            return session.add_frame(image, capture_time_s=capture_time_s)

    def stop(self, session_id: str) -> dict[str, object]:
        with self._lock:
            session = self._require_session(session_id)
            if session.num_frames == 0:
                self._session = None
                return {"session_id": session_id, "frame_count": 0, "report": None}
            report, json_path = self._save_session(session)
            self._session = None
            payload = report.to_dict()
            details = _english_violation_details(report, self.rc.assess)
            for violation, detail_en in zip(payload["violations"], details, strict=True):
                violation["detail_en"] = detail_en
            return {
                "session_id": session_id,
                "frame_count": session.num_frames,
                "report": payload,
                "report_path": str(json_path),
            }

    def _save_session(self, session: LiveSession):
        report = session.report()
        target = ensure_dir(self.rc.resolve_out_dir() / "camera" / session.session_id)
        json_path, _ = save_report(report, target, stem="report")
        write_jsonl(target / "predictions.jsonl", session.prediction_rows())
        return report, json_path

    def _require_session(self, session_id: str) -> LiveSession:
        if self._session is None or self._session.session_id != session_id:
            raise DataError("The session does not exist or has ended. Please start again.")
        return self._session


class _CameraHandler(BaseHTTPRequestHandler):
    server_version = "HandwashCamera/1"

    @property
    def app(self) -> CameraApp:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        log.info("camera HTTP: " + format, *args)

    def _send(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_video(self, path: Path) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match or not any(match.groups()):
                self.send_error(416, "Invalid video range")
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                count = int(last)
                start = max(0, size - count)
            if start > end or start >= size or (not first and count == 0):
                self.send_error(416, "Invalid video range")
                return
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as source:
            source.seek(start)
            remaining = end - start + 1
            try:
                while remaining:
                    chunk = source.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path.startswith("/api/video/"):
            preview = self.app.preview_file(path.removeprefix("/api/video/"))
            if preview is None:
                self._send(404, {"error": "Video preview not found"})
            else:
                self._send_video(preview)
            return
        if path != "/":
            self._send(404, {"error": "Page not found"})
            return
        body = files("handwash").joinpath("static/camera.html").read_bytes()
        # 步骤名称只在 core/labels.py 定义一次；页面加载时注入，避免前端再抄一份。
        body = body.replace(b"__HANDWASH_LABELS__", _LABELS_JSON)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        try:
            host = self.headers.get("Host", "")
            port = self.server.server_port
            if host not in {"127.0.0.1", "localhost", f"127.0.0.1:{port}", f"localhost:{port}"}:
                raise DataError("The camera service only accepts requests from this computer")
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{host}":
                raise DataError("The camera service does not accept requests from other websites")
            if parsed.path == "/api/video":
                if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/octet-stream":
                    raise DataError("Video uploads must use application/octet-stream")
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise DataError("Invalid video length") from exc
                if length < 1 or length > _MAX_VIDEO_BYTES:
                    raise DataError("The selected video must be between 1 byte and 128 MB")
                self._send(200, {"url": self.app.prepare_video_preview(self.rfile, length)})
                return
            if parsed.path == "/api/start":
                self._send(200, self.app.start())
                return
            session_id = query.get("session_id", [""])[0]
            if parsed.path == "/api/stop":
                self._send(200, self.app.stop(session_id))
                return
            if parsed.path == "/api/frame":
                raw_time = query.get("capture_time_s", [""])[0]
                try:
                    capture_time_s = float(raw_time)
                except ValueError as exc:
                    raise DataError("Missing a valid frame capture time") from exc
                if not math.isfinite(capture_time_s):
                    raise DataError("The frame capture time must be a finite number")
                if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "image/jpeg":
                    raise DataError("Frames must be sent as image/jpeg")
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise DataError("Invalid frame length") from exc
                if length < 1 or length > _MAX_JPEG_BYTES:
                    raise DataError(f"The JPEG frame must be between 1 and {_MAX_JPEG_BYTES} bytes")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise DataError("The JPEG frame was not received completely")
                try:
                    with Image.open(io.BytesIO(raw)) as image:
                        width, height = image.size
                        if width > _MAX_IMAGE_SIDE or height > _MAX_IMAGE_SIDE:
                            raise DataError("The frame is too large")
                        frame = np.asarray(image.convert("RGB"))
                except (UnidentifiedImageError, OSError) as exc:
                    raise DataError("Could not decode the JPEG frame") from exc
                self._send(200, self.app.frame(session_id, frame, capture_time_s=capture_time_s))
                return
            self._send(404, {"error": "Endpoint not found"})
        except HandwashError as exc:
            # 下游模块的错误信息是中文；网页只显示英文，原文记在终端日志里。
            log.warning("摄像头请求被拒绝：%s", exc)
            self._send(400, {"error": str(exc)})
        except Exception:
            log.exception("摄像头请求处理失败")
            self._send(500, {"error": "The server failed to process the request. See the terminal log."})


def serve_camera(
    rc: ResolvedConfig,
    *,
    checkpoint: str | Path | None = None,
    port: int = 8765,
    demo_exp: bool = False,
    temporal_head: str | Path | bool = True,
) -> None:
    """只监听本机；浏览器打开 http://127.0.0.1:<port>/。"""
    if not 1 <= port <= 65535:
        raise DataError(f"端口必须在 1..65535，实际 {port}")
    app = CameraApp(rc, checkpoint=checkpoint, demo_exp=demo_exp, temporal_head=temporal_head)
    server = ThreadingHTTPServer(("127.0.0.1", port), _CameraHandler)
    server.app = app  # type: ignore[attr-defined]
    log.info("摄像头页面：http://127.0.0.1:%d/；模型：%s", port, app.checkpoint)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        app.close()
