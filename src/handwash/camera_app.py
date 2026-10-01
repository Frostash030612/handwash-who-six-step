"""本机摄像头网页与逐帧推理 HTTP 入口。"""

from __future__ import annotations

import io
import json
import math
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image, UnidentifiedImageError

from handwash.core.config import ResolvedConfig
from handwash.errors import DataError, HandwashError
from handwash.io.utils import write_jsonl
from handwash.logging import get_logger
from handwash.paths import checkpoint_path, ensure_dir, project_path
from handwash.pipelines.assess import save_report
from handwash.pipelines.common import resolve_device
from handwash.pipelines.evaluate import _load_model_from_checkpoint
from handwash.pipelines.exp_demo import ExpDemoClassifier
from handwash.pipelines.live import LiveClassifier, LiveSession

__all__ = ["CameraApp", "serve_camera"]

log = get_logger(__name__)
_MAX_JPEG_BYTES = 4 * 1024 * 1024
_MAX_IMAGE_SIDE = 4096


class CameraApp:
    """一个本机浏览器摄像头会话；模型只加载一次。"""

    def __init__(
        self,
        rc: ResolvedConfig,
        *,
        checkpoint: str | Path | None = None,
        demo_exp: bool = False,
    ) -> None:
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
            self.classifier = ExpDemoClassifier(rc, target)
        else:
            device = resolve_device(rc.runtime.device)
            model, _ = _load_model_from_checkpoint(rc, target, device=device)
            self.classifier = LiveClassifier(rc, model)
        self.rc = rc
        self.checkpoint = target
        self.demo_exp = demo_exp
        self._lock = threading.Lock()
        self._session: LiveSession | None = None

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
            return {
                "session_id": session_id,
                "frame_count": session.num_frames,
                "report": report.to_dict(),
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
            raise DataError("摄像头会话不存在或已结束，请重新开始")
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

    def do_GET(self) -> None:
        if urlsplit(self.path).path != "/":
            self._send(404, {"error": "页面不存在"})
            return
        body = files("handwash").joinpath("static/camera.html").read_bytes()
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
                raise DataError("摄像头服务仅接受本机页面请求")
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{host}":
                raise DataError("摄像头接口不接受其他网站的请求")
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
                    raise DataError("缺少有效的摄像头采集时间") from exc
                if not math.isfinite(capture_time_s):
                    raise DataError("摄像头采集时间必须是有限数值")
                if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "image/jpeg":
                    raise DataError("摄像头帧必须以 image/jpeg 发送")
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise DataError("摄像头帧长度无效") from exc
                if length < 1 or length > _MAX_JPEG_BYTES:
                    raise DataError(f"摄像头 JPEG 必须在 1 到 {_MAX_JPEG_BYTES} 字节之间")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise DataError("摄像头 JPEG 传输不完整")
                try:
                    with Image.open(io.BytesIO(raw)) as image:
                        width, height = image.size
                        if width > _MAX_IMAGE_SIDE or height > _MAX_IMAGE_SIDE:
                            raise DataError("摄像头画面尺寸过大")
                        frame = np.asarray(image.convert("RGB"))
                except (UnidentifiedImageError, OSError) as exc:
                    raise DataError("无法解码摄像头 JPEG") from exc
                self._send(200, self.app.frame(session_id, frame, capture_time_s=capture_time_s))
                return
            self._send(404, {"error": "接口不存在"})
        except HandwashError as exc:
            self._send(400, {"error": str(exc)})
        except Exception:
            log.exception("摄像头请求处理失败")
            self._send(500, {"error": "服务处理失败，请查看终端日志"})


def serve_camera(
    rc: ResolvedConfig,
    *,
    checkpoint: str | Path | None = None,
    port: int = 8765,
    demo_exp: bool = False,
) -> None:
    """只监听本机；浏览器打开 http://127.0.0.1:<port>/。"""
    if not 1 <= port <= 65535:
        raise DataError(f"端口必须在 1..65535，实际 {port}")
    app = CameraApp(rc, checkpoint=checkpoint, demo_exp=demo_exp)
    server = ThreadingHTTPServer(("127.0.0.1", port), _CameraHandler)
    server.app = app  # type: ignore[attr-defined]
    log.info("摄像头页面：http://127.0.0.1:%d/；模型：%s", port, app.checkpoint)
    try:
        server.serve_forever()
    finally:
        server.server_close()
