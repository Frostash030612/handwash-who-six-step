"""Loopback-only upload page for comparing three archived handwashing models."""

from __future__ import annotations

import json
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from handwash.core.config import ResolvedConfig
from handwash.pipelines.compare_local import compare_video

MAX_UPLOAD_BYTES = 256 * 1024 * 1024


class CompareApp:
    def __init__(self, cfg: ResolvedConfig) -> None:
        self.cfg = cfg
        self.lock = threading.Lock()
        self.job: dict | None = None

    def submit(self, source, length: int) -> dict:
        if not 0 < length <= MAX_UPLOAD_BYTES:
            raise ValueError("视频大小须在 1 B 到 256 MB 之间")
        with self.lock:
            if self.job and self.job["state"] in {"uploading", "running"}:
                raise ValueError("上一段视频还在分析，请稍后再上传")
            job_id = uuid.uuid4().hex
            self.job = {"id": job_id, "state": "uploading", "progress": "接收视频"}
        try:
            with tempfile.NamedTemporaryFile(prefix="handwash-compare-", suffix=".mp4", delete=False) as handle:
                path = Path(handle.name)
                remaining = length
                while remaining:
                    part = source.read(min(1024 * 1024, remaining))
                    if not part:
                        raise ValueError("视频上传不完整")
                    handle.write(part)
                    remaining -= len(part)
        except Exception:
            if "path" in locals():
                path.unlink(missing_ok=True)
            with self.lock:
                self.job = None
            raise
        with self.lock:
            self.job["state"] = "running"
            self.job["progress"] = "等待分析"
        threading.Thread(target=self._run, args=(job_id, path), daemon=True).start()
        return {"id": job_id}

    def _run(self, job_id: str, path: Path) -> None:
        def update(message: str) -> None:
            with self.lock:
                if self.job and self.job["id"] == job_id:
                    self.job["progress"] = message

        try:
            result = compare_video(path, self.cfg, update)
            with self.lock:
                self.job = {"id": job_id, "state": "done", "progress": "完成", "result": result}
        except Exception as exc:
            with self.lock:
                self.job = {"id": job_id, "state": "error", "progress": "失败", "error": str(exc)}
        finally:
            path.unlink(missing_ok=True)

    def status(self, job_id: str) -> dict:
        with self.lock:
            if not self.job or self.job["id"] != job_id:
                raise ValueError("找不到这次分析任务")
            return dict(self.job)


class _Handler(BaseHTTPRequestHandler):
    server_version = "HandwashCompare/1"

    @property
    def app(self) -> CompareApp:
        return self.server.app  # type: ignore[attr-defined]

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/":
            body = files("handwash").joinpath("static/compare.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/status":
            try:
                self._send_json(200, self.app.status(parse_qs(parsed.query).get("id", [""])[0]))
            except ValueError as exc:
                self._send_json(404, {"error": str(exc)})
            return
        self._send_json(404, {"error": "页面不存在"})

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/compare":
            self._send_json(404, {"error": "接口不存在"})
            return
        host = self.headers.get("Host", "")
        port = self.server.server_port
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            self._send_json(403, {"error": "仅允许本机访问"})
            return
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{host}":
            self._send_json(403, {"error": "不接受其他网页上传"})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/octet-stream":
            self._send_json(400, {"error": "请上传视频文件"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            self._send_json(200, self.app.submit(self.rfile, length))
        except (ValueError, OSError) as exc:
            self._send_json(400, {"error": str(exc)})


def serve_compare(cfg: ResolvedConfig, port: int = 8766) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    server.app = CompareApp(cfg)  # type: ignore[attr-defined]
    print(f"同视频模型对照页面：http://127.0.0.1:{port}/", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
