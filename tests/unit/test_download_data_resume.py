"""Interrupted Zenodo responses must resume rather than discard useful bytes."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


def _load_downloader():
    script = Path(__file__).resolve().parents[2] / "scripts" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_data", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, data: bytes, *, status: int, headers: dict[str, str]):
        self.data = data
        self.status = status
        self.headers = headers
        self.sent = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, _size: int) -> bytes:
        if self.sent:
            return b""
        self.sent = True
        return self.data


def test_truncated_response_is_resumed(tmp_path, monkeypatch) -> None:
    module = _load_downloader()
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    requests = []

    def open_response(request, **_):
        requests.append(request)
        if len(requests) == 1:
            return _Response(b"ab", status=200, headers={"Content-Length": "5"})
        return _Response(
            b"cde", status=206,
            headers={"Content-Length": "3", "Content-Range": "bytes 2-4/5"},
        )

    monkeypatch.setattr(module.urllib.request, "urlopen", open_response)
    item = {"key": "DataSet4.zip", "size": 5, "md5": hashlib.md5(b"abcde").hexdigest()}

    ok, reason = module.download_file("pskuss", item, destination=tmp_path, retries=2, quiet=True)

    assert ok, reason
    assert (tmp_path / "DataSet4.zip").read_bytes() == b"abcde"
    assert not (tmp_path / "DataSet4.zip.part").exists()
    assert requests[1].get_header("Range") == "bytes=2-"
