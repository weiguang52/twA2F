from __future__ import annotations

import inspect
import json
import queue
import threading
import time
import textwrap
from collections import deque
from http import server
from typing import Deque, Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .render_motor_npy import (
    ACCENT,
    BG,
    CURSOR,
    FG,
    H,
    W,
    WAVE,
    DEFAULT_MOTOR_NEUTRAL,
    draw_brow,
    draw_chin,
    draw_energy_bar,
    draw_eye,
    draw_face_outline,
    draw_motor_panel,
    draw_mouth,
)


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


class _ThreadingHTTPServer(server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _load_cn_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


FONT_32 = _load_cn_font(32)
FONT_28 = _load_cn_font(28)
FONT_24 = _load_cn_font(24)
FONT_20 = _load_cn_font(20)
FONT_18 = _load_cn_font(18)


def _draw_motor_panel_compat(img: np.ndarray, x: int, y: int, motor_names, motor_values, neutral_map) -> None:
    """Render a simple local motor panel without depending on render_motor_npy.draw_motor_panel signature."""
    names = list(motor_names or [])
    values = list(motor_values or [])
    neutral_map = dict(neutral_map or {})

    panel_w = 380
    row_h = 22
    max_rows = 14
    header_h = 34
    padding = 12
    visible = min(max_rows, max(len(names), len(values), 1))
    panel_h = header_h + padding + visible * row_h + padding

    # background
    cv2.rectangle(img, (x, y), (x + panel_w, y + panel_h), (26, 31, 40), -1)
    cv2.rectangle(img, (x, y), (x + panel_w, y + panel_h), (52, 61, 76), 1)
    cv2.putText(img, "motor panel", (x + 12, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, FG, 2, cv2.LINE_AA)

    if not names:
        cv2.putText(img, "(no motor output)", (x + 14, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FG, 1, cv2.LINE_AA)
        return

    bar_x = x + 150
    bar_w = 170
    mid_x = bar_x + bar_w // 2

    for i, name in enumerate(names[:max_rows]):
        v = float(values[i]) if i < len(values) else 0.0
        neutral = float(neutral_map.get(name, 0.0))
        delta = max(-1.0, min(1.0, v - neutral))
        yy = y + header_h + padding + i * row_h

        cv2.putText(img, str(name)[:18], (x + 12, yy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, FG, 1, cv2.LINE_AA)
        cv2.rectangle(img, (bar_x, yy), (bar_x + bar_w, yy + 10), (44, 52, 66), -1)
        cv2.line(img, (mid_x, yy - 2), (mid_x, yy + 12), CURSOR, 1, cv2.LINE_AA)

        half = bar_w // 2
        if delta >= 0:
            fill = int(round(delta * half))
            if fill > 0:
                cv2.rectangle(img, (mid_x, yy), (mid_x + fill, yy + 10), ACCENT, -1)
        else:
            fill = int(round(-delta * half))
            if fill > 0:
                cv2.rectangle(img, (mid_x - fill, yy), (mid_x, yy + 10), WAVE, -1)

        cv2.putText(img, f"{v:.3f}", (bar_x + bar_w + 10, yy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, FG, 1, cv2.LINE_AA)



def _wrap_cn_lines(text: str, max_chars: int) -> list[str]:
    lines = []
    for raw in str(text).splitlines() or [""]:
        lines.extend(
            textwrap.wrap(
                raw,
                width=max_chars,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
        )
    return lines


def draw_text_batch_cn(frame_bgr: np.ndarray, ops: list[dict]) -> np.ndarray:
    if not ops:
        return frame_bgr

    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    for op in ops:
        kind = op.get("kind", "text")
        x, y = op["xy"]
        font = op["font"]
        bgr = op.get("color", (0, 0, 0))
        fill = (bgr[2], bgr[1], bgr[0])

        if kind == "text":
            draw.text((x, y), str(op.get("text", "")), font=font, fill=fill)
            continue

        yy = y
        max_chars = int(op.get("max_chars", 34))
        line_spacing = int(op.get("line_spacing", 8))
        for line in _wrap_cn_lines(op.get("text", ""), max_chars=max_chars):
            draw.text((x, yy), line, font=font, fill=fill)
            yy += font.size + line_spacing

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


class InteractiveWebRealtimeA2FPreview:
    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        render_fps: float = 30.0,
        waveform_seconds: float = 6.0,
        transcript_chars: int = 180,
        title: str = "A2F Interactive Web Preview",
    ) -> None:
        self.host = host
        self.port = int(port)
        self.render_fps = max(2.0, float(render_fps))
        self.waveform_seconds = max(1.0, float(waveform_seconds))
        self.transcript_chars = max(40, int(transcript_chars))
        self.title = title

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._render_thread: Optional[threading.Thread] = None
        self._server_thread: Optional[threading.Thread] = None
        self._httpd: Optional[_ThreadingHTTPServer] = None
        self._prompt_queue: "queue.Queue[str]" = queue.Queue()

        self._round_index = 0
        self._busy = False
        self._busy_text = ""
        self._session_id = "-"
        self._request_id = "-"
        self._server_action_desc = ""
        self._transcript_parts: Deque[str] = deque(maxlen=400)
        self._latest_preview: Optional[Dict[str, object]] = None
        self._latest_audio_sr = 24000
        self._latest_audio_channels = 1
        self._latest_audio_bytes = 0
        self._audio_ring: np.ndarray = np.zeros(0, dtype=np.float32)
        self._latest_audio_latency_s: Optional[float] = None
        self._first_local_frame_latency_s: Optional[float] = None
        self._received_audio_ms = 0.0
        self._server_audio_chunks = 0
        self._local_frames = 0
        self._status_line = "waiting for text input..."
        self._last_error = ""
        self._last_submit_ts: Optional[float] = None
        self._last_finish_ts: Optional[float] = None
        self._last_frame_bgr = self._build_placeholder_frame("waiting for text input...")
        ok, enc = cv2.imencode(".jpg", self._last_frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        self._latest_jpeg = enc.tobytes() if ok else b""
        self._latest_frame_ts = time.time()

    def start(self) -> None:
        if self._httpd is not None:
            return

        owner = self

        class Handler(server.BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                return

            def _send_json(self, code: int, payload: Dict[str, object]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    body = owner._build_index_html().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/state":
                    self._send_json(200, owner.get_state())
                    return

                if self.path == "/snapshot.jpg":
                    payload = owner.get_latest_jpeg()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                if self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    last_ts = 0.0
                    try:
                        while not owner._stop_event.is_set():
                            jpeg, ts = owner.get_latest_jpeg_with_ts()
                            if ts == last_ts:
                                time.sleep(0.03)
                                continue
                            last_ts = ts
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return

                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(body.decode("utf-8") or "{}")
                except Exception:
                    payload = {}

                if self.path == "/submit":
                    print("[WEB] /submit payload =", payload, flush=True)
                    text = str(payload.get("text", "")).strip()
                    print("[WEB] parsed text =", repr(text), flush=True)
                    if not text:
                        self._send_json(400, {"ok": False, "error": "text is empty"})
                        return
                    owner.enqueue_text(text)
                    self._send_json(200, {"ok": True, "queued": owner.pending_count, "text": text})
                    return

                if self.path == "/clear":
                    owner.clear_overlay_text()
                    self._send_json(200, {"ok": True})
                    return

                self.send_response(404)
                self.end_headers()

        self._httpd = _ThreadingHTTPServer((self.host, self.port), Handler)
        self._server_thread = threading.Thread(target=self._httpd.serve_forever, name="A2FInteractiveWebHTTP", daemon=True)
        self._server_thread.start()
        self._render_thread = threading.Thread(target=self._render_loop, name="A2FInteractiveWebRender", daemon=True)
        self._render_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
            except Exception:
                pass
            try:
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None
        if self._render_thread is not None:
            self._render_thread.join(timeout=2.0)
            self._render_thread = None

    @property
    def url(self) -> str:
        return f"http://{self.host if self.host != '0.0.0.0' else '127.0.0.1'}:{self.port}"

    @property
    def pending_count(self) -> int:
        return self._prompt_queue.qsize()

    def enqueue_text(self, text: str) -> None:
        print("[WEB] enqueue_text =", repr(text), flush=True)
        self._prompt_queue.put(str(text))
        print("[WEB] queue size =", self._prompt_queue.qsize(), flush=True)
        with self._lock:
            self._last_submit_ts = time.time()
            if not self._busy:
                self._status_line = "queued; waiting to start..."

    def get_next_text(self, timeout: float = 0.5) -> Optional[str]:
        if self._stop_event.is_set():
            return None
        try:
            return self._prompt_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def start_round(self, *, session_id: str, request_id: str, text: str) -> None:
        with self._lock:
            self._round_index += 1
            self._busy = True
            self._busy_text = text
            self._session_id = session_id
            self._request_id = request_id
            self._server_action_desc = ""
            self._transcript_parts.clear()
            self._latest_preview = None
            self._latest_audio_bytes = 0
            self._audio_ring = np.zeros(0, dtype=np.float32)
            self._latest_audio_latency_s = None
            self._first_local_frame_latency_s = None
            self._received_audio_ms = 0.0
            self._server_audio_chunks = 0
            self._local_frames = 0
            self._status_line = "running..."
            self._last_error = ""
            self._last_finish_ts = None

    def finish_round(self, *, status: str = "finished") -> None:
        with self._lock:
            self._busy = False
            self._status_line = status
            self._last_finish_ts = time.time()

    def set_error(self, text: str) -> None:
        with self._lock:
            self._busy = False
            self._last_error = str(text)
            self._status_line = f"error: {text}"
            self._last_finish_ts = time.time()

    def clear_overlay_text(self) -> None:
        with self._lock:
            self._transcript_parts.clear()
            self._server_action_desc = ""
            self._status_line = "cleared"

    def update(
        self,
        *,
        preview: Optional[Dict[str, object]] = None,
        transcript: Optional[str] = None,
        server_action_desc: Optional[str] = None,
        pcm16_chunk: Optional[bytes] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        first_audio_latency_s: Optional[float] = None,
        first_local_frame_latency_s: Optional[float] = None,
        status: Optional[str] = None,
    ) -> None:
        with self._lock:
            if transcript:
                self._transcript_parts.append(str(transcript))
            if server_action_desc is not None and server_action_desc != "":
                self._server_action_desc = str(server_action_desc)
            if preview is not None:
                self._latest_preview = dict(preview)
                self._session_id = str(preview.get("session_id", self._session_id))
                self._request_id = str(preview.get("request_id", self._request_id))
                self._local_frames = int(preview.get("local_frame_count", self._local_frames))
            if sample_rate:
                self._latest_audio_sr = int(sample_rate)
            if channels:
                self._latest_audio_channels = int(channels)
            if first_audio_latency_s is not None:
                self._latest_audio_latency_s = float(first_audio_latency_s)
            if first_local_frame_latency_s is not None:
                self._first_local_frame_latency_s = float(first_local_frame_latency_s)
            if status is not None:
                self._status_line = str(status)

            if pcm16_chunk:
                self._server_audio_chunks += 1
                self._latest_audio_bytes = len(pcm16_chunk)
                sr = max(1, self._latest_audio_sr)
                ch = max(1, self._latest_audio_channels)
                chunk_arr = np.frombuffer(pcm16_chunk, dtype=np.int16).astype(np.float32)
                if ch > 1:
                    usable = (len(chunk_arr) // ch) * ch
                    chunk_arr = chunk_arr[:usable].reshape(-1, ch).mean(axis=1)
                chunk_arr /= 32768.0
                self._audio_ring = np.concatenate([self._audio_ring, chunk_arr], axis=0)
                max_samples = int(self.waveform_seconds * sr)
                if len(self._audio_ring) > max_samples:
                    self._audio_ring = self._audio_ring[-max_samples:]
                self._received_audio_ms += (len(chunk_arr) / float(sr)) * 1000.0

    def get_latest_jpeg(self) -> bytes:
        with self._lock:
            return bytes(self._latest_jpeg)

    def get_latest_jpeg_with_ts(self) -> tuple[bytes, float]:
        with self._lock:
            return bytes(self._latest_jpeg), float(self._latest_frame_ts)

    def get_state(self) -> Dict[str, object]:
        with self._lock:
            transcript = "".join(self._transcript_parts)
            preview = dict(self._latest_preview or {})
            return {
                "session_id": self._session_id,
                "request_id": self._request_id,
                "status": self._status_line,
                "busy": self._busy,
                "busy_text": self._busy_text,
                "pending_count": self._prompt_queue.qsize(),
                "round_index": self._round_index,
                "last_error": self._last_error,
                "server_action_desc": self._server_action_desc,
                "transcript": transcript,
                "received_audio_ms": round(self._received_audio_ms, 1),
                "server_audio_chunks": self._server_audio_chunks,
                "local_frames": self._local_frames,
                "latest_audio_bytes": self._latest_audio_bytes,
                "first_audio_latency_s": self._latest_audio_latency_s,
                "first_local_frame_latency_s": self._first_local_frame_latency_s,
                "current_t": round(_safe_float(preview.get("t"), 0.0), 3),
                "last_local_process_ms": round(_safe_float(preview.get("last_local_process_ms"), 0.0), 2),
                "avg_local_process_ms": round(_safe_float(preview.get("avg_local_process_ms"), 0.0), 2),
                "max_local_process_ms": round(_safe_float(preview.get("max_local_process_ms"), 0.0), 2),
                "local_lag_ms": round(_safe_float(preview.get("local_lag_ms"), 0.0), 1),
                "updated_at": self._latest_frame_ts,
                "last_submit_ts": self._last_submit_ts,
                "last_finish_ts": self._last_finish_ts,
            }

    def _render_loop(self) -> None:
        frame_interval = 1.0 / self.render_fps
        while not self._stop_event.is_set():
            t0 = time.time()
            frame = self._build_frame()
            display = cv2.resize(frame, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                with self._lock:
                    self._last_frame_bgr = frame
                    self._latest_jpeg = enc.tobytes()
                    self._latest_frame_ts = time.time()
            spent = time.time() - t0
            if spent < frame_interval:
                time.sleep(frame_interval - spent)

    def _build_index_html(self) -> str:
        return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{self.title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #0f1115; color: #eef1f4; }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 16px; }}
    .top {{ display: grid; grid-template-columns: 1fr 380px; gap: 16px; align-items: start; }}
    .card {{ background: #171b22; border: 1px solid #272d37; border-radius: 12px; overflow: hidden; box-shadow: 0 8px 30px rgba(0,0,0,.25); }}
    .imgbox {{ background: #0b0e13; }}
    canvas {{ display: block; width: 100%; height: auto; background: #0b0e13; }}
    .meta {{ padding: 14px 16px; line-height: 1.55; font-size: 14px; }}
    .label {{ color: #8ea2c1; }}
    .value {{ color: #ffffff; font-weight: 600; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; word-break: break-all; }}
    h1 {{ font-size: 20px; margin: 0 0 14px; }}
    h2 {{ font-size: 15px; margin: 0 0 8px; color: #9fc2ff; }}
    h3 {{ font-size: 14px; margin: 12px 0 6px; color: #b9d1ff; }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; color: #f3f6fa; min-height: 40px; }}
    .foot {{ margin-top: 12px; color: #90a0b7; font-size: 12px; }}
    textarea {{ width: 100%; min-height: 90px; background: #0f1319; color: #eef1f4; border: 1px solid #313b49; border-radius: 10px; padding: 10px 12px; box-sizing: border-box; resize: vertical; }}
    .row {{ display: flex; gap: 8px; margin-top: 8px; }}
    button {{ background: #3c82f6; color: white; border: none; border-radius: 10px; padding: 10px 14px; cursor: pointer; font-weight: 600; }}
    button.secondary {{ background: #39404a; }}
    button:disabled {{ opacity: 0.6; cursor: not-allowed; }}
    .small {{ font-size: 12px; color: #91a4bf; }}
    .status-pill {{ display:inline-block; padding:4px 9px; border-radius:999px; background:#243247; color:#d6e4ff; font-size:12px; margin-left:6px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{self.title}<span class="status-pill" id="busy_state">idle</span></h1>
    <div class="top">
      <div class="card imgbox">
        <canvas id="frame_canvas" width="1280" height="720"></canvas>
      </div>
      <div class="card meta">
        <h2>Send new text</h2>
        <textarea id="text_input" placeholder="输入一条新的文本，点击发送后会进入队列并开始新一轮 TTS -> A2F -> 实时预览"></textarea>
        <div class="row">
          <button id="send_btn" onclick="submitText()">发送</button>
          <button class="secondary" onclick="clearOverlay()">清空字幕/动作描述</button>
        </div>
        <div class="small" id="submit_msg">前端以 30fps 目标频率轮询 /snapshot.jpg；支持连续多轮发送。</div>

        <h3>Live status</h3>
        <div><span class="label">session_id:</span> <span class="value mono" id="session_id">-</span></div>
        <div><span class="label">request_id:</span> <span class="value mono" id="request_id">-</span></div>
        <div><span class="label">status:</span> <span class="value" id="status">-</span></div>
        <div><span class="label">round_index:</span> <span class="value" id="round_index">0</span></div>
        <div><span class="label">pending_count:</span> <span class="value" id="pending_count">0</span></div>
        <div><span class="label">current_t:</span> <span class="value" id="current_t">0</span></div>
        <div><span class="label">received_audio_ms:</span> <span class="value" id="received_audio_ms">0</span></div>
        <div><span class="label">server_audio_chunks:</span> <span class="value" id="server_audio_chunks">0</span></div>
        <div><span class="label">local_frames:</span> <span class="value" id="local_frames">0</span></div>
        <div><span class="label">latest_audio_bytes:</span> <span class="value" id="latest_audio_bytes">0</span></div>
        <div><span class="label">first_audio_latency_s:</span> <span class="value" id="first_audio_latency_s">-</span></div>
        <div><span class="label">first_local_frame_latency_s:</span> <span class="value" id="first_local_frame_latency_s">-</span></div>
        <div><span class="label">last_local_process_ms:</span> <span class="value" id="last_local_process_ms">0</span></div>
        <div><span class="label">avg_local_process_ms:</span> <span class="value" id="avg_local_process_ms">0</span></div>
        <div><span class="label">max_local_process_ms:</span> <span class="value" id="max_local_process_ms">0</span></div>
        <div><span class="label">local_lag_ms:</span> <span class="value" id="local_lag_ms">0</span></div>
        <div><span class="label">last_error:</span> <span class="value" id="last_error">-</span></div>

        <h3>Current text</h3>
        <pre id="busy_text">-</pre>
        <h3>Transcript</h3>
        <pre id="transcript">-</pre>
        <h3>Server action</h3>
        <pre id="server_action_desc">-</pre>
      </div>
    </div>
    <div class="foot">本页每 500ms 刷新状态，并以 30fps 目标频率通过 /snapshot.jpg 拉帧绘制到 canvas。</div>
  </div>
  <script>
    const TARGET_FPS = 30;
    const FRAME_INTERVAL_MS = Math.max(1, Math.round(1000 / TARGET_FPS));
    const canvas = document.getElementById('frame_canvas');
    const ctx = canvas.getContext('2d');
    let previewRunning = true;
    let previewInflight = false;
    let previewObjectUrl = null;

    function sleep(ms) {{
      return new Promise(resolve => setTimeout(resolve, ms));
    }}

    async function submitText() {{
      const btn = document.getElementById('send_btn');
      const input = document.getElementById('text_input');
      const msg = document.getElementById('submit_msg');
      const text = input.value.trim();
      if (!text) {{ msg.textContent = '请输入文本'; return; }}
      btn.disabled = true;
      msg.textContent = '发送中...';
      try {{
        const resp = await fetch('/submit', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ text }})
        }});
        const data = await resp.json();
        if (!resp.ok || !data.ok) throw new Error(data.error || 'submit failed');
        msg.textContent = `已加入队列，当前队列长度：${{data.queued}}`;
        input.value = '';
      }} catch (e) {{
        msg.textContent = '发送失败：' + (e?.message || e);
      }} finally {{
        btn.disabled = false;
      }}
    }}

    async function clearOverlay() {{
      try {{
        await fetch('/clear', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: '{{}}' }});
      }} catch (e) {{}}
    }}

    async function refreshState() {{
      try {{
        const resp = await fetch('/state', {{ cache: 'no-store' }});
        const s = await resp.json();
        for (const key of [
          'session_id','request_id','status','round_index','pending_count','current_t',
          'received_audio_ms','server_audio_chunks','local_frames','latest_audio_bytes',
          'transcript','server_action_desc','busy_text','last_error',
          'last_local_process_ms','avg_local_process_ms','max_local_process_ms','local_lag_ms'
        ]) {{
          const el = document.getElementById(key);
          if (el) el.textContent = (s[key] ?? '-');
        }}
        const busyPill = document.getElementById('busy_state');
        if (busyPill) busyPill.textContent = s.busy ? 'busy' : 'idle';
        const a = document.getElementById('first_audio_latency_s');
        if (a) a.textContent = s.first_audio_latency_s == null ? '-' : s.first_audio_latency_s;
        const b = document.getElementById('first_local_frame_latency_s');
        if (b) b.textContent = s.first_local_frame_latency_s == null ? '-' : s.first_local_frame_latency_s;
      }} catch (e) {{}}
    }}

    async function drawSnapshot() {{
      if (!previewRunning || previewInflight) return;
      previewInflight = true;
      const t0 = performance.now();
      try {{
        const resp = await fetch('/snapshot.jpg?t=' + Date.now(), {{ cache: 'no-store' }});
        if (!resp.ok) throw new Error('snapshot fetch failed: ' + resp.status);
        const blob = await resp.blob();
        if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
        previewObjectUrl = URL.createObjectURL(blob);
        const img = new Image();
        await new Promise((resolve, reject) => {{
          img.onload = resolve;
          img.onerror = reject;
          img.src = previewObjectUrl;
        }});
        if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {{
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
        }}
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      }} catch (e) {{
      }} finally {{
        previewInflight = false;
        const elapsed = performance.now() - t0;
        if (previewRunning) setTimeout(drawSnapshot, Math.max(1, FRAME_INTERVAL_MS - elapsed));
      }}
    }}

    document.addEventListener('visibilitychange', () => {{
      previewRunning = !document.hidden;
      if (previewRunning && !previewInflight) drawSnapshot();
    }});

    setInterval(refreshState, 500);
    refreshState();
    drawSnapshot();
  </script>
</body>
</html>
"""

    def _build_placeholder_frame(self, text: str) -> np.ndarray:
        img = np.full((H, W, 3), BG, dtype=np.uint8)
        return draw_text_batch_cn(
            img,
            [
                {"kind": "text", "text": self.title, "xy": (60, 60), "font": FONT_32, "color": FG},
                {"kind": "text", "text": text, "xy": (60, 120), "font": FONT_24, "color": ACCENT},
            ],
        )

    def _build_frame(self) -> np.ndarray:
        with self._lock:
            preview = dict(self._latest_preview or {})
            transcript = "".join(self._transcript_parts)
            server_action_desc = self._server_action_desc
            waveform = self._audio_ring.copy()
            audio_sr = int(self._latest_audio_sr)
            latest_audio_bytes = int(self._latest_audio_bytes)
            first_audio_latency_s = self._latest_audio_latency_s
            first_local_frame_latency_s = self._first_local_frame_latency_s
            status_line = self._status_line
            session_id = self._session_id
            request_id = self._request_id
            received_audio_ms = float(self._received_audio_ms)
            server_audio_chunks = int(self._server_audio_chunks)
            local_frames = int(self._local_frames)
            round_index = int(self._round_index)
            busy = bool(self._busy)
            busy_text = str(self._busy_text)
            pending_count = self._prompt_queue.qsize()
            last_local_process_ms = float(preview.get("last_local_process_ms", 0.0))
            avg_local_process_ms = float(preview.get("avg_local_process_ms", 0.0))
            max_local_process_ms = float(preview.get("max_local_process_ms", 0.0))
            local_lag_ms = float(preview.get("local_lag_ms", 0.0))

        img = np.full((H, W, 3), BG, dtype=np.uint8)
        draw_face_outline(img)
        face_cx = W // 2 - 120

        features = preview.get("features") or {}
        motor_names = list(preview.get("motor_names") or [])
        motor_values = list(preview.get("motor_values") or [])

        jaw_open = np.clip(_safe_float(features.get("jaw_open"), 0.0), 0.0, 1.0)
        mouth_wide = np.clip(_safe_float(features.get("mouth_wide"), 0.0), 0.0, 1.0)
        mouth_round = np.clip(_safe_float(features.get("mouth_round"), 0.0), 0.0, 1.0)
        mouth_lr = np.clip(_safe_float(features.get("mouth_left_right"), 0.0), -1.0, 1.0)
        upper_face = np.clip(_safe_float(features.get("upper_face_activity"), 0.0), 0.0, 1.0)
        blink = np.clip(_safe_float(features.get("blink_like"), 0.0), 0.0, 1.0)
        eye_act = np.clip(_safe_float(features.get("eye_activity"), 0.0), 0.0, 1.0)

        brow_lift = np.clip(0.25 + 0.85 * upper_face, 0.0, 1.0)
        brow_tilt = 0.15 * mouth_lr
        eye_open = np.clip(1.0 - 0.9 * blink + 0.1 * eye_act, 0.03, 1.0)
        lower_raise = np.clip(0.35 * blink + 0.2 * eye_act, 0.0, 1.0)

        draw_brow(img, face_cx - 130, 280, 120, brow_tilt, brow_lift, side=-1)
        draw_brow(img, face_cx + 130, 280, 120, brow_tilt, brow_lift, side=1)
        draw_eye(img, face_cx - 130, 390, eye_open, lower_raise)
        draw_eye(img, face_cx + 130, 390, eye_open, lower_raise)
        draw_mouth(img, face_cx, 620, jaw_open, mouth_wide, mouth_round, mouth_lr)
        draw_chin(img, face_cx, 700, jaw_open)

        draw_energy_bar(img, 60, 90, 180, 18, np.clip(_safe_float(features.get("skin_energy"), 0.0), 0.0, 1.0), label="jaw")
        draw_energy_bar(img, 60, 140, 180, 18, np.clip(_safe_float(features.get("eyes_energy"), 0.0), 0.0, 1.0), label="eyes")
        draw_energy_bar(img, 60, 190, 180, 18, np.clip(_safe_float(features.get("tongue_energy"), 0.0), 0.0, 1.0), label="tongue")

        cv2.rectangle(img, (230, 90), (760, 170), (210, 210, 210), 2)
        cv2.rectangle(img, (230, 190), (760, 285), (210, 210, 210), 2)
        cv2.rectangle(img, (230, 305), (760, 395), (210, 210, 210), 2)

        _draw_motor_panel_compat(
            img,
            760,
            175,
            motor_names=motor_names,
            motor_values=motor_values,
            neutral_map=DEFAULT_MOTOR_NEUTRAL,
        )

        x0, y0, ww, hh = 60, 950, 980, 130
        cv2.rectangle(img, (x0, y0), (x0 + ww, y0 + hh), (210, 210, 210), 2)
        if len(waveform) > 2:
            if len(waveform) > ww:
                idx = np.linspace(0, len(waveform) - 1, ww).astype(np.int32)
                wave = waveform[idx]
            else:
                wave = np.interp(np.linspace(0, len(waveform) - 1, ww), np.arange(len(waveform)), waveform)
            mid = y0 + hh // 2
            amp = int(hh * 0.42)
            pts = []
            for i, v in enumerate(wave):
                yy = int(mid - np.clip(v, -1.0, 1.0) * amp)
                pts.append((x0 + i, yy))
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], WAVE, 1, cv2.LINE_AA)
            cv2.line(img, (x0, mid), (x0 + ww, mid), CURSOR, 1, cv2.LINE_AA)
            cv2.line(img, (x0 + ww - 1, y0), (x0 + ww - 1, y0 + hh), CURSOR, 2, cv2.LINE_AA)

        text_ops = [
            {"kind": "text", "text": self.title, "xy": (50, 50), "font": FONT_32, "color": FG},
            {"kind": "text", "text": f"round: {round_index}   busy: {1 if busy else 0}   queue: {pending_count}", "xy": (50, 88), "font": FONT_20, "color": FG},
            {"kind": "text", "text": f"local a2f time: {_safe_float(preview.get('t'), 0.0):.2f}s", "xy": (760, 50), "font": FONT_24, "color": FG},
            {"kind": "text", "text": f"server audio chunks: {server_audio_chunks}", "xy": (760, 88), "font": FONT_20, "color": FG},
            {"kind": "text", "text": f"received audio: {received_audio_ms/1000.0:.2f}s", "xy": (760, 122), "font": FONT_20, "color": FG},
            {"kind": "text", "text": "robot motor preview", "xy": (760, 155), "font": FONT_28, "color": FG},
            {"kind": "multiline", "text": busy_text or "(waiting for next text...)", "xy": (245, 105), "font": FONT_28, "color": FG, "max_chars": 28},
            {"kind": "multiline", "text": transcript or "(waiting transcript...)", "xy": (245, 205), "font": FONT_24, "color": FG, "max_chars": 34},
            {"kind": "multiline", "text": server_action_desc or "-", "xy": (245, 320), "font": FONT_20, "color": FG, "max_chars": 48},
            {"kind": "text", "text": f"session: {session_id}", "xy": (60, 835), "font": FONT_20, "color": FG},
            {"kind": "text", "text": f"request: {request_id}", "xy": (60, 870), "font": FONT_20, "color": FG},
            {"kind": "text", "text": f"status: {status_line}", "xy": (60, 905), "font": FONT_24, "color": ACCENT},
            {"kind": "text", "text": f"first audio latency: {'-' if first_audio_latency_s is None else f'{first_audio_latency_s:.3f}s'}", "xy": (760, 200), "font": FONT_18, "color": FG},
            {"kind": "text", "text": f"first local frame: {'-' if first_local_frame_latency_s is None else f'{first_local_frame_latency_s:.3f}s'}", "xy": (760, 225), "font": FONT_18, "color": FG},
            {"kind": "text", "text": f"local frames: {local_frames}", "xy": (760, 250), "font": FONT_18, "color": FG},
            {"kind": "text", "text": f"last proc: {last_local_process_ms:.2f}ms", "xy": (760, 275), "font": FONT_18, "color": FG},
            {"kind": "text", "text": f"avg proc: {avg_local_process_ms:.2f}ms", "xy": (760, 300), "font": FONT_18, "color": FG},
            {"kind": "text", "text": f"max proc: {max_local_process_ms:.2f}ms", "xy": (760, 325), "font": FONT_18, "color": FG},
            {"kind": "text", "text": f"local lag: {local_lag_ms:.1f}ms", "xy": (760, 350), "font": FONT_18, "color": FG},
            {"kind": "text", "text": f"latest bytes: {latest_audio_bytes}", "xy": (760, 375), "font": FONT_18, "color": FG},
            {"kind": "text", "text": f"recent audio waveform ({self.waveform_seconds:.1f}s window, sr={audio_sr})", "xy": (60, 920), "font": FONT_24, "color": FG},
        ]
        return draw_text_batch_cn(img, text_ops)
