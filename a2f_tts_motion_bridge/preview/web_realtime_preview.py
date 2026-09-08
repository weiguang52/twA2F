from __future__ import annotations

import json
import threading
import time
from collections import deque
from http import server
from typing import Deque, Dict, Optional

import cv2
import numpy as np

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


class WebRealtimeA2FPreview:
    """Headless web preview for realtime A2F.

    - Starts an HTTP server.
    - Renders frames in memory.
    - Exposes:
        /              HTML page
        /stream.mjpg   MJPEG stream
        /snapshot.jpg  latest JPEG frame
        /state         JSON state
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        render_fps: float = 20.0,
        waveform_seconds: float = 6.0,
        transcript_chars: int = 100,
        title: str = "A2F Realtime Web Preview",
        keep_serving_after_stream_end: bool = True,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.render_fps = max(2.0, float(render_fps))
        self.waveform_seconds = max(1.0, float(waveform_seconds))
        self.transcript_chars = max(20, int(transcript_chars))
        self.title = title
        self.keep_serving_after_stream_end = bool(keep_serving_after_stream_end)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._stream_end = False
        self._render_thread: Optional[threading.Thread] = None
        self._server_thread: Optional[threading.Thread] = None
        self._httpd: Optional[_ThreadingHTTPServer] = None

        self._session_id = "-"
        self._server_action_desc = ""
        self._transcript_parts: Deque[str] = deque(maxlen=120)
        self._latest_preview: Optional[Dict[str, object]] = None
        self._latest_audio_sr = 24000
        self._latest_audio_channels = 1
        self._latest_audio_bytes = 0
        self._audio_ring: np.ndarray = np.zeros(0, dtype=np.float32)
        self._latest_audio_latency_s: Optional[float] = None
        self._received_audio_ms = 0.0
        self._server_audio_chunks = 0
        self._local_frames = 0
        self._status_line = "waiting for stream..."
        self._last_frame_bgr = self._build_placeholder_frame("waiting for stream...")
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
                    body = json.dumps(owner.get_state(), ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
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

        self._httpd = _ThreadingHTTPServer((self.host, self.port), Handler)
        self._server_thread = threading.Thread(target=self._httpd.serve_forever, name="A2FWebHTTP", daemon=True)
        self._server_thread.start()

        self._render_thread = threading.Thread(target=self._render_loop, name="A2FWebRender", daemon=True)
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

    def mark_stream_end(self) -> None:
        with self._lock:
            self._stream_end = True
            self._status_line = "stream ended"

    def set_status(self, text: str) -> None:
        with self._lock:
            self._status_line = str(text)

    def update_transcript(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._transcript_parts.append(str(text))

    def update_server_action(self, desc: str) -> None:
        with self._lock:
            self._server_action_desc = str(desc or "")

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
                self._local_frames = int(preview.get("local_frame_count", self._local_frames))
            if sample_rate:
                self._latest_audio_sr = int(sample_rate)
            if channels:
                self._latest_audio_channels = int(channels)
            if first_audio_latency_s is not None:
                self._latest_audio_latency_s = float(first_audio_latency_s)
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
                "status": self._status_line,
                "server_action_desc": self._server_action_desc,
                "transcript": transcript,
                "received_audio_ms": round(self._received_audio_ms, 1),
                "server_audio_chunks": self._server_audio_chunks,
                "local_frames": self._local_frames,
                "latest_audio_bytes": self._latest_audio_bytes,
                "first_audio_latency_s": self._latest_audio_latency_s,
                "current_t": round(_safe_float(preview.get("t"), 0.0), 3),
                "stream_end": self._stream_end,
                "updated_at": self._latest_frame_ts,
            }

    def _render_loop(self) -> None:
        frame_interval = 1.0 / self.render_fps
        while not self._stop_event.is_set():
            t0 = time.time()
            frame = self._build_frame()
            ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if ok:
                with self._lock:
                    self._last_frame_bgr = frame
                    self._latest_jpeg = enc.tobytes()
                    self._latest_frame_ts = time.time()
            with self._lock:
                stream_end = self._stream_end
            if stream_end and not self.keep_serving_after_stream_end:
                self._stop_event.set()
                break
            spent = time.time() - t0
            if spent < frame_interval:
                time.sleep(frame_interval - spent)

    def _build_index_html(self) -> str:
        return f"""<!doctype html>
<html lang=\"zh\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{self.title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #0f1115; color: #eef1f4; }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 16px; }}
    .top {{ display: grid; grid-template-columns: 1fr 360px; gap: 16px; align-items: start; }}
    .card {{ background: #171b22; border: 1px solid #272d37; border-radius: 12px; overflow: hidden; box-shadow: 0 8px 30px rgba(0,0,0,.25); }}
    .imgbox {{ background: #0b0e13; }}
    img {{ display: block; width: 100%; height: auto; }}
    .meta {{ padding: 14px 16px; line-height: 1.55; font-size: 14px; }}
    .label {{ color: #8ea2c1; }}
    .value {{ color: #ffffff; font-weight: 600; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; word-break: break-all; }}
    h1 {{ font-size: 20px; margin: 0 0 14px; }}
    h2 {{ font-size: 15px; margin: 0 0 8px; color: #9fc2ff; }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; color: #f3f6fa; }}
    .foot {{ margin-top: 12px; color: #90a0b7; font-size: 12px; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>{self.title}</h1>
    <div class=\"top\">
      <div class=\"card imgbox\">
        <img id=\"stream\" src=\"/stream.mjpg\" alt=\"A2F realtime stream\" />
      </div>
      <div class=\"card meta\">
        <h2>Live status</h2>
        <div><span class=\"label\">session_id:</span> <span class=\"value mono\" id=\"session_id\">-</span></div>
        <div><span class=\"label\">status:</span> <span class=\"value\" id=\"status\">-</span></div>
        <div><span class=\"label\">current_t:</span> <span class=\"value\" id=\"current_t\">0</span></div>
        <div><span class=\"label\">received_audio_ms:</span> <span class=\"value\" id=\"received_audio_ms\">0</span></div>
        <div><span class=\"label\">server_audio_chunks:</span> <span class=\"value\" id=\"server_audio_chunks\">0</span></div>
        <div><span class=\"label\">local_frames:</span> <span class=\"value\" id=\"local_frames\">0</span></div>
        <div><span class=\"label\">latest_audio_bytes:</span> <span class=\"value\" id=\"latest_audio_bytes\">0</span></div>
        <div><span class=\"label\">first_audio_latency_s:</span> <span class=\"value\" id=\"first_audio_latency_s\">-</span></div>
        <h2 style=\"margin-top:14px\">Transcript</h2>
        <pre id=\"transcript\">-</pre>
        <h2 style=\"margin-top:14px\">Server action</h2>
        <pre id=\"server_action_desc\">-</pre>
      </div>
    </div>
    <div class=\"foot\">This page auto-refreshes metadata every 500ms. MJPEG stream updates continuously.</div>
  </div>
  <script>
    async function refreshState() {{
      try {{
        const resp = await fetch('/state', {{ cache: 'no-store' }});
        const s = await resp.json();
        for (const key of ['session_id','status','current_t','received_audio_ms','server_audio_chunks','local_frames','latest_audio_bytes','transcript','server_action_desc']) {{
          const el = document.getElementById(key);
          if (el) el.textContent = (s[key] ?? '-');
        }}
        const lat = document.getElementById('first_audio_latency_s');
        if (lat) lat.textContent = s.first_audio_latency_s == null ? '-' : s.first_audio_latency_s;
      }} catch (e) {{}}
    }}
    setInterval(refreshState, 500);
    refreshState();
  </script>
</body>
</html>
"""

    def _build_placeholder_frame(self, text: str) -> np.ndarray:
        img = np.full((H, W, 3), BG, dtype=np.uint8)
        cv2.putText(img, self.title, (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, FG, 2, cv2.LINE_AA)
        cv2.putText(img, text, (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ACCENT, 2, cv2.LINE_AA)
        return img

    def _build_frame(self) -> np.ndarray:
        with self._lock:
            preview = dict(self._latest_preview or {})
            transcript = "".join(self._transcript_parts)
            server_action_desc = self._server_action_desc
            waveform = self._audio_ring.copy()
            audio_sr = int(self._latest_audio_sr)
            latest_audio_bytes = int(self._latest_audio_bytes)
            first_audio_latency_s = self._latest_audio_latency_s
            status_line = self._status_line
            session_id = self._session_id
            received_audio_ms = float(self._received_audio_ms)
            server_audio_chunks = int(self._server_audio_chunks)
            local_frames = int(self._local_frames)

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

        draw_energy_bar(img, 60, 90, 180, 18, np.clip(_safe_float(features.get("skin_energy"), 0.0), 0.0, 1.0), "skin")
        draw_energy_bar(img, 60, 130, 180, 18, np.clip(_safe_float(features.get("jaw_energy"), jaw_open), 0.0, 1.0), "jaw")
        draw_energy_bar(img, 60, 170, 180, 18, np.clip(_safe_float(features.get("eyes_energy"), eye_act), 0.0, 1.0), "eyes")
        draw_energy_bar(img, 60, 210, 180, 18, np.clip(_safe_float(features.get("tongue_energy"), mouth_round), 0.0, 1.0), "tongue")

        cv2.putText(img, self.title, (40, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, FG, 2, cv2.LINE_AA)
        cv2.putText(img, f"session: {session_id}", (40, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FG, 1, cv2.LINE_AA)
        cv2.putText(img, f"status: {status_line}", (40, 812), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ACCENT, 2, cv2.LINE_AA)

        current_t = _safe_float(preview.get("t"), 0.0)
        cv2.putText(img, f"local a2f time: {current_t:0.2f}s", (900, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FG, 2, cv2.LINE_AA)
        cv2.putText(img, f"server audio chunks: {server_audio_chunks}", (900, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, FG, 2, cv2.LINE_AA)
        cv2.putText(img, f"received audio: {received_audio_ms/1000.0:0.2f}s", (900, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, FG, 2, cv2.LINE_AA)
        cv2.putText(img, f"local frames: {local_frames}", (900, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.65, FG, 2, cv2.LINE_AA)
        if first_audio_latency_s is not None:
            cv2.putText(img, f"first audio latency: {first_audio_latency_s:0.3f}s", (900, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.65, FG, 2, cv2.LINE_AA)
        cv2.putText(img, f"last audio bytes: {latest_audio_bytes}", (900, 206), cv2.FONT_HERSHEY_SIMPLEX, 0.65, FG, 2, cv2.LINE_AA)

        transcript_text = transcript[-self.transcript_chars:] if transcript else "(waiting transcript...)"
        cv2.rectangle(img, (250, 80), (860, 150), (235, 235, 235), -1)
        cv2.rectangle(img, (250, 80), (860, 150), (210, 210, 210), 1)
        cv2.putText(img, transcript_text, (280, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, FG, 2, cv2.LINE_AA)

        action_text = server_action_desc[-78:] if server_action_desc else "-"
        cv2.rectangle(img, (250, 160), (860, 230), (240, 240, 240), -1)
        cv2.rectangle(img, (250, 160), (860, 230), (210, 210, 210), 1)
        cv2.putText(img, "server action", (270, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.55, FG, 1, cv2.LINE_AA)
        cv2.putText(img, action_text, (270, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.58, FG, 1, cv2.LINE_AA)

        if motor_names and motor_values:
            draw_motor_panel(img, motor_names, motor_values)
        else:
            fallback_names = list(DEFAULT_MOTOR_NEUTRAL.keys())
            fallback_values = [DEFAULT_MOTOR_NEUTRAL[k] for k in fallback_names]
            draw_motor_panel(img, fallback_names, fallback_values)

        self._draw_waveform_panel(img, waveform, audio_sr)
        return img

    def _draw_waveform_panel(self, img: np.ndarray, waveform: np.ndarray, audio_sr: int) -> None:
        wave_x = 70
        wave_y = 900
        wave_w = W - 140
        wave_h = 140

        panel = np.full((wave_h, wave_w, 3), 252, dtype=np.uint8)
        mid_y = wave_h // 2
        cv2.line(panel, (0, mid_y), (wave_w - 1, mid_y), (210, 210, 210), 1, cv2.LINE_AA)

        if len(waveform) > 0:
            peak = float(np.max(np.abs(waveform)))
            if peak > 1e-6:
                waveform = waveform / peak
            samples_per_col = max(1, len(waveform) // wave_w)
            for x in range(wave_w):
                start = x * samples_per_col
                end = min(len(waveform), start + samples_per_col)
                seg = waveform[start:end]
                if len(seg) == 0:
                    continue
                ymin = float(np.min(seg))
                ymax = float(np.max(seg))
                y1 = int(mid_y - ymax * (wave_h * 0.42))
                y2 = int(mid_y - ymin * (wave_h * 0.42))
                cv2.line(panel, (x, y1), (x, y2), WAVE, 1, cv2.LINE_AA)
            cursor_x = wave_w - 2
            cv2.line(panel, (cursor_x, 0), (cursor_x, wave_h - 1), CURSOR, 2, cv2.LINE_AA)

        img[wave_y:wave_y + wave_h, wave_x:wave_x + wave_w] = panel
        cv2.rectangle(img, (wave_x, wave_y), (wave_x + wave_w, wave_y + wave_h), (190, 190, 190), 1)
        cv2.putText(img, f"recent audio waveform ({self.waveform_seconds:.1f}s window, sr={audio_sr})", (wave_x, wave_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, FG, 2, cv2.LINE_AA)
