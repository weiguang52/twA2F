from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional, Set

import cv2
import numpy as np

try:
    from aiohttp import web, WSMsgType
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "web_realtime_preview_test.py requires aiohttp. Please install it in your env first: pip install aiohttp"
    ) from e


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


class WebRealtimeA2FPreview:
    """WebSocket-based realtime preview for tts_audio_consumer_web_test.

    Why this version:
    - Keeps the existing sync TTS/A2F pipeline unchanged.
    - Reuses locally rendered JPEG frames.
    - Pushes frames + state to the browser over WebSocket.
    - Avoids MJPEG and high-frequency HTTP polling, which were brittle in the browser.

    Exposes:
      /           HTML page
      /ws         WebSocket (text JSON for state, binary JPEG for frames)
      /state      Debug JSON endpoint
      /snapshot.jpg  Debug latest JPEG endpoint
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        render_fps: float = 30.0,
        waveform_seconds: float = 6.0,
        transcript_chars: int = 100,
        title: str = "A2F Realtime Web Preview (WebSocket Test)",
        keep_serving_after_stream_end: bool = True,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.render_fps = max(2.0, float(render_fps))
        self.waveform_seconds = max(1.0, float(waveform_seconds))
        self.transcript_chars = max(20, int(transcript_chars))
        self.title = str(title)
        self.keep_serving_after_stream_end = bool(keep_serving_after_stream_end)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._render_thread: Optional[threading.Thread] = None
        self._server_thread: Optional[threading.Thread] = None

        self._session_id = "-"
        self._request_id = "-"
        self._input_text = ""
        self._status_line = "waiting for stream..."
        self._server_action_desc = ""
        self._transcript_parts: Deque[str] = deque(maxlen=120)
        self._latest_preview: Optional[Dict[str, object]] = None

        self._latest_audio_sr = 24000
        self._latest_audio_channels = 1
        self._latest_audio_bytes = 0
        self._latest_audio_latency_s: Optional[float] = None
        self._first_local_frame_latency_s: Optional[float] = None
        self._received_audio_ms = 0.0
        self._server_audio_chunks = 0
        self._local_frames = 0
        self._stream_end = False
        self._tick = 0

        self._last_frame_bgr = self._build_placeholder_frame("booting...")
        ok, enc = cv2.imencode(".jpg", self._last_frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        self._latest_jpeg = enc.tobytes() if ok else b""
        self._latest_frame_ts = time.time()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._clients: Set[web.WebSocketResponse] = set()

    @property
    def url(self) -> str:
        host = self.host if self.host != "0.0.0.0" else "127.0.0.1"
        return f"http://{host}:{self.port}"

    def start(self) -> None:
        if self._server_thread is not None:
            return

        self._server_thread = threading.Thread(target=self._server_main, name="A2FWSHTTP", daemon=True)
        self._server_thread.start()

        # wait briefly for loop/app startup
        t0 = time.time()
        while self._loop is None and time.time() - t0 < 5.0:
            time.sleep(0.02)

        self._render_thread = threading.Thread(target=self._render_loop, name="A2FWSRender", daemon=True)
        self._render_thread.start()

    def close(self) -> None:
        self._stop_event.set()

        if self._loop is not None:
            fut = asyncio.run_coroutine_threadsafe(self._shutdown_async(), self._loop)
            try:
                fut.result(timeout=3.0)
            except Exception:
                pass

        if self._server_thread is not None:
            self._server_thread.join(timeout=3.0)
            self._server_thread = None

        if self._render_thread is not None:
            self._render_thread.join(timeout=2.0)
            self._render_thread = None

    # ---------- entrypoint compatibility ----------
    def start_round(self, session_id: str, request_id: str, text: str = "") -> None:
        with self._lock:
            self._session_id = str(session_id or "-")
            self._request_id = str(request_id or "-")
            self._input_text = str(text or "")
            self._status_line = "round started"
            self._server_action_desc = ""
            self._transcript_parts.clear()
            self._latest_preview = None
            self._latest_audio_latency_s = None
            self._first_local_frame_latency_s = None
            self._received_audio_ms = 0.0
            self._server_audio_chunks = 0
            self._local_frames = 0
            self._latest_audio_bytes = 0
            self._stream_end = False
        self._schedule_state_push()

    def finish_round(self, status: str = "round finished") -> None:
        with self._lock:
            self._status_line = str(status)
            self._stream_end = True
        self._schedule_state_push()

    def mark_stream_end(self) -> None:
        with self._lock:
            self._stream_end = True
            self._status_line = "stream ended"
        self._schedule_state_push()

    def set_status(self, text: str) -> None:
        with self._lock:
            self._status_line = str(text)
        self._schedule_state_push()

    def update_transcript(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._transcript_parts.append(str(text))
        self._schedule_state_push()

    def update_server_action(self, desc: str) -> None:
        with self._lock:
            self._server_action_desc = str(desc or "")
        self._schedule_state_push()

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
        **kwargs,
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
                self._received_audio_ms += (len(chunk_arr) / float(sr)) * 1000.0
        self._schedule_state_push()

    # ---------- state ----------
    def get_state(self) -> Dict[str, object]:
        with self._lock:
            transcript = "".join(self._transcript_parts)
            preview = dict(self._latest_preview or {})
            return {
                "type": "state",
                "session_id": self._session_id,
                "request_id": self._request_id,
                "input_text": self._input_text,
                "status": self._status_line,
                "server_action_desc": self._server_action_desc,
                "transcript": transcript,
                "received_audio_ms": round(self._received_audio_ms, 1),
                "server_audio_chunks": self._server_audio_chunks,
                "local_frames": self._local_frames,
                "latest_audio_bytes": self._latest_audio_bytes,
                "first_audio_latency_s": self._latest_audio_latency_s,
                "first_local_frame_latency_s": self._first_local_frame_latency_s,
                "current_t": round(_safe_float(preview.get("t"), 0.0), 3),
                "stream_end": self._stream_end,
                "tick": self._tick,
                "updated_at": self._latest_frame_ts,
            }

    def get_latest_jpeg(self) -> bytes:
        with self._lock:
            return bytes(self._latest_jpeg)

    # ---------- aiohttp ----------
    async def _handle_index(self, request: web.Request) -> web.Response:
        return web.Response(text=self._build_index_html(), content_type="text/html")

    async def _handle_state(self, request: web.Request) -> web.Response:
        return web.json_response(self.get_state())

    async def _handle_snapshot(self, request: web.Request) -> web.Response:
        return web.Response(body=self.get_latest_jpeg(), content_type="image/jpeg")

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024)
        await ws.prepare(request)
        self._clients.add(ws)

        # send initial state + initial frame
        try:
            await ws.send_str(json.dumps(self.get_state(), ensure_ascii=False))
            await ws.send_bytes(self.get_latest_jpeg())

            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    # currently output-only; keep ping/hello path for future interactive submit
                    txt = (msg.data or "").strip()
                    if txt == "ping":
                        await ws.send_str('{"type":"pong"}')
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(ws)
        return ws

    async def _broadcast_state(self) -> None:
        if not self._clients:
            return
        payload = json.dumps(self.get_state(), ensure_ascii=False)
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_str(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def _broadcast_frame(self, jpeg: bytes) -> None:
        if not self._clients:
            return
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_bytes(jpeg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def _shutdown_async(self) -> None:
        try:
            for ws in list(self._clients):
                try:
                    await ws.close()
                except Exception:
                    pass
            self._clients.clear()
            if self._runner is not None:
                await self._runner.cleanup()
                self._runner = None
        finally:
            loop = asyncio.get_running_loop()
            loop.stop()

    def _schedule_state_push(self) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast_state(), self._loop)
        except Exception:
            pass

    def _schedule_frame_push(self, jpeg: bytes) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast_frame(jpeg), self._loop)
        except Exception:
            pass

    def _server_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        async def _start() -> None:
            self._app = web.Application()
            self._app.router.add_get("/", self._handle_index)
            self._app.router.add_get("/index.html", self._handle_index)
            self._app.router.add_get("/state", self._handle_state)
            self._app.router.add_get("/snapshot.jpg", self._handle_snapshot)
            self._app.router.add_get("/ws", self._handle_ws)

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, host=self.host, port=self.port)
            await self._site.start()

        loop.run_until_complete(_start())
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                try:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
            loop.close()
            self._loop = None

    # ---------- rendering ----------
    def _render_loop(self) -> None:
        frame_interval = 1.0 / self.render_fps
        while not self._stop_event.is_set():
            t0 = time.time()
            frame = self._build_frame()
            # shrink modestly to reduce wire size while keeping clarity
            display = cv2.resize(frame, None, fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 76])
            if ok:
                jpeg = enc.tobytes()
                with self._lock:
                    self._last_frame_bgr = frame
                    self._latest_jpeg = jpeg
                    self._latest_frame_ts = time.time()
                    self._tick += 1
                    stream_end = self._stream_end
                self._schedule_frame_push(jpeg)
            else:
                with self._lock:
                    stream_end = self._stream_end

            if stream_end and not self.keep_serving_after_stream_end:
                self._stop_event.set()
                break

            spent = time.time() - t0
            if spent < frame_interval:
                time.sleep(frame_interval - spent)

    def _build_index_html(self) -> str:
        # WebSocket binary frames -> blob url -> <img>. JSON messages update side panel.
        return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{self.title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #0f1115; color: #eef1f4; }}
    .wrap {{ max-width: 1480px; margin: 0 auto; padding: 16px; }}
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
    .ok {{ color: #7ee787; }}
    .bad {{ color: #ff7b72; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{self.title}</h1>
    <div class="top">
      <div class="card imgbox">
        <img id="stream" src="/snapshot.jpg" alt="A2F realtime stream" />
      </div>
      <div class="card meta">
        <h2>WebSocket status</h2>
        <div><span class="label">ws:</span> <span class="value" id="ws_status">connecting</span></div>
        <div><span class="label">last_frame_bytes:</span> <span class="value" id="last_frame_bytes">0</span></div>
        <div><span class="label">frame_recv_count:</span> <span class="value" id="frame_recv_count">0</span></div>
        <h2 style="margin-top:14px">Live status</h2>
        <div><span class="label">session_id:</span> <span class="value mono" id="session_id">-</span></div>
        <div><span class="label">request_id:</span> <span class="value mono" id="request_id">-</span></div>
        <div><span class="label">status:</span> <span class="value" id="status">-</span></div>
        <div><span class="label">current_t:</span> <span class="value" id="current_t">0</span></div>
        <div><span class="label">received_audio_ms:</span> <span class="value" id="received_audio_ms">0</span></div>
        <div><span class="label">server_audio_chunks:</span> <span class="value" id="server_audio_chunks">0</span></div>
        <div><span class="label">local_frames:</span> <span class="value" id="local_frames">0</span></div>
        <div><span class="label">latest_audio_bytes:</span> <span class="value" id="latest_audio_bytes">0</span></div>
        <div><span class="label">first_audio_latency_s:</span> <span class="value" id="first_audio_latency_s">-</span></div>
        <div><span class="label">first_local_frame_latency_s:</span> <span class="value" id="first_local_frame_latency_s">-</span></div>
        <div><span class="label">tick:</span> <span class="value" id="tick">0</span></div>
        <h2 style="margin-top:14px">Input text</h2>
        <pre id="input_text">-</pre>
        <h2 style="margin-top:14px">Transcript</h2>
        <pre id="transcript">-</pre>
        <h2 style="margin-top:14px">Server action</h2>
        <pre id="server_action_desc">-</pre>
      </div>
    </div>
    <div class="foot">Frames and state are pushed over WebSocket. No MJPEG. No snapshot polling loop.</div>
  </div>
  <script>
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = proto + '://' + location.host + '/ws';
    let ws = null;
    let frameCount = 0;
    let lastUrl = null;

    function setWsStatus(text, good) {{
      const el = document.getElementById('ws_status');
      if (!el) return;
      el.textContent = text;
      el.className = good ? 'value ok' : 'value bad';
    }}

    function applyState(s) {{
      for (const key of [
        'session_id','request_id','status','current_t','received_audio_ms','server_audio_chunks',
        'local_frames','latest_audio_bytes','transcript','server_action_desc','input_text','tick'
      ]) {{
        const el = document.getElementById(key);
        if (el) el.textContent = (s[key] ?? '-');
      }}
      const a = document.getElementById('first_audio_latency_s');
      if (a) a.textContent = s.first_audio_latency_s == null ? '-' : s.first_audio_latency_s;
      const b = document.getElementById('first_local_frame_latency_s');
      if (b) b.textContent = s.first_local_frame_latency_s == null ? '-' : s.first_local_frame_latency_s;
    }}

    function connect() {{
      ws = new WebSocket(wsUrl);
      ws.binaryType = 'blob';

      ws.onopen = () => {{
        setWsStatus('connected', true);
        ws.send('ping');
      }};

      ws.onmessage = async (event) => {{
        if (typeof event.data === 'string') {{
          try {{
            const msg = JSON.parse(event.data);
            if (msg.type === 'state') applyState(msg);
          }} catch (e) {{}}
          return;
        }}

        const blob = event.data;
        frameCount += 1;
        const frameBytesEl = document.getElementById('last_frame_bytes');
        if (frameBytesEl) frameBytesEl.textContent = blob.size;
        const frameCountEl = document.getElementById('frame_recv_count');
        if (frameCountEl) frameCountEl.textContent = frameCount;

        const url = URL.createObjectURL(blob);
        const img = document.getElementById('stream');
        img.onload = () => {{
          if (lastUrl) URL.revokeObjectURL(lastUrl);
          lastUrl = url;
        }};
        img.src = url;
      }};

      ws.onclose = () => {{
        setWsStatus('reconnecting...', false);
        window.setTimeout(connect, 1000);
      }};

      ws.onerror = () => {{
        setWsStatus('error', false);
      }};
    }}

    connect();
  </script>
</body>
</html>
"""

    def _build_placeholder_frame(self, text: str) -> np.ndarray:
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img[:] = (15, 19, 26)
        cv2.putText(img, self.title, (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 245, 250), 2, cv2.LINE_AA)
        cv2.putText(img, text, (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 180, 255), 2, cv2.LINE_AA)
        return img

    def _build_frame(self) -> np.ndarray:
        with self._lock:
            preview = dict(self._latest_preview or {})
            features = preview.get("features") or {}
            transcript = "".join(self._transcript_parts)
            server_action_desc = self._server_action_desc
            session_id = self._session_id
            request_id = self._request_id
            status_line = self._status_line
            current_t = _safe_float(preview.get("t"), 0.0)
            local_frames = self._local_frames
            received_audio_ms = self._received_audio_ms
            first_audio_latency_s = self._latest_audio_latency_s
            first_local_frame_latency_s = self._first_local_frame_latency_s
            tick = self._tick

        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img[:] = (15, 19, 26)

        # simple expressive face from features only
        jaw_open = np.clip(_safe_float(features.get("jaw_open"), 0.0), 0.0, 1.0)
        mouth_wide = np.clip(_safe_float(features.get("mouth_wide"), 0.0), 0.0, 1.0)
        mouth_round = np.clip(_safe_float(features.get("mouth_round"), 0.0), 0.0, 1.0)
        mouth_lr = np.clip(_safe_float(features.get("mouth_left_right"), 0.0), -1.0, 1.0)
        upper_face = np.clip(_safe_float(features.get("upper_face_activity"), 0.0), 0.0, 1.0)
        blink = np.clip(_safe_float(features.get("blink_like"), 0.0), 0.0, 1.0)
        eye_act = np.clip(_safe_float(features.get("eye_activity"), 0.0), 0.0, 1.0)

        cx, cy = 460, 360
        cv2.ellipse(img, (cx, cy), (220, 270), 0, 0, 360, (215, 225, 235), 2, cv2.LINE_AA)

        # brows
        brow_y = 250 - int(28 * upper_face)
        brow_tilt = int(16 * mouth_lr)
        cv2.line(img, (cx - 125, brow_y + brow_tilt), (cx - 55, brow_y - brow_tilt), (215, 225, 235), 4, cv2.LINE_AA)
        cv2.line(img, (cx + 55, brow_y - brow_tilt), (cx + 125, brow_y + brow_tilt), (215, 225, 235), 4, cv2.LINE_AA)

        # eyes
        eye_open = max(3, int(22 * (1.0 - 0.9 * blink + 0.15 * eye_act)))
        cv2.ellipse(img, (cx - 90, 335), (42, eye_open), 0, 0, 360, (215, 225, 235), 2, cv2.LINE_AA)
        cv2.ellipse(img, (cx + 90, 335), (42, eye_open), 0, 0, 360, (215, 225, 235), 2, cv2.LINE_AA)
        cv2.circle(img, (cx - 90, 335), 8, (90, 180, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (cx + 90, 335), 8, (90, 180, 255), -1, cv2.LINE_AA)

        # mouth
        mouth_cx = cx + int(28 * mouth_lr)
        mouth_w = 60 + int(55 * mouth_wide) + int(25 * mouth_round)
        mouth_h = 12 + int(60 * jaw_open) + int(18 * mouth_round)
        cv2.ellipse(img, (mouth_cx, 470), (mouth_w, mouth_h), 0, 0, 360, (90, 180, 255), 3, cv2.LINE_AA)

        # animated cursor bar proves the page is updating even before speech
        bar_x = 80 + int((tick * 11) % 420)
        cv2.rectangle(img, (80, 620), (560, 650), (60, 70, 85), 2)
        cv2.rectangle(img, (bar_x, 623), (bar_x + 80, 647), (90, 180, 255), -1)

        def put(y: int, text: str, scale: float = 0.62, color=(220, 230, 240), thick: int = 2) -> None:
            cv2.putText(img, text[:118], (700, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

        put(60, self.title, 0.95, (240, 245, 250), 2)
        put(105, f"session: {session_id}", 0.55)
        put(140, f"request: {request_id}", 0.55)
        put(180, f"status: {status_line}", 0.58, (90, 180, 255), 2)
        put(220, f"tick: {tick}", 0.6)
        put(255, f"current_t: {current_t:.3f}s", 0.6)
        put(290, f"local_frames: {local_frames}", 0.6)
        put(325, f"received_audio_ms: {received_audio_ms:.1f}", 0.6)
        put(360, f"first_audio_latency_s: {'-' if first_audio_latency_s is None else f'{first_audio_latency_s:.3f}s'}", 0.55)
        put(395, f"first_local_frame_latency_s: {'-' if first_local_frame_latency_s is None else f'{first_local_frame_latency_s:.3f}s'}", 0.55)
        put(450, f"transcript: {transcript}", 0.5)
        put(505, f"server_action: {server_action_desc}", 0.47)

        return img
