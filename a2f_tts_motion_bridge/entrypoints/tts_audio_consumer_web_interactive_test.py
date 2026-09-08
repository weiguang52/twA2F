from __future__ import annotations

import argparse
import time
import uuid
from typing import Dict, Iterator, Optional

import grpc

from ..motion_core.types import AudioChunk, SessionConfig
from ..motion_core.motion_session import MotorStreamSession
from ..protocol.voiceagent_generated import ActionImageMediaStream_pb2 as vapb2
from ..protocol.voiceagent_generated import ActionImageMediaStream_pb2_grpc as vapb2_grpc
from ..preview.web_realtime_preview_interactive_test import InteractiveWebRealtimeA2FPreview

PCM16 = getattr(vapb2, "PCM16", 0)
TRANSCRIPT = getattr(vapb2, "TRANSCRIPT", 1)
AUDIO = getattr(vapb2, "AUDIO", 0)
ACTION = getattr(vapb2, "ACTION", 2)
ERROR = getattr(vapb2, "ERROR", 3)


def log(msg: str) -> None:
    print(f"[TTSWebInteractiveTest] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def pcm_bytes_to_ms(num_bytes: int, sample_rate: int, channels: int, bits_per_sample: int = 16) -> int:
    bytes_per_sample = max(1, bits_per_sample // 8)
    frame_bytes = max(1, channels * bytes_per_sample)
    frame_count = float(num_bytes) / float(frame_bytes)
    duration_ms = frame_count * 1000.0 / float(max(1, sample_rate))
    return int(round(duration_ms))


def build_text_trigger_requests(
    *,
    session_id: str,
    request_id: str,
    robot_id: str,
    text: str,
    emotion: str,
    intensity: float,
    locale: str,
    extra_meta: Optional[Dict[str, str]] = None,
) -> Iterator[vapb2.AudioRequest]:
    meta: Dict[str, str] = {
        "text": text,
        "emotion": emotion,
        "intensity": str(intensity),
    }
    if locale:
        meta["locale"] = locale
    if extra_meta:
        meta.update(extra_meta)

    yield vapb2.AudioRequest(
        session_id=session_id,
        robot_id=robot_id,
        request_id=request_id,
        ts=int(time.time() * 1000),
        codec=PCM16,
        sample_rate=0,
        channels=0,
        audio_chunk=b"",
        end_of_stream=True,
        meta=meta,
    )


class LocalA2FRealtimeRunner:
    def __init__(
        self,
        *,
        model_path: str,
        session_id: str,
        request_id: str,
        emotion: str,
        intensity: float,
        output_fps: float,
        force_cpu: bool,
        warmup: int,
    ) -> None:
        cfg = SessionConfig(
            session_id=session_id,
            emotion=emotion,
            intensity=float(intensity),
            output_fps=float(output_fps),
            save_npy=False,
            npy_save_dir=None,
            request_id=request_id,
            extra={},
        )
        self.session = MotorStreamSession(
            model_path=model_path,
            config=cfg,
            force_cpu=force_cpu,
            warmup=warmup,
        )
        self.sample_rate: Optional[int] = None
        self.channels: Optional[int] = None
        self.chunk_id = 0
        self.audio_received_ms = 0

        self.last_local_process_ms = 0.0
        self.total_local_process_ms = 0.0
        self.local_process_calls = 0
        self.max_local_process_ms = 0.0
        self.first_local_frame_ts: Optional[float] = None

    def push_tts_audio(self, pcm_bytes: bytes, sample_rate: int, channels: int) -> None:
        if not pcm_bytes:
            return
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        duration_ms = pcm_bytes_to_ms(
            len(pcm_bytes),
            sample_rate=sample_rate,
            channels=channels,
            bits_per_sample=16,
        )
        self.audio_received_ms += duration_ms
        chunk = AudioChunk(
            session_id=self.session.config.session_id,
            chunk_id=self.chunk_id,
            pts_ms=self.audio_received_ms,
            pcm_bytes=pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
            bits_per_sample=16,
            emotion=self.session.config.emotion,
            intensity=self.session.config.intensity,
        )
        self.chunk_id += 1

        rec = self.session.recorder
        before_frames = len(rec.frame_times)
        t0 = time.perf_counter()
        self.session.process_chunk(chunk)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        self.last_local_process_ms = dt_ms
        self.total_local_process_ms += dt_ms
        self.local_process_calls += 1
        self.max_local_process_ms = max(self.max_local_process_ms, dt_ms)

        after_frames = len(rec.frame_times)
        if self.first_local_frame_ts is None and after_frames > before_frames:
            self.first_local_frame_ts = time.time()

    def finalize(self) -> None:
        self.session.finalize()

    def get_latest_preview(self) -> Optional[Dict[str, object]]:
        rec = self.session.recorder
        if not rec.frame_times:
            return None

        idx = len(rec.frame_times) - 1
        feature_row = list(rec.features[idx]) if rec.features else []
        feature_names = [
            "jaw_open",
            "mouth_round",
            "mouth_wide",
            "mouth_left_right",
            "upper_face_activity",
            "eye_activity",
            "blink_like",
        ]
        features: Dict[str, float] = {
            name: float(feature_row[i]) if i < len(feature_row) else 0.0
            for i, name in enumerate(feature_names)
        }
        features["audio_rms"] = float(rec.audio_rms[idx]) if rec.audio_rms else 0.0
        features["speech_gate"] = float(rec.speech_gate[idx]) if rec.speech_gate else 0.0
        features["skin_energy"] = float(features["upper_face_activity"])
        features["jaw_energy"] = float(features["jaw_open"])
        features["eyes_energy"] = float(features["eye_activity"])
        features["tongue_energy"] = float(features["mouth_round"])

        current_t = float(rec.frame_times[idx])
        local_lag_ms = max(0.0, self.audio_received_ms - 1000.0 * current_t)
        avg_local_process_ms = (
            self.total_local_process_ms / self.local_process_calls if self.local_process_calls > 0 else 0.0
        )

        return {
            "session_id": self.session.config.session_id,
            "request_id": self.session.config.request_id,
            "t": current_t,
            "motor_names": list(rec.motor_names),
            "motor_values": list(rec.motor_values[idx]) if rec.motor_values else [],
            "features": features,
            "local_frame_count": len(rec.frame_times),
            "last_local_process_ms": float(self.last_local_process_ms),
            "avg_local_process_ms": float(avg_local_process_ms),
            "max_local_process_ms": float(self.max_local_process_ms),
            "local_lag_ms": float(local_lag_ms),
        }


def run_single_text_round(
    *,
    host: str,
    port: int,
    model_path: str,
    text: str,
    robot_id: str,
    emotion: str,
    intensity: float,
    locale: str,
    tts_sample_rate: int,
    tts_channels: int,
    output_fps: float,
    force_cpu: bool,
    warmup: int,
    preview: InteractiveWebRealtimeA2FPreview,
) -> None:
    session_id = f"tts-live-{uuid.uuid4().hex[:8]}"
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    runner = LocalA2FRealtimeRunner(
        model_path=model_path,
        session_id=session_id,
        request_id=request_id,
        emotion=emotion,
        intensity=intensity,
        output_fps=output_fps,
        force_cpu=force_cpu,
        warmup=warmup,
    )

    req_iter = build_text_trigger_requests(
        session_id=session_id,
        request_id=request_id,
        robot_id=robot_id,
        text=text,
        emotion=emotion,
        intensity=intensity,
        locale=locale,
        extra_meta=None,
    )

    preview.start_round(session_id=session_id, request_id=request_id, text=text)
    log(f"ROUND_START session_id={session_id} request_id={request_id} text={text}")

    target = f"{host}:{port}"
    channel = grpc.insecure_channel(target)
    stub = vapb2_grpc.VoiceAgentStub(channel)
    start_ts = time.time()
    first_audio_ts: Optional[float] = None
    first_local_frame_logged = False

    try:
        responses = stub.StreamSession(req_iter)
        for resp in responses:
            if resp.type == TRANSCRIPT:
                if resp.transcript:
                    log(f"TRANSCRIPT text={resp.transcript}")
                    preview.update(transcript=resp.transcript, status="receiving transcript...")

            elif resp.type == AUDIO:
                chunk = bytes(resp.audio_chunk or b"")
                if chunk:
                    if first_audio_ts is None:
                        first_audio_ts = time.time()
                        latency = first_audio_ts - start_ts
                        log(f"FIRST_AUDIO latency_s={latency:.3f}")

                    runner.push_tts_audio(chunk, tts_sample_rate, tts_channels)
                    p = runner.get_latest_preview()

                    if (not first_local_frame_logged) and runner.first_local_frame_ts is not None:
                        first_local_frame_logged = True
                        log(f"FIRST_LOCAL_FRAME latency_s={(runner.first_local_frame_ts - start_ts):.3f}")

                    avg_local_ms = (
                        runner.total_local_process_ms / runner.local_process_calls
                        if runner.local_process_calls
                        else 0.0
                    )
                    log(
                        f"AUDIO bytes={len(chunk)} sr={tts_sample_rate} ch={tts_channels} "
                        f"local_audio_ms={runner.audio_received_ms} "
                        f"last_local_process_ms={runner.last_local_process_ms:.2f} "
                        f"avg_local_process_ms={avg_local_ms:.2f}"
                    )
                    preview.update(
                        preview=p,
                        pcm16_chunk=chunk,
                        sample_rate=tts_sample_rate,
                        channels=tts_channels,
                        first_audio_latency_s=(first_audio_ts - start_ts) if first_audio_ts else None,
                        first_local_frame_latency_s=(runner.first_local_frame_ts - start_ts)
                        if runner.first_local_frame_ts
                        else None,
                        status="receiving audio...",
                    )

            elif resp.type == ACTION:
                desc = str(resp.action_description or "")
                log(f"SERVER_ACTION desc={desc} bytes={len(resp.action_npz or b'')}")
                preview.update(server_action_desc=desc, status="receiving server action...")

            elif resp.type == ERROR:
                msg = str(resp.error_msg or "unknown server error")
                log(f"SERVER_ERROR msg={msg}")
                preview.set_error(msg)

            else:
                log(f"UNKNOWN_RESPONSE type={resp.type}")

            if resp.end_of_response:
                log("END_OF_RESPONSE received=true")
                break

        runner.finalize()
        p = runner.get_latest_preview()
        preview.update(preview=p, status="round finished")
        preview.finish_round(status="round finished")
        log(
            f"ROUND_DONE session_id={session_id} request_id={request_id} "
            f"audio_ms={runner.audio_received_ms} final_local_lag_ms={(p.get('local_lag_ms', 0.0) if p else 0.0):.1f}"
        )

    except grpc.RpcError as e:
        msg = f"gRPC {e.code()}: {e.details()}"
        log(f"GRPC_ERROR {msg}")
        preview.set_error(msg)
    except Exception as e:
        msg = f"local exception: {e}"
        log(f"LOCAL_ERROR {msg}")
        preview.set_error(msg)
        raise
    finally:
        channel.close()


def run_interactive_loop(
    *,
    host: str,
    port: int,
    model_path: str,
    robot_id: str,
    emotion: str,
    intensity: float,
    locale: str,
    tts_sample_rate: int,
    tts_channels: int,
    output_fps: float,
    force_cpu: bool,
    warmup: int,
    web_host: str,
    web_port: int,
    render_fps: float,
) -> None:
    preview = InteractiveWebRealtimeA2FPreview(
        host=web_host,
        port=web_port,
        render_fps=render_fps,
        title="A2F Interactive Web Preview",
    )
    preview.start()
    log(f"WEB_READY url={preview.url}")
    log("waiting for browser text input...")

    try:
        while True:
            text = preview.get_next_text(timeout=0.5)
            if text is None:
                continue
            print(f"[WEB] dequeued text = {text!r}", flush=True)
            run_single_text_round(
                host=host,
                port=port,
                model_path=model_path,
                text=text,
                robot_id=robot_id,
                emotion=emotion,
                intensity=intensity,
                locale=locale,
                tts_sample_rate=tts_sample_rate,
                tts_channels=tts_channels,
                output_fps=output_fps,
                force_cpu=force_cpu,
                warmup=warmup,
                preview=preview,
            )
            log(f"idle; pending_queue={preview.pending_count}")
    except KeyboardInterrupt:
        log("interrupted by user")
    finally:
        preview.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive web input + continuous multi-round TTS stream consumer + local realtime A2F preview (test version)."
    )
    parser.add_argument("--host", default="36.103.236.220")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--model", required=True)
    parser.add_argument("--robot_id", default="a2f_interactive_web")
    parser.add_argument("--emotion", default="neutral")
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--locale", default="zh-CN")
    parser.add_argument("--tts_sample_rate", type=int, default=24000)
    parser.add_argument("--tts_channels", type=int, default=1)
    parser.add_argument("--output_fps", type=float, default=30.0)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--web_host", default="0.0.0.0")
    parser.add_argument("--web_port", type=int, default=8765)
    parser.add_argument("--render_fps", type=float, default=30.0)
    args = parser.parse_args()

    run_interactive_loop(
        host=args.host,
        port=args.port,
        model_path=args.model,
        robot_id=args.robot_id,
        emotion=args.emotion,
        intensity=args.intensity,
        locale=args.locale,
        tts_sample_rate=args.tts_sample_rate,
        tts_channels=args.tts_channels,
        output_fps=args.output_fps,
        force_cpu=args.force_cpu,
        warmup=args.warmup,
        web_host=args.web_host,
        web_port=args.web_port,
        render_fps=args.render_fps,
    )


if __name__ == "__main__":
    main()
