from __future__ import annotations

import argparse
import time
import uuid
import wave
from typing import Dict, Iterator, Optional, Tuple

import grpc

from ..motion_core.types import AudioChunk, SessionConfig
from ..motion_core.motion_session import MotorStreamSession
from ..protocol.voiceagent_generated import ActionImageMediaStream_pb2 as vapb2
from ..protocol.voiceagent_generated import ActionImageMediaStream_pb2_grpc as vapb2_grpc
from ..preview.web_realtime_preview import WebRealtimeA2FPreview


PCM16 = getattr(vapb2, "PCM16", 0)
TRANSCRIPT = getattr(vapb2, "TRANSCRIPT", 1)
AUDIO = getattr(vapb2, "AUDIO", 0)
ACTION = getattr(vapb2, "ACTION", 2)
ERROR = getattr(vapb2, "ERROR", 3)


def log(msg: str) -> None:
    print(f"[TTSWebRealtime] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def pcm_bytes_to_ms(num_bytes: int, sample_rate: int, channels: int, bits_per_sample: int = 16) -> int:
    bytes_per_sample = max(1, bits_per_sample // 8)
    frame_bytes = max(1, channels * bytes_per_sample)
    frame_count = float(num_bytes) / float(frame_bytes)
    duration_ms = frame_count * 1000.0 / float(max(1, sample_rate))
    return int(round(duration_ms))


def load_wav_pcm16(path: str) -> Tuple[bytes, int, int]:
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise ValueError(f"Only PCM16 wav is supported, got sample_width={sample_width}")
    return frames, sample_rate, channels


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


def build_wav_push_requests(
    pcm_bytes: bytes,
    *,
    session_id: str,
    request_id: str,
    robot_id: str,
    sample_rate: int,
    channels: int,
    chunk_ms: int,
    emotion: str,
    intensity: float,
    extra_meta: Optional[Dict[str, str]] = None,
) -> Iterator[vapb2.AudioRequest]:
    meta_base: Dict[str, str] = {
        "emotion": emotion,
        "intensity": str(intensity),
    }
    if extra_meta:
        meta_base.update(extra_meta)

    bytes_per_ms = max(1, int(sample_rate * channels * 2 / 1000))
    chunk_bytes = max(bytes_per_ms, bytes_per_ms * int(chunk_ms))

    offset = 0
    chunk_index = 0
    while offset < len(pcm_bytes):
        chunk = pcm_bytes[offset: offset + chunk_bytes]
        offset += len(chunk)
        yield vapb2.AudioRequest(
            session_id=session_id,
            robot_id=robot_id,
            request_id=request_id,
            ts=chunk_index * chunk_ms,
            codec=PCM16,
            sample_rate=sample_rate,
            channels=channels,
            audio_chunk=chunk,
            end_of_stream=False,
            meta=meta_base,
        )
        chunk_index += 1

    yield vapb2.AudioRequest(
        session_id=session_id,
        robot_id=robot_id,
        request_id=request_id,
        ts=chunk_index * chunk_ms,
        codec=PCM16,
        sample_rate=sample_rate,
        channels=channels,
        audio_chunk=b"",
        end_of_stream=True,
        meta=meta_base,
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
        self.session.process_chunk(chunk)

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

        return {
            "session_id": self.session.config.session_id,
            "request_id": self.session.config.request_id,
            "t": float(rec.frame_times[idx]),
            "motor_names": list(rec.motor_names),
            "motor_values": list(rec.motor_values[idx]) if rec.motor_values else [],
            "features": features,
            "local_frame_count": len(rec.frame_times),
        }


def consume_tts_stream_web(
    *,
    host: str,
    port: int,
    model_path: str,
    text: Optional[str],
    input_wav: Optional[str],
    robot_id: str,
    emotion: str,
    intensity: float,
    locale: str,
    tts_sample_rate: int,
    tts_channels: int,
    request_chunk_ms: int,
    output_fps: float,
    force_cpu: bool,
    warmup: int,
    web_host: str,
    web_port: int,
    render_fps: float,
    waveform_seconds: float,
    keep_serving: bool,
) -> None:
    session_id = f"tts-web-{uuid.uuid4().hex[:8]}"
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
    viz = WebRealtimeA2FPreview(
        host=web_host,
        port=web_port,
        render_fps=render_fps,
        waveform_seconds=waveform_seconds,
        title="A2F Realtime Web Preview",
        keep_serving_after_stream_end=keep_serving,
    )
    viz.start()

    transcript_parts = []
    first_audio_ts: Optional[float] = None
    start_ts = time.time()

    if text:
        req_iter = build_text_trigger_requests(
            session_id=session_id,
            request_id=request_id,
            robot_id=robot_id,
            text=text,
            emotion=emotion,
            intensity=intensity,
            locale=locale,
        )
        log(f"REQUEST_MODE text session_id={session_id} request_id={request_id}")
        viz.set_status("connected, waiting for remote TTS audio...")
    else:
        pcm_bytes, wav_sr, wav_ch = load_wav_pcm16(input_wav)
        req_iter = build_wav_push_requests(
            pcm_bytes,
            session_id=session_id,
            request_id=request_id,
            robot_id=robot_id,
            sample_rate=wav_sr,
            channels=wav_ch,
            chunk_ms=request_chunk_ms,
            emotion=emotion,
            intensity=intensity,
        )
        log(f"REQUEST_MODE wav session_id={session_id} request_id={request_id} wav={input_wav}")
        viz.set_status("connected, pushing wav upstream...")

    log(f"WEB_PREVIEW url=http://{web_host}:{web_port}")
    target = f"{host}:{port}"
    log(f"CONNECT target={target}")
    channel = grpc.insecure_channel(target)
    stub = vapb2_grpc.VoiceAgentStub(channel)

    try:
        responses = stub.StreamSession(req_iter)
        for resp in responses:
            if resp.type == TRANSCRIPT:
                if resp.transcript:
                    transcript_parts.append(resp.transcript)
                    viz.update_transcript(resp.transcript)
                    viz.set_status("receiving transcript + audio...")
                    log(f"TRANSCRIPT text={resp.transcript}")

            elif resp.type == AUDIO:
                chunk = bytes(resp.audio_chunk or b"")
                if chunk:
                    if first_audio_ts is None:
                        first_audio_ts = time.time()
                        log(f"FIRST_AUDIO latency_s={first_audio_ts - start_ts:.3f}")
                    runner.push_tts_audio(
                        pcm_bytes=chunk,
                        sample_rate=tts_sample_rate,
                        channels=tts_channels,
                    )
                    preview = runner.get_latest_preview()
                    viz.update(
                        preview=preview,
                        pcm16_chunk=chunk,
                        sample_rate=tts_sample_rate,
                        channels=tts_channels,
                        first_audio_latency_s=(first_audio_ts - start_ts) if first_audio_ts is not None else None,
                        status="streaming remote TTS -> local A2F",
                    )
                    log(
                        f"AUDIO bytes={len(chunk)} sr={tts_sample_rate} ch={tts_channels} "
                        f"local_audio_ms={runner.audio_received_ms}"
                    )

            elif resp.type == ACTION:
                desc = resp.action_description or ""
                viz.update_server_action(desc)
                log(f"SERVER_ACTION desc={desc or '-'} bytes={len(resp.action_npz or b'')}")

            elif resp.type == ERROR:
                viz.set_status(f"server error: {resp.error_msg}")
                log(f"SERVER_ERROR msg={resp.error_msg}")

            else:
                log(f"UNKNOWN_RESPONSE type={resp.type}")

            if resp.end_of_response:
                log("END_OF_RESPONSE received=true")
                break

        runner.finalize()
        final_preview = runner.get_latest_preview()
        if final_preview is not None:
            viz.update(preview=final_preview, status="remote stream finished; local A2F finalized")
        viz.mark_stream_end()

        if keep_serving:
            log("stream finished; web preview remains available, press Ctrl+C to exit")
            while True:
                time.sleep(0.5)
        else:
            time.sleep(0.5)

    except KeyboardInterrupt:
        log("interrupted by user")
    except grpc.RpcError as e:
        viz.set_status(f"grpc error: {e.code()} {e.details()}")
        log(f"GRPC_ERROR code={e.code()} details={e.details()}")
        raise
    finally:
        channel.close()
        viz.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headless web consumer: connect to remote VoiceAgent, receive AUDIO, run local A2F, and preview in browser."
    )
    parser.add_argument("--host", default="36.103.236.220")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--model", required=True)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Text prompt sent in meta['text'] to trigger remote TTS.")
    group.add_argument("--input_wav", type=str, help="Optional local PCM16 wav to push upstream.")

    parser.add_argument("--robot_id", default="a2f_web_consumer")
    parser.add_argument("--emotion", default="neutral")
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--locale", default="zh-CN")

    parser.add_argument("--tts_sample_rate", type=int, default=24000)
    parser.add_argument("--tts_channels", type=int, default=1)
    parser.add_argument("--request_chunk_ms", type=int, default=200)
    parser.add_argument("--output_fps", type=float, default=30.0)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)

    parser.add_argument("--web_host", default="0.0.0.0")
    parser.add_argument("--web_port", type=int, default=8765)
    parser.add_argument("--render_fps", type=float, default=20.0)
    parser.add_argument("--waveform_seconds", type=float, default=6.0)
    parser.add_argument("--no_keep_serving", action="store_true", help="Exit automatically after stream ends.")

    args = parser.parse_args()

    consume_tts_stream_web(
        host=args.host,
        port=args.port,
        model_path=args.model,
        text=args.text,
        input_wav=args.input_wav,
        robot_id=args.robot_id,
        emotion=args.emotion,
        intensity=args.intensity,
        locale=args.locale,
        tts_sample_rate=args.tts_sample_rate,
        tts_channels=args.tts_channels,
        request_chunk_ms=args.request_chunk_ms,
        output_fps=args.output_fps,
        force_cpu=args.force_cpu,
        warmup=args.warmup,
        web_host=args.web_host,
        web_port=args.web_port,
        render_fps=args.render_fps,
        waveform_seconds=args.waveform_seconds,
        keep_serving=not args.no_keep_serving,
    )


if __name__ == "__main__":
    main()
