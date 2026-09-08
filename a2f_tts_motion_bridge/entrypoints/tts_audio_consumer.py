import argparse
import os
import time
import uuid
import wave
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import grpc

from ..outputs.npy_segment_exporter import ActionSegment, RollingNpySegmentExporter
from ..motion_core.types import AudioChunk, SessionConfig
from ..motion_core.motion_session import MotorStreamSession
from ..protocol.voiceagent_generated import ActionImageMediaStream_pb2 as vapb2
from ..protocol.voiceagent_generated import ActionImageMediaStream_pb2_grpc as vapb2_grpc


PCM16 = getattr(vapb2, "PCM16", 0)
OPUS = getattr(vapb2, "OPUS", 1)

AUDIO = getattr(vapb2, "AUDIO", 0)
TRANSCRIPT = getattr(vapb2, "TRANSCRIPT", 1)
ACTION = getattr(vapb2, "ACTION", 2)
ERROR = getattr(vapb2, "ERROR", 3)


def log(msg: str) -> None:
    print(f"[TTSConsumer] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


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


def save_wav_pcm16(path: str, pcm_bytes: bytes, sample_rate: int, channels: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


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

    # 这里沿用你给的 dist 思路：text 放在 meta 里，直接 end_of_stream=True
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


class LocalA2FRunner:
    def __init__(
        self,
        *,
        model_path: str,
        session_id: str,
        request_id: str,
        emotion: str,
        intensity: float,
        output_fps: float,
        segment_ms: int,
        save_dir: str,
        force_cpu: bool,
        warmup: int,
    ):
        self.save_dir = save_dir
        self.segment_dir = os.path.join(save_dir, "local_action_segments")
        self.server_action_dir = os.path.join(save_dir, "server_action_segments")
        os.makedirs(self.segment_dir, exist_ok=True)
        os.makedirs(self.server_action_dir, exist_ok=True)

        cfg = SessionConfig(
            session_id=session_id,
            emotion=emotion,
            intensity=float(intensity),
            output_fps=float(output_fps),
            save_npy=True,
            npy_save_dir=save_dir,
            request_id=request_id,
            extra={},
        )
        self.session = MotorStreamSession(
            model_path=model_path,
            config=cfg,
            force_cpu=force_cpu,
            warmup=warmup,
        )
        self.exporter = RollingNpySegmentExporter(
            self.session,
            segment_ms=int(segment_ms),
            emit_cover_slack_ms=80,
        )
        self.sample_rate: Optional[int] = None
        self.channels: Optional[int] = None
        self.chunk_id = 0
        self.audio_received_ms = 0
        self.local_segment_count = 0
        self.server_segment_count = 0
        self.total_audio_bytes = 0
        self.received_audio_chunks = 0

    def push_tts_audio(self, pcm_bytes: bytes, sample_rate: int, channels: int) -> List[ActionSegment]:
        if not pcm_bytes:
            return []
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.total_audio_bytes += len(pcm_bytes)
        self.received_audio_chunks += 1

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
        segments = self.exporter.collect_ready_segments(self.audio_received_ms)
        for seg in segments:
            self._save_local_segment(seg)
        return segments

    def finalize(self) -> List[ActionSegment]:
        self.session.finalize()
        segments = self.exporter.flush_final_segments(self.audio_received_ms)
        for seg in segments:
            self._save_local_segment(seg)
        out_path = self.session.save_artifact()
        if out_path:
            log(f"LOCAL_FINAL_NPY saved={out_path}")
        return segments

    def _save_local_segment(self, seg: ActionSegment) -> None:
        out_path = os.path.join(
            self.segment_dir,
            f"{self.session.config.session_id}.seg{seg.segment_index:03d}.npy",
        )
        Path(out_path).write_bytes(seg.npy_bytes)
        self.local_segment_count += 1
        log(
            "LOCAL_SEGMENT "
            f"idx={seg.segment_index} start_ms={seg.start_ms} end_ms={seg.end_ms} "
            f"frames={seg.frame_count} final={int(seg.is_final)} saved={out_path}"
        )

    def save_server_action(self, action_bytes: bytes, suffix: str = "") -> Optional[str]:
        if not action_bytes:
            return None
        idx = self.server_segment_count
        name = f"{self.session.config.session_id}.server.seg{idx:03d}"
        if suffix:
            name += f".{suffix}"
        out_path = os.path.join(self.server_action_dir, name + ".npy")
        Path(out_path).write_bytes(action_bytes)
        self.server_segment_count += 1
        log(f"SERVER_ACTION saved={out_path}")
        return out_path


def consume_tts_stream(
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
    segment_ms: int,
    output_fps: float,
    save_dir: str,
    force_cpu: bool,
    warmup: int,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    session_id = f"tts-a2f-{uuid.uuid4().hex[:8]}"
    request_id = f"req-{uuid.uuid4().hex[:8]}"

    runner = LocalA2FRunner(
        model_path=model_path,
        session_id=session_id,
        request_id=request_id,
        emotion=emotion,
        intensity=intensity,
        output_fps=output_fps,
        segment_ms=segment_ms,
        save_dir=save_dir,
        force_cpu=force_cpu,
        warmup=warmup,
    )

    transcript_parts: List[str] = []
    received_audio = bytearray()
    server_audio_chunks = 0
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
            extra_meta=None,
        )
        log(f"REQUEST_MODE text session_id={session_id} request_id={request_id}")
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
            extra_meta=None,
        )
        log(
            f"REQUEST_MODE wav session_id={session_id} request_id={request_id} "
            f"wav={input_wav} sample_rate={wav_sr} channels={wav_ch}"
        )

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
                    log(f"TRANSCRIPT text={resp.transcript}")

            elif resp.type == AUDIO:
                chunk = bytes(resp.audio_chunk or b"")
                if chunk:
                    if first_audio_ts is None:
                        first_audio_ts = time.time()
                        log(f"FIRST_AUDIO latency_s={first_audio_ts - start_ts:.3f}")
                    server_audio_chunks += 1
                    received_audio.extend(chunk)
                    log(
                        f"AUDIO chunk_idx={server_audio_chunks - 1} "
                        f"bytes={len(chunk)} sample_rate={tts_sample_rate} channels={tts_channels}"
                    )
                    runner.push_tts_audio(
                        pcm_bytes=chunk,
                        sample_rate=tts_sample_rate,
                        channels=tts_channels,
                    )

            elif resp.type == ACTION:
                log(
                    f"SERVER_ACTION_RESPONSE desc={resp.action_description or '-'} "
                    f"bytes={len(resp.action_npz or b'')}"
                )
                runner.save_server_action(bytes(resp.action_npz or b""))

            elif resp.type == ERROR:
                log(f"SERVER_ERROR msg={resp.error_msg}")

            else:
                log(f"UNKNOWN_RESPONSE type={resp.type}")

            if resp.end_of_response:
                log("END_OF_RESPONSE received=true")
                break

    except grpc.RpcError as e:
        log(f"GRPC_ERROR code={e.code()} details={e.details()}")
        raise
    finally:
        channel.close()

    if received_audio:
        wav_path = os.path.join(save_dir, f"{session_id}.tts_received.wav")
        save_wav_pcm16(wav_path, bytes(received_audio), tts_sample_rate, tts_channels)
        log(f"TTS_WAV saved={wav_path}")

    if transcript_parts:
        transcript_path = os.path.join(save_dir, f"{session_id}.transcript.txt")
        Path(transcript_path).write_text("\n".join(transcript_parts), encoding="utf-8")
        log(f"TRANSCRIPT_FILE saved={transcript_path}")

    runner.finalize()

    elapsed = time.time() - start_ts
    log(
        "SUMMARY "
        f"elapsed_s={elapsed:.3f} "
        f"server_audio_chunks={server_audio_chunks} "
        f"received_audio_bytes={len(received_audio)} "
        f"local_segments={runner.local_segment_count} "
        f"server_actions={runner.server_segment_count} "
        f"audio_received_ms={runner.audio_received_ms}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connect to remote VoiceAgent TTS stream, receive AUDIO, and run local A2F."
    )
    parser.add_argument("--host", default="36.103.236.220")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to local ONNX model, e.g. /home/cxm2/data/a2f/audio2face-3d-model/network.onnx",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Text prompt sent in meta['text'] to trigger remote TTS.")
    group.add_argument("--input_wav", type=str, help="Optional local PCM16 wav to push upstream instead of text.")

    parser.add_argument("--robot_id", default="a2f_consumer")
    parser.add_argument("--emotion", default="neutral")
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--locale", default="zh-CN")

    # 远端返回 AUDIO 时，这里假设它是 PCM16；如果对方实际不是这个采样率，需要改成真实值
    parser.add_argument("--tts_sample_rate", type=int, default=24000)
    parser.add_argument("--tts_channels", type=int, default=1)

    parser.add_argument("--request_chunk_ms", type=int, default=200)
    parser.add_argument("--segment_ms", type=int, default=640)
    parser.add_argument("--output_fps", type=float, default=30.0)

    parser.add_argument("--save_dir", default="./tts_consumer_out")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)

    args = parser.parse_args()

    consume_tts_stream(
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
        segment_ms=args.segment_ms,
        output_fps=args.output_fps,
        save_dir=args.save_dir,
        force_cpu=args.force_cpu,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
