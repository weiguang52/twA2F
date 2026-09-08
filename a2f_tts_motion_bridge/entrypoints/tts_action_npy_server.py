import argparse
import time
import traceback
import uuid
from concurrent import futures
from typing import Dict, Iterator, Optional, Tuple

import grpc

from ..outputs.npy_segment_exporter import ActionSegment, RollingNpySegmentExporter
from ..motion_core.a2f_stream_infer import A2FModelRuntime
from ..motion_core.arkit52_to_motor import OnlineRetargeter
from ..motion_core.types import AudioChunk, SessionConfig
from ..motion_core.motion_session import MotorStreamSession
from ..protocol.voiceagent_generated import ActionImageMediaStream_pb2 as vapb2
from ..protocol.voiceagent_generated import ActionImageMediaStream_pb2_grpc as vapb2_grpc

EXPRESSION_SEGMENT_MS = 640
EXPRESSION_FIRST_SEGMENT_MS = 320


PCM16 = getattr(vapb2, "PCM16", 0)
AUDIO = getattr(vapb2, "AUDIO", 0)
TRANSCRIPT = getattr(vapb2, "TRANSCRIPT", 1)
ACTION = getattr(vapb2, "ACTION", 2)
ERROR = getattr(vapb2, "ERROR", 3)


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_float(value: Optional[str], default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def _parse_positive_int(value: Optional[str], default: int) -> int:
    try:
        parsed = int(str(value).strip()) if value is not None else int(default)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _pcm_bytes_to_ms(num_bytes: int, sample_rate: int, channels: int, bits_per_sample: int = 16) -> int:
    bytes_per_sample = max(1, bits_per_sample // 8)
    frame_bytes = max(1, channels * bytes_per_sample)
    frame_count = float(num_bytes) / float(frame_bytes)
    duration_ms = frame_count * 1000.0 / float(max(1, sample_rate))
    return int(round(duration_ms))

def _log(msg: str) -> None:
    print(f"[VoiceAgentCompat] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def _short_meta(meta: Dict[str, str], max_items: int = 12) -> Dict[str, str]:
    if not meta:
        return {}
    out: Dict[str, str] = {}
    for idx, (k, v) in enumerate(meta.items()):
        if idx >= max_items:
            out["..."] = f"+{len(meta) - max_items} more"
            break
        sv = str(v)
        if len(sv) > 120:
            sv = sv[:117] + "..."
        out[str(k)] = sv
    return out

class VoiceAgentCompat(vapb2_grpc.VoiceAgentServicer):
    def __init__(
        self,
        model_path: str,
        *,
        force_cpu: bool = False,
        warmup: int = 1,
        default_segment_ms: int = EXPRESSION_SEGMENT_MS,
        default_first_segment_ms: int = EXPRESSION_FIRST_SEGMENT_MS,
        default_output_fps: float = 30.0,
        default_save_dir: str = "./artifacts",
        runtime: Optional[A2FModelRuntime] = None,
    ):
        self.model_path = model_path
        self.force_cpu = force_cpu
        self.warmup = warmup
        self.default_segment_ms = int(default_segment_ms)
        self.default_first_segment_ms = int(default_first_segment_ms)
        self.default_output_fps = float(default_output_fps)
        self.default_save_dir = default_save_dir
        self.runtime = runtime

    def _build_session(self, first_req: vapb2.AudioRequest) -> Tuple[MotorStreamSession, RollingNpySegmentExporter, Dict[str, object]]:
        build_started = time.perf_counter()
        meta = dict(first_req.meta)
        session_id = first_req.session_id or meta.get("session_id") or f"va-{uuid.uuid4().hex[:12]}"
        emotion = meta.get("emotion", "neutral")
        intensity = _parse_float(meta.get("intensity"), 1.0)
        output_fps = _parse_float(meta.get("output_fps"), self.default_output_fps)
        save_server_npy = _parse_bool(meta.get("save_server_npy"), False)
        save_server_npy_dir = meta.get("save_server_npy_dir", self.default_save_dir)
        speaker_id = meta.get("speaker_id", "")
        request_id = first_req.request_id or meta.get("request_id", "")
        # Keep the established 640 ms cadence after the first expression
        # packet, but allow a shorter first packet to leave as soon as the
        # initial A2F inference covers it.
        requested_segment_ms = meta.get("segment_ms")
        requested_first_segment_ms = meta.get("first_segment_ms")
        segment_ms = self.default_segment_ms
        first_segment_ms = min(
            segment_ms,
            _parse_positive_int(
                requested_first_segment_ms,
                self.default_first_segment_ms,
            ),
        )

        _log(
            "BUILD_SESSION "
            f"session_id={session_id} request_id={request_id or '-'} "
            f"emotion={emotion} intensity={intensity} output_fps={output_fps} "
            f"save_server_npy={save_server_npy} save_dir={save_server_npy_dir} "
            f"speaker_id={speaker_id or '-'} "
            f"sample_rate={int(first_req.sample_rate or int(meta.get('sample_rate', 24000)))} "
            f"channels={int(first_req.channels or int(meta.get('channels', 1)))} "
            f"segment_ms={segment_ms} first_segment_ms={first_segment_ms} "
            f"requested_segment_ms={requested_segment_ms or '-'} "
            f"requested_first_segment_ms={requested_first_segment_ms or '-'} "
            f"meta={_short_meta(meta)}"
        )
        cfg = SessionConfig(
            session_id=session_id,
            emotion=emotion,
            intensity=intensity,
            output_fps=output_fps,
            save_npy=save_server_npy,
            npy_save_dir=save_server_npy_dir,
            request_id=request_id,
            speaker_id=speaker_id,
            extra=meta,
        )
        session = MotorStreamSession(
            self.model_path,
            cfg,
            force_cpu=self.force_cpu,
            warmup=self.warmup,
            runtime=self.runtime,
        )
        exporter = RollingNpySegmentExporter(
            session,
            segment_ms=segment_ms,
            first_segment_ms=first_segment_ms,
            emit_cover_slack_ms=int(meta.get("emit_cover_slack_ms", 80)),
        )
        build_ms = (time.perf_counter() - build_started) * 1000.0
        _log(
            "BUILD_SESSION_READY "
            f"session_id={session_id} request_id={request_id or '-'} "
            f"total_ms={build_ms:.1f} engine_init_ms={session.engine_init_ms:.1f} "
            f"retarget_init_ms={session.retarget_init_ms:.1f}"
        )
        state: Dict[str, object] = {
            "emotion": emotion,
            "intensity": intensity,
            "sample_rate": int(first_req.sample_rate or int(meta.get("sample_rate", 24000))),
            "channels": int(first_req.channels or int(meta.get("channels", 1))),
            "request_id": request_id or f"req-{uuid.uuid4().hex[:10]}",
            "chunk_id": 0,
            "audio_received_ms": 0,
            "bits_per_sample": 16,
            "session_build_ms": build_ms,
            "first_audio_perf": None,
            "first_infer_logged": False,
            "first_segment_logged": False,
        }
        return session, exporter, state

    def _update_control_state(
        self,
        req: vapb2.AudioRequest,
        session: MotorStreamSession,
        exporter: RollingNpySegmentExporter,
        state: Dict[str, object],
    ) -> None:
        meta = dict(req.meta)
        emotion = meta.get("emotion")
        intensity = _parse_float(meta.get("intensity"), float(state["intensity"])) if "intensity" in meta else None
        if emotion is not None or intensity is not None:
            session.update_emotion(emotion or session.config.emotion, intensity if intensity is not None else session.config.intensity)
            exporter.update_emotion(emotion, intensity)
            if emotion is not None:
                state["emotion"] = emotion
            if intensity is not None:
                state["intensity"] = float(intensity)
        if req.sample_rate:
            state["sample_rate"] = int(req.sample_rate)
        if req.channels:
            state["channels"] = int(req.channels)

    def _segment_to_response(self, request_id: str, segment: ActionSegment) -> vapb2.AudioResponse:
        desc = (
            f"Expression segment #{segment.segment_index}: "
            f"{segment.start_ms}-{segment.end_ms} ms, "
            f"frames={segment.frame_count}, final={int(segment.is_final)}"
        )
        model_version = (
            "motorstream|"
            f"segment_ms={segment.end_ms - segment.start_ms}|"
            f"segment_idx={segment.segment_index}|"
            f"start_ms={segment.start_ms}|"
            f"end_ms={segment.end_ms}|"
            f"final={int(segment.is_final)}"
        )
        return vapb2.AudioResponse(
            request_id=request_id,
            ts=int(time.time() * 1000),
            type=ACTION,
            action_npz=segment.npy_bytes,
            action_description=desc,
            model_version=model_version,
            end_of_response=False,
        )

    def StreamSession(self, request_iterator, context):
        request_iter = iter(request_iterator)
        try:
            first_req = next(request_iter)
        except StopIteration:
            _log("STREAM_EMPTY no request received")
            return
        stream_open_perf = time.perf_counter()

        peer = "-"
        try:
            peer = context.peer()
        except Exception:
            pass

        first_meta = dict(first_req.meta)
        _log(
            "STREAM_OPEN "
            f"peer={peer} "
            f"session_id={first_req.session_id or first_meta.get('session_id', '-')}"
            f" request_id={first_req.request_id or first_meta.get('request_id', '-')}"
            f" ts={getattr(first_req, 'ts', 0)} "
            f"codec={getattr(first_req, 'codec', None)} "
            f"sample_rate={getattr(first_req, 'sample_rate', 0)} "
            f"channels={getattr(first_req, 'channels', 0)} "
            f"audio_len={len(bytes(first_req.audio_chunk or b''))} "
            f"end_of_stream={bool(first_req.end_of_stream)} "
            f"meta={_short_meta(first_meta)}"
        )

        if getattr(first_req, "codec", PCM16) != PCM16:
            _log(
                "STREAM_REJECT "
                f"peer={peer} request_id={first_req.request_id or '-'} "
                f"reason=unsupported_codec codec={getattr(first_req, 'codec', None)} expected={PCM16}"
            )
            yield vapb2.AudioResponse(
                request_id=first_req.request_id,
                ts=int(time.time() * 1000),
                type=ERROR,
                error_msg="VoiceAgentCompat currently supports PCM16 audio_chunk only.",
                end_of_response=True,
            )
            return

        try:
            session, exporter, state = self._build_session(first_req)
            state["stream_open_perf"] = stream_open_perf
            request_id = str(state["request_id"])
            seen_audio = False

            def log_first_inference_if_ready(phase: str) -> None:
                if bool(state["first_infer_logged"]) or session.engine.infer_frames <= 0:
                    return
                state["first_infer_logged"] = True
                completed_perf = time.perf_counter()
                first_audio_perf = state.get("first_audio_perf")
                since_first_audio_ms = (
                    (completed_perf - float(first_audio_perf)) * 1000.0
                    if first_audio_perf is not None
                    else -1.0
                )
                _log(
                    "FIRST_INFERENCE "
                    f"session_id={session.config.session_id} request_id={request_id} "
                    f"phase={phase} infer_ms={(session.engine.first_infer_sec or 0.0) * 1000.0:.1f} "
                    f"since_first_audio_ms={since_first_audio_ms:.1f} "
                    f"audio_received_ms={state['audio_received_ms']} "
                    f"native_frames={session.engine.infer_frames}"
                )

            def build_segment_response(segment: ActionSegment, event: str) -> vapb2.AudioResponse:
                response_started = time.perf_counter()
                response = self._segment_to_response(request_id, segment)
                response_build_ms = (time.perf_counter() - response_started) * 1000.0
                if not bool(state["first_segment_logged"]):
                    state["first_segment_logged"] = True
                    now_perf = time.perf_counter()
                    first_audio_perf = state.get("first_audio_perf")
                    since_first_audio_ms = (
                        (now_perf - float(first_audio_perf)) * 1000.0
                        if first_audio_perf is not None
                        else -1.0
                    )
                    _log(
                        "FIRST_SEGMENT_YIELD "
                        f"session_id={session.config.session_id} request_id={request_id} "
                        f"event={event} segment_idx={segment.segment_index} "
                        f"start_ms={segment.start_ms} end_ms={segment.end_ms} "
                        f"since_stream_open_ms={(now_perf - stream_open_perf) * 1000.0:.1f} "
                        f"since_first_audio_ms={since_first_audio_ms:.1f} "
                        f"session_build_ms={float(state['session_build_ms']):.1f} "
                        f"segment_build_ms={segment.build_ms:.1f} "
                        f"response_build_ms={response_build_ms:.1f}"
                    )
                return response

            def handle_req(req: vapb2.AudioRequest) -> Iterator[vapb2.AudioResponse]:
                nonlocal seen_audio
                meta = dict(req.meta)
                self._update_control_state(req, session, exporter, state)
                audio_chunk = bytes(req.audio_chunk or b"")

                _log(
                    "REQ "
                    f"peer={peer} session_id={session.config.session_id} request_id={request_id} "
                    f"chunk_id={state['chunk_id']} "
                    f"codec={getattr(req, 'codec', None)} "
                    f"sample_rate={int(state['sample_rate'])} channels={int(state['channels'])} "
                    f"audio_len={len(audio_chunk)} end_of_stream={bool(req.end_of_stream)} "
                    f"emotion={state['emotion']} intensity={state['intensity']} "
                    f"meta={_short_meta(meta)}"
                )

                if audio_chunk:
                    is_first_audio = state["first_audio_perf"] is None
                    if is_first_audio:
                        state["first_audio_perf"] = (
                            stream_open_perf if req is first_req else time.perf_counter()
                        )
                        _log(
                            "FIRST_AUDIO_RECEIVED "
                            f"session_id={session.config.session_id} request_id={request_id} "
                            f"since_stream_open_ms={(float(state['first_audio_perf']) - stream_open_perf) * 1000.0:.1f} "
                            f"bytes={len(audio_chunk)}"
                        )
                    seen_audio = True
                    sample_rate = int(state["sample_rate"])
                    channels = int(state["channels"])
                    duration_ms = _pcm_bytes_to_ms(
                        len(audio_chunk),
                        sample_rate=sample_rate,
                        channels=channels,
                        bits_per_sample=16,
                    )
                    state["audio_received_ms"] = int(state["audio_received_ms"]) + duration_ms
                    chunk = AudioChunk(
                        session_id=session.config.session_id,
                        chunk_id=int(state["chunk_id"]),
                        pts_ms=int(state["audio_received_ms"]),
                        pcm_bytes=audio_chunk,
                        sample_rate=sample_rate,
                        channels=channels,
                        bits_per_sample=16,
                        emotion=str(state["emotion"]),
                        intensity=float(state["intensity"]),
                    )
                    state["chunk_id"] = int(state["chunk_id"]) + 1
                    _log(
                        "AUDIO_CHUNK "
                        f"session_id={session.config.session_id} request_id={request_id} "
                        f"chunk_id={chunk.chunk_id} bytes={len(audio_chunk)} "
                        f"duration_ms={duration_ms} audio_received_ms={state['audio_received_ms']}"
                    )
                    process_started = time.perf_counter()
                    session.process_chunk(chunk)
                    process_ms = (time.perf_counter() - process_started) * 1000.0
                    if is_first_audio:
                        _log(
                            "FIRST_CHUNK_PROCESSED "
                            f"session_id={session.config.session_id} request_id={request_id} "
                            f"process_ms={process_ms:.1f} duration_ms={duration_ms} "
                            f"native_frames={session.engine.infer_frames}"
                        )
                    log_first_inference_if_ready("stream")
                    for segment in exporter.collect_ready_segments(int(state["audio_received_ms"])):
                        _log(
                            "SEGMENT_READY "
                            f"session_id={session.config.session_id} request_id={request_id} "
                            f"segment_idx={segment.segment_index} start_ms={segment.start_ms} "
                            f"end_ms={segment.end_ms} frames={segment.frame_count} final={int(segment.is_final)} "
                            f"payload_bytes={len(segment.npy_bytes)} build_ms={segment.build_ms:.1f}"
                        )
                        yield build_segment_response(segment, "stream")

                if req.end_of_stream:
                    _log(
                        "EOS "
                        f"session_id={session.config.session_id} request_id={request_id} "
                        f"seen_audio={seen_audio} total_audio_ms={state['audio_received_ms']}"
                    )
                    finalize_started = time.perf_counter()
                    session.finalize()
                    _log(
                        "FINALIZE_DONE "
                        f"session_id={session.config.session_id} request_id={request_id} "
                        f"finalize_ms={(time.perf_counter() - finalize_started) * 1000.0:.1f}"
                    )
                    log_first_inference_if_ready("finalize")
                    for segment in exporter.flush_final_segments(int(state["audio_received_ms"])):
                        _log(
                            "SEGMENT_FINAL "
                            f"session_id={session.config.session_id} request_id={request_id} "
                            f"segment_idx={segment.segment_index} start_ms={segment.start_ms} "
                            f"end_ms={segment.end_ms} frames={segment.frame_count} final={int(segment.is_final)} "
                            f"payload_bytes={len(segment.npy_bytes)} build_ms={segment.build_ms:.1f}"
                        )
                        yield build_segment_response(segment, "final")
                    if session.config.save_npy:
                        _log(
                            "SAVE_ARTIFACT "
                            f"session_id={session.config.session_id} request_id={request_id} "
                            f"save_dir={session.config.npy_save_dir}"
                        )
                        session.save_artifact()
                    _log(
                        "STREAM_CLOSE "
                        f"session_id={session.config.session_id} request_id={request_id} "
                        f"status={'ACTION' if seen_audio else 'ERROR_NO_AUDIO'}"
                    )
                    yield vapb2.AudioResponse(
                        request_id=request_id,
                        ts=int(time.time() * 1000),
                        end_of_response=True,
                        type=ACTION if seen_audio else ERROR,
                        error_msg="" if seen_audio else "No audio_chunk received.",
                        model_version="motorstream|stream_closed",
                    )

            for resp in handle_req(first_req):
                yield resp
            if first_req.end_of_stream:
                return
            for req in request_iter:
                for resp in handle_req(req):
                    yield resp
                if req.end_of_stream:
                    return

        except Exception as e:
            _log(
                "STREAM_EXCEPTION "
                f"peer={peer} request_id={getattr(first_req, 'request_id', '-') or '-'} "
                f"error={repr(e)}"
            )
            traceback.print_exc()
            yield vapb2.AudioResponse(
                request_id=getattr(first_req, "request_id", ""),
                ts=int(time.time() * 1000),
                type=ERROR,
                error_msg=f"server exception: {e}",
                end_of_response=True,
            )
            return

    def ProcessAudioUnary(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("ProcessAudioUnary is not implemented in VoiceAgentCompat. Use StreamSession.")
        raise NotImplementedError("ProcessAudioUnary is not implemented.")

    def SetModelPreference(self, request, context):
        return vapb2.ModelPreferenceResponse(
            request_id=request.request_id,
            accepted_model="motorstream-a2f-3d",
            model_version="motorstream|action-stream",
        )

    def UploadImage(self, request, context):
        return vapb2.ImageUploadResponse(
            request_id=request.request_id,
            image_id="unsupported",
            error_msg="UploadImage is not used by motorstream integration.",
        )


def serve(
    model_path: str,
    *,
    host: str = "0.0.0.0",
    port: int = 50061,
    force_cpu: bool = False,
    warmup: int = 1,
    segment_ms: int = EXPRESSION_SEGMENT_MS,
    first_segment_ms: int = EXPRESSION_FIRST_SEGMENT_MS,
    output_fps: float = 30.0,
    save_dir: str = "./artifacts",
    max_workers: int = 8,
) -> None:
    _log(
        f"MODEL_LOADING model_path={model_path} force_cpu={force_cpu} "
        f"warmup={warmup}"
    )
    runtime_started = time.perf_counter()
    runtime = A2FModelRuntime(
        model_path,
        force_cpu=force_cpu,
        warmup=warmup,
    )
    for run_index, run_sec in enumerate(runtime.warmup_run_sec, start=1):
        _log(f"MODEL_WARMUP run={run_index} infer_ms={run_sec * 1000.0:.1f}")
    _log(
        f"MODEL_READY total_ms={(time.perf_counter() - runtime_started) * 1000.0:.1f} "
        f"session_init_ms={runtime.session_init_sec * 1000.0:.1f} "
        f"warmup_ms={runtime.warmup_sec * 1000.0:.1f} "
        f"warmup_runs={len(runtime.warmup_run_sec)} providers={runtime.providers}"
    )

    retarget_started = time.perf_counter()
    retarget_cache_hit = OnlineRetargeter.preload_model_bundle(model_path)
    _log(
        "RETARGET_READY "
        f"load_ms={(time.perf_counter() - retarget_started) * 1000.0:.1f} "
        f"cache_hit={int(retarget_cache_hit)} model_path={model_path}"
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    vapb2_grpc.add_VoiceAgentServicer_to_server(
        VoiceAgentCompat(
            model_path,
            force_cpu=force_cpu,
            warmup=warmup,
            default_segment_ms=segment_ms,
            default_first_segment_ms=first_segment_ms,
            default_output_fps=output_fps,
            default_save_dir=save_dir,
            runtime=runtime,
        ),
        server,
    )
    addr = f"{host}:{port}"
    server.add_insecure_port(addr)
    server.start()
    _log(
        f"LISTEN addr={addr} model_path={model_path} force_cpu={force_cpu} "
        f"warmup={warmup} segment_ms={segment_ms} output_fps={output_fps} "
        f"first_segment_ms={first_segment_ms} "
        f"save_dir={save_dir} max_workers={max_workers}"
    )
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50061)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--segment_ms",
        type=int,
        default=EXPRESSION_SEGMENT_MS,
        help="Duration of every expression segment after the first one.",
    )
    parser.add_argument(
        "--first_segment_ms",
        type=int,
        default=EXPRESSION_FIRST_SEGMENT_MS,
        help="Shorter initial expression segment emitted before the 640 ms cadence.",
    )
    parser.add_argument("--output_fps", type=float, default=30.0)
    parser.add_argument("--save_dir", default="./artifacts")
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()
    serve(
        model_path=args.model,
        host=args.host,
        port=args.port,
        force_cpu=args.force_cpu,
        warmup=args.warmup,
        segment_ms=args.segment_ms,
        first_segment_ms=args.first_segment_ms,
        output_fps=args.output_fps,
        save_dir=args.save_dir,
        max_workers=args.max_workers,
    )
