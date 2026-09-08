import os
import time
from typing import List, Optional

from .audio_stream import preprocess_stream_chunk_to_16k_mono
from .settings import EXPORT_FPS, MOTOR_CFG, TARGET_SR
from .types import AudioChunk, MotorFrame, SessionConfig, SessionSummaryData
from .a2f_stream_infer import A2FModelRuntime, StreamingA2FEngine
from .motion_recorder import SessionRecorder
from .arkit52_to_motor import OnlineRetargeter


def now_ms() -> int:
    return int(round(time.time() * 1000.0))


class MotorStreamSession:
    def __init__(
        self,
        model_path: str,
        config: SessionConfig,
        force_cpu: bool = False,
        warmup: int = 1,
        runtime: Optional[A2FModelRuntime] = None,
    ):
        self.model_path = model_path
        self.config = config
        engine_started = time.perf_counter()
        self.engine = StreamingA2FEngine(
            model_path=model_path,
            target_sr=TARGET_SR,
            force_cpu=force_cpu,
            warmup=warmup,
            runtime=runtime,
        )
        self.engine_init_ms = (time.perf_counter() - engine_started) * 1000.0
        model_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        retarget_started = time.perf_counter()
        self.retargeter = OnlineRetargeter(model_dir=model_dir, calib_frames=20)
        self.retarget_init_ms = (time.perf_counter() - retarget_started) * 1000.0
        self.recorder = SessionRecorder()
        self.recorder.set_meta_extra(
            retarget_version=getattr(self.retargeter, "retarget_version", "unknown"),
            arkit_mapper="raw169_to_arkit52_hybrid",
            arkit_model_dir=model_dir,
        )
        self.motor_names = list(MOTOR_CFG.keys())
        self.frame_id = 0
        self.batch_id = 0
        self.total_audio_ms = 0
        self.preprocess_ms_total = 0.0
        self.retarget_ms_total = 0.0
        self.preprocess_calls = 0
        self.retarget_calls = 0
        self.npy_path = ""

    def update_emotion(self, emotion: str, intensity: float):
        self.config.emotion = emotion or self.config.emotion
        if intensity is not None:
            self.config.intensity = float(intensity)

    def process_chunk(self, chunk: AudioChunk) -> List[MotorFrame]:
        t0 = time.perf_counter()
        audio_16k = preprocess_stream_chunk_to_16k_mono(
            pcm_bytes=chunk.pcm_bytes,
            sample_rate=chunk.sample_rate,
            channels=chunk.channels,
            target_sr=TARGET_SR,
            bits_per_sample=chunk.bits_per_sample,
        )
        t1 = time.perf_counter()
        self.preprocess_ms_total += (t1 - t0) * 1000.0
        self.preprocess_calls += 1

        self.recorder.append_audio_16k(audio_16k)
        if chunk.pts_ms >= 0:
            self.total_audio_ms = max(self.total_audio_ms, chunk.pts_ms)

        emotion = chunk.emotion if chunk.emotion is not None else self.config.emotion
        intensity = chunk.intensity if chunk.intensity is not None else self.config.intensity
        infer_items = self.engine.push_audio_chunk(audio_16k, emotion, intensity)

        frames = []
        for item in infer_items:
            t2 = time.perf_counter()
            item_emotion = item.get("emotion", emotion)
            item_intensity = float(item.get("intensity", intensity))
            feats, motors = self.retargeter.update(
                item["weights169"],
                emotion=item_emotion,
                intensity=item_intensity,
                audio_rms=item.get("audio_rms"),
            )
            t3 = time.perf_counter()
            self.retarget_ms_total += (t3 - t2) * 1000.0
            self.retarget_calls += 1

            self.recorder.append_frame(
                item["time_code_s"],
                item["weights169"],
                feats,
                motors,
                arkit52=getattr(self.retargeter, "last_blendshapes", None),
                audio_rms=item.get("audio_rms"),
                debug_extra=getattr(self.retargeter, "last_debug", None),
            )
            frame = MotorFrame(
                frame_id=self.frame_id,
                time_code_ms=int(round(item["time_code_s"] * 1000.0)),
                motor_values=[motors[name] for name in self.motor_names],
                debug_features=[
                    feats["jaw_open"], feats["mouth_round"], feats["mouth_wide"],
                    feats["mouth_left_right"], feats["upper_face_activity"],
                    feats["eye_activity"], feats["blink_like"],
                ],
            )
            self.frame_id += 1
            frames.append(frame)
        return self._resample_frames_to_30hz(frames)

    def _resample_frames_to_30hz(self, frames: List[MotorFrame]) -> List[MotorFrame]:
        # 当前模型原生帧率低于 30Hz。这里先保持现有的轻量级零阶保持行为，
        # 避免对线上接口做大改。若后续仍需要进一步压低 lag，优先在 recorder
        # 的 30fps 导出里改成线性插值，而不是继续加大 retarget 侧平滑。
        if not frames:
            return []
        target = []
        step_ms = int(round(1000.0 / EXPORT_FPS))
        for i, frame in enumerate(frames):
            next_tc = frames[i + 1].time_code_ms if i + 1 < len(frames) else frame.time_code_ms + 100
            gap = max(step_ms, next_tc - frame.time_code_ms)
            n = max(1, int(round(gap / step_ms)))
            n = min(n, 3)
            for k in range(n):
                tc = frame.time_code_ms + k * step_ms
                target.append(MotorFrame(
                    frame_id=frame.frame_id * 10 + k,
                    time_code_ms=tc,
                    motor_values=list(frame.motor_values),
                    debug_features=list(frame.debug_features),
                ))
        return target

    def finalize(self) -> List[MotorFrame]:
        infer_items = self.engine.finalize()
        frames = []
        final_emotion = getattr(self.engine, "current_emotion", self.config.emotion)
        final_intensity = float(getattr(self.engine, "current_intensity", self.config.intensity))
        for item in infer_items:
            t2 = time.perf_counter()
            item_emotion = item.get("emotion", final_emotion)
            item_intensity = float(item.get("intensity", final_intensity))
            feats, motors = self.retargeter.update(
                item["weights169"],
                emotion=item_emotion,
                intensity=item_intensity,
                audio_rms=item.get("audio_rms"),
            )
            t3 = time.perf_counter()
            self.retarget_ms_total += (t3 - t2) * 1000.0
            self.retarget_calls += 1

            self.recorder.append_frame(
                item["time_code_s"],
                item["weights169"],
                feats,
                motors,
                arkit52=getattr(self.retargeter, "last_blendshapes", None),
                audio_rms=item.get("audio_rms"),
                debug_extra=getattr(self.retargeter, "last_debug", None),
            )
            frame = MotorFrame(
                frame_id=self.frame_id,
                time_code_ms=int(round(item["time_code_s"] * 1000.0)),
                motor_values=[motors[name] for name in self.motor_names],
                debug_features=[
                    feats["jaw_open"], feats["mouth_round"], feats["mouth_wide"],
                    feats["mouth_left_right"], feats["upper_face_activity"],
                    feats["eye_activity"], feats["blink_like"],
                ],
            )
            self.frame_id += 1
            frames.append(frame)
        return self._resample_frames_to_30hz(frames)

    def save_artifact(self) -> str:
        if not self.config.save_npy:
            return ""
        out_dir = self.config.npy_save_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{self.config.session_id}.npy")
        self.npy_path = self.recorder.save(out_path)
        return self.npy_path

    def build_summary(self) -> SessionSummaryData:
        actual_output_fps = EXPORT_FPS
        return SessionSummaryData(
            session_id=self.config.session_id,
            total_audio_ms=int(self.total_audio_ms),
            total_frames=int(self.frame_id),
            actual_output_fps=float(actual_output_fps),
            npy_path=self.npy_path,
            preprocess_ms_avg=(self.preprocess_ms_total / self.preprocess_calls) if self.preprocess_calls else 0.0,
            infer_ms_avg=(self.engine.pure_infer_sec * 1000.0 / self.engine.infer_frames) if self.engine.infer_frames else 0.0,
            retarget_ms_avg=(self.retarget_ms_total / self.retarget_calls) if self.retarget_calls else 0.0,
        )
