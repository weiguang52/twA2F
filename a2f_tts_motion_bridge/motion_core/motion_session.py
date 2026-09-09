import os
import time
from typing import List, Optional

from .audio_stream import preprocess_stream_chunk_to_16k_mono
from .settings import EXPORT_FPS, MOTOR_CFG, TARGET_SR
from .types import AudioChunk, MotorFrame, SessionConfig, SessionSummaryData
from .a2f_stream_infer import A2FModelRuntime, StreamingA2FEngine
from .motion_recorder import SessionRecorder
from .arkit52_to_motor import OnlineRetargeter
from .emotion_control import EmotionControl, canonical_emotion
from .emotion_track import EmotionTrack
from .behavior_layer import BehaviorLayer


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
        self.emotion_control = EmotionControl(config.emotion, config.intensity).patch(config.extra)
        engine_started = time.perf_counter()
        self.engine = StreamingA2FEngine(
            model_path=model_path,
            target_sr=TARGET_SR,
            force_cpu=force_cpu,
            warmup=warmup,
            runtime=runtime,
        )
        self.engine_init_ms = (time.perf_counter() - engine_started) * 1000.0
        self.emotion_track = EmotionTrack(getattr(self.engine.runtime, 'emotion_latent', None))
        self.emotion_track.schedule(self.emotion_control, config.extra, 0.)
        self.engine.emotion_provider = self.emotion_track.sample
        model_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        retarget_started = time.perf_counter()
        self.behavior_layer = BehaviorLayer()
        self.behavior_layer.schedule(config.extra, 0.)
        self.retargeter = OnlineRetargeter(model_dir=model_dir, calib_frames=20,
            behaviors=self.behavior_layer, robot_id=config.extra.get('robot_id'))
        self.retarget_init_ms = (time.perf_counter() - retarget_started) * 1000.0
        self.recorder = SessionRecorder()
        self.recorder.behavior_layer = self.behavior_layer
        self.recorder.set_meta_extra(
            retarget_version=getattr(self.retargeter, "retarget_version", "unknown"),
            arkit_mapper="raw169_to_arkit52_hybrid",
            arkit_model_dir=model_dir,
            **self.emotion_control.metadata(),
            **self.emotion_track.metadata(),
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
        self.update_emotion_meta(dict(emotion=emotion or self.config.emotion,
                                      intensity=self.config.intensity if intensity is None else intensity))

    def update_emotion_meta(self, meta):
        if 'robot_id' in meta and meta['robot_id'] != self.config.extra.get('robot_id'):
            raise ValueError('robot_id cannot change inside a session')
        prepared = (self.behavior_layer.prepare(meta, int(self.engine.total_received)/float(TARGET_SR))
                    if hasattr(self, 'behavior_layer') else None)
        updated = self.emotion_control.patch(meta)
        if hasattr(self, 'emotion_track'):
            self.emotion_track.schedule(updated, meta, int(self.engine.total_received) / float(TARGET_SR))
        # Both protocol and track validation finish before mutating session state.
        if prepared is not None:
            self.behavior_layer.commit(prepared)
        self.emotion_control = updated
        self.config.emotion = updated.emotion
        self.config.intensity = updated.intensity
        self.recorder.set_meta_extra(**updated.metadata())
        if hasattr(self, 'emotion_track'):
            self.recorder.set_meta_extra(**self.emotion_track.metadata())

    def process_chunk(self, chunk: AudioChunk) -> List[MotorFrame]:
        changes = {}
        if chunk.emotion is not None and canonical_emotion(chunk.emotion) != self.emotion_control.emotion:
            changes['emotion'] = chunk.emotion
        if chunk.intensity is not None:
            changes['intensity'] = chunk.intensity
        meta = dict(chunk.emotion_meta)
        if chunk.emotion_mix is not None:
            if 'emotion_mix' in meta:
                raise ValueError('Specify AudioChunk.emotion_mix or meta emotion_mix, not both')
            meta['emotion_mix'] = chunk.emotion_mix
        control = self.emotion_control.patch(changes).patch(meta)
        prepared = self.behavior_layer.prepare(meta, int(self.engine.total_received)/float(TARGET_SR))
        self.emotion_track.schedule(control, meta, int(self.engine.total_received) / float(TARGET_SR))
        # Validate before adding any audio; a rejected update leaves no partial data.
        self.behavior_layer.commit(prepared)
        self.emotion_control = control
        self.config.emotion, self.config.intensity = control.emotion, control.intensity
        self.recorder.set_meta_extra(**control.metadata())
        self.recorder.set_meta_extra(**self.emotion_track.metadata())
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

        emotion, intensity = control.emotion, control.intensity
        infer_items = self.engine.push_audio_chunk(audio_16k, emotion, intensity, control=control)

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
                control=item.get("emotion_control"),
                time_code_s=item['time_code_s'],
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
                base_motors=getattr(self.retargeter, 'last_base_dofs', None),
            )
            frame = MotorFrame(
                frame_id=self.frame_id,
                time_code_ms=int(round(item["time_code_s"] * 1000.0)),
                motor_values=list(self.recorder.base_motor_values[-1]),
                speech_gate=max(self.recorder.speech_gate[-1], min(1.,self.recorder.audio_rms[-1]/.035)),
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
                    motor_values=list(self.behavior_layer.apply(
                        dict(zip(self.motor_names, frame.motor_values)), tc/1000., frame.speech_gate).values()),
                    speech_gate=frame.speech_gate,
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
                control=item.get("emotion_control"),
                time_code_s=item['time_code_s'],
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
                base_motors=getattr(self.retargeter, 'last_base_dofs', None),
            )
            frame = MotorFrame(
                frame_id=self.frame_id,
                time_code_ms=int(round(item["time_code_s"] * 1000.0)),
                motor_values=list(self.recorder.base_motor_values[-1]),
                speech_gate=max(self.recorder.speech_gate[-1], min(1.,self.recorder.audio_rms[-1]/.035)),
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
        return self._save_artifact()

    def tick_idle(self, time_s: float) -> MotorFrame:
        """Host-driven independent idle clock, no audio or ONNX call required."""
        motors=self.behavior_layer.tick(time_s)
        return MotorFrame(int(round(time_s*30)), int(round(time_s*1000)),
                          [motors[k] for k in self.motor_names], [0.]*7)

    def _save_artifact(self) -> str:
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
