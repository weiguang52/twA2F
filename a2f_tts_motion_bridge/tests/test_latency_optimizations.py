import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from a2f_tts_motion_bridge.motion_core.a2f169_to_arkit52 import A2F169ToARKit52
from a2f_tts_motion_bridge.motion_core.motion_recorder import SessionRecorder
from a2f_tts_motion_bridge.motion_core.settings import NUM_FEATURES
from a2f_tts_motion_bridge.outputs.npy_segment_exporter import RollingNpySegmentExporter


FEATURE_NAMES = (
    "jaw_open",
    "mouth_round",
    "mouth_wide",
    "mouth_left_right",
    "upper_face_activity",
    "eye_activity",
    "blink_like",
)


def append_native_frame(recorder: SessionRecorder, time_s: float, value: float) -> None:
    recorder.append_frame(
        time_s,
        np.full(NUM_FEATURES, value, dtype=np.float32),
        {name: value for name in FEATURE_NAMES},
        {name: value for name in recorder.motor_names},
        audio_rms=value,
        debug_extra={"speech_gate": value},
    )


class RetargetBundleCacheTests(unittest.TestCase):
    def tearDown(self) -> None:
        with A2F169ToARKit52._bundle_cache_lock:
            A2F169ToARKit52._bundle_cache.clear()

    def test_static_bundle_is_loaded_once_but_session_state_is_independent(self):
        marker = object()
        fake_bundle = {
            "active_indices": np.asarray([0], dtype=np.int64),
            "marker": marker,
        }
        with A2F169ToARKit52._bundle_cache_lock:
            A2F169ToARKit52._bundle_cache.clear()

        with (
            mock.patch.object(A2F169ToARKit52, "_find_model_dir", return_value=Path("fake-model")),
            mock.patch.object(A2F169ToARKit52, "_read_bundle", return_value=fake_bundle) as read_bundle,
        ):
            first = A2F169ToARKit52(calib_frames=20)
            second = A2F169ToARKit52(calib_frames=20)

        self.assertEqual(read_bundle.call_count, 1)
        self.assertIs(first.marker, second.marker)
        self.assertIsNot(first._jaw_hist, second._jaw_hist)
        self.assertIsNot(first.audio_gate, second.audio_gate)


class FirstSegmentTests(unittest.TestCase):
    @staticmethod
    def make_exporter(recorder: SessionRecorder) -> RollingNpySegmentExporter:
        session = SimpleNamespace(
            recorder=recorder,
            config=SimpleNamespace(
                session_id="test-session",
                emotion="neutral",
                intensity=1.0,
                output_fps=90.0,
            ),
        )
        return RollingNpySegmentExporter(
            session,
            segment_ms=640,
            first_segment_ms=320,
            emit_cover_slack_ms=80,
        )

    def test_first_expression_segment_is_early_then_uses_640ms_cadence(self):
        recorder = SessionRecorder()
        append_native_frame(recorder, 0.26, 0.2)
        exporter = self.make_exporter(recorder)

        first = exporter.collect_ready_segments(321)

        self.assertEqual(len(first), 1)
        self.assertEqual((first[0].start_ms, first[0].end_ms), (0, 320))
        self.assertGreaterEqual(first[0].build_ms, 0.0)

        append_native_frame(recorder, 0.56, 0.8)
        second = exporter.collect_ready_segments(641)

        self.assertEqual(len(second), 1)
        self.assertEqual((second[0].start_ms, second[0].end_ms), (320, 640))

        append_native_frame(recorder, 1.20, 0.8)
        third = exporter.collect_ready_segments(1281)

        self.assertEqual(len(third), 1)
        self.assertEqual((third[0].start_ms, third[0].end_ms), (640, 1280))

    def test_default_segment_size_is_640ms(self):
        recorder = SessionRecorder()
        session = SimpleNamespace(
            recorder=recorder,
            config=SimpleNamespace(
                session_id="test-session",
                emotion="neutral",
                intensity=1.0,
                output_fps=90.0,
            ),
        )
        exporter = RollingNpySegmentExporter(session)

        self.assertEqual(exporter.segment_ms, 640)
        self.assertEqual(exporter.first_segment_ms, 640)
        self.assertEqual(exporter.next_segment_end_ms, 640)


if __name__ == "__main__":
    unittest.main()
