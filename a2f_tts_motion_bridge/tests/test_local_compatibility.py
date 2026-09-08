import importlib
import io
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from a2f_tts_motion_bridge.motion_core.motion_recorder import SessionRecorder
from a2f_tts_motion_bridge.motion_core.settings import MOTOR_CFG
from a2f_tts_motion_bridge.outputs.npy_segment_exporter import RollingNpySegmentExporter
from a2f_tts_motion_bridge.preview import render_motor_npy as preview
from a2f_tts_motion_bridge.tests.test_latency_optimizations import append_native_frame


class LocalCompatibilityTests(unittest.TestCase):
    def test_demo_imports_from_project_root(self):
        importlib.import_module('a2f_tts_motion_bridge.entrypoints.demo_motion_from_wav')

    def test_current_head_neutral_matches_settings(self):
        for name, config in MOTOR_CFG.items():
            self.assertEqual(preview.DEFAULT_MOTOR_NEUTRAL[name], config['neutral'])
        self.assertEqual(preview.DEFAULT_MOTOR_NEUTRAL['jaw_open'], 0.0)
        self.assertEqual(preview.DEFAULT_MOTOR_NEUTRAL['jaw'], 0.5)

    def test_panel_displays_all_current_head_channels(self):
        image = np.zeros((preview.H, preview.W, 3), dtype=np.uint8)
        names = list(MOTOR_CFG)
        with mock.patch.object(preview.cv2, 'putText') as put_text:
            preview.draw_motor_panel(image, names, np.full(len(names), 0.5))
        labels = [call.args[1] for call in put_text.call_args_list]
        for name in names:
            self.assertTrue(any(label.startswith(name + ':') for label in labels))

    def test_segments_cover_audio_and_finalize_once_at_both_frame_rates(self):
        for fps in (30.0, 90.0):
            with self.subTest(fps=fps):
                recorder = SessionRecorder()
                session = SimpleNamespace(recorder=recorder, config=SimpleNamespace(
                    session_id='compat', emotion='neutral', intensity=1.0, output_fps=fps))
                exporter = RollingNpySegmentExporter(session, segment_ms=640, first_segment_ms=320)
                append_native_frame(recorder, 0.26, 0.2)
                segments = exporter.collect_ready_segments(321)
                append_native_frame(recorder, 0.56, 0.4)
                segments += exporter.collect_ready_segments(641)
                append_native_frame(recorder, 1.20, 0.8)
                segments += exporter.collect_ready_segments(1280)
                segments += exporter.flush_final_segments(1280)
                self.assertEqual([(s.start_ms, s.end_ms) for s in segments],
                                 [(0, 320), (320, 640), (640, 1280)])
                self.assertEqual([s.is_final for s in segments], [False, False, True])
                self.assertEqual(exporter.flush_final_segments(1280), [])
                for segment in segments:
                    payload = np.load(io.BytesIO(segment.npy_bytes), allow_pickle=True).item()
                    self.assertEqual(payload['motor_values_30fps'].shape,
                                     (segment.frame_count, len(MOTOR_CFG)))
                    self.assertTrue(np.isfinite(payload['motor_values_30fps']).all())
                    self.assertTrue(np.all(payload['frame_times_30fps'] < segment.end_ms / 1000.0))


if __name__ == '__main__':
    unittest.main()
