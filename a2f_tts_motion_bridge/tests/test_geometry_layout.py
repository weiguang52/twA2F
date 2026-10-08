import unittest
import numpy as np
from a2f_tts_motion_bridge.motion_core.a2f169_to_arkit52 import A2F169ToARKit52


class GeometryLayoutTests(unittest.TestCase):
    def test_contiguous_gather_preserves_target(self):
        rng=np.random.default_rng(42)
        original=rng.normal(size=(140,73,3)).astype(np.float32)
        gathered=original[:,np.array([60,4,22,1,48,22]),:]
        self.assertFalse(gathered.flags.c_contiguous)
        contiguous=np.ascontiguousarray(gathered,dtype=np.float32)
        self.assertTrue(contiguous.flags.c_contiguous)
        before=A2F169ToARKit52.__new__(A2F169ToARKit52)
        after=A2F169ToARKit52.__new__(A2F169ToARKit52)
        for obj,matrix in [(before,gathered),(after,contiguous)]:
            obj.shapes_matrix_skin_masked=matrix
            obj.model_cfg={'skin_strength':1.1,'lower_face_strength':1.25,'blink_strength':.9}
            obj.lip_open_pose_delta_masked=np.ones(18,dtype=np.float32)
            obj.eye_close_pose_delta_masked=np.full(18,.3,dtype=np.float32)
        for _ in range(20):
            skin=rng.normal(size=140).astype(np.float32)
            args=(skin,np.zeros(15),np.zeros(4),float(rng.random()),float(rng.random()))
            np.testing.assert_array_equal(before._build_target_delta(*args),after._build_target_delta(*args))
        np.testing.assert_array_equal(gathered,contiguous)
