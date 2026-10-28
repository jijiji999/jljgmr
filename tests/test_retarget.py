import copy
import json
from pathlib import Path
import tempfile
import unittest

import mujoco as mj
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation as R
from jlj_gmr import A00Retargeter
from jlj_gmr.bvh import load_bvh, HumanMotion
from jlj_gmr.motion import resample, save_motion, load_robot_motion


class RetargetTests(unittest.TestCase):
    @staticmethod
    def reference_frame(r):
        """Encode the known robot reference pose as source world poses."""
        d = r.configuration.data
        frame = {}
        for robot, human, _, _, offset in r.stages[0]:
            b = r.model.body(robot).id
            rot = R.from_quat(d.xquat[b], scalar_first=True) * offset.inv()
            frame[human] = (d.xpos[b].copy() / 1.05, rot.as_quat(scalar_first=True))
        return frame

    def test_reference_pose_all_sources_and_input_immutable(self):
        for source in ("bvh_nokov", "bvh_lafan1", "smplx"):
            with self.subTest(source=source):
                r = A00Retargeter(source, ground_mode="none")
                expected = r.configuration.q.copy()
                frame = self.reference_frame(r)
                before = copy.deepcopy(frame)
                q = r.retarget(frame)
                assert_allclose(q, expected, atol=1e-5)
                for name in frame:
                    assert_allclose(frame[name][0], before[name][0])
                    assert_allclose(frame[name][1], before[name][1])

    def test_height_normalization_and_missing_data(self):
        r = A00Retargeter(actual_human_height=2.)
        self.assertAlmostEqual(r.human_scale_table[r.human_root_name], 1.05 * 1.8 / 2.)
        with self.assertRaisesRegex(ValueError, "Missing human bodies"):
            r.retarget({})
        frame = self.reference_frame(r)
        frame[r.human_root_name] = ([0, 0, 1], [0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "Invalid pose"):
            r.retarget(frame)

    def test_ground_and_joint_speed_limits(self):
        r = A00Retargeter(fps=30., max_joint_speed=2.)
        frame = self.reference_frame(r)
        r.retarget(frame)
        previous = r.configuration.q.copy()
        frame["LeftHand"] = (frame["LeftHand"][0] + [0, 0, .5], [1, 0, 0, 0])
        q = r.retarget(frame)
        self.assertLessEqual(np.max(np.abs(q[7:] - previous[7:])), 2. / 30. + 1e-10)
        self.assertGreaterEqual(r.sole_min_z(), -1e-10)
        self.assertAlmostEqual(np.linalg.norm(q[3:7]), 1.)
        for j in r.joint_ids:
            lo, hi = r.model.jnt_range[j]
            self.assertGreaterEqual(q[r.model.jnt_qposadr[j]], lo - 1e-10)
            self.assertLessEqual(q[r.model.jnt_qposadr[j]], hi + 1e-10)

    def test_initial_ground_preserves_vertical_root_motion(self):
        r = A00Retargeter(ground_mode="initial")
        frame = self.reference_frame(r)
        r.retarget(frame)
        first = r.target_stages[0]["base_link"][0].copy()
        raised = {k: (p + [0, 0, .2], q) for k, (p, q) in frame.items()}
        r.retarget(raised)
        assert_allclose(r.target_stages[0]["base_link"][0] - first, [0, 0, .21], atol=1e-8)

    def test_stage_specific_offsets(self):
        base = A00Retargeter(ground_mode="none")
        config = copy.deepcopy(base.config)
        config["morphology_scaling"] = False
        config["ik_match_table2"]["base_link"][3] = [.1, 0, 0]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps(config))
            r = A00Retargeter(config_path=path, ground_mode="none")
            r.update_targets(self.reference_frame(r))
            assert_allclose(r.target_stages[1]["base_link"][0] - r.target_stages[0]["base_link"][0], [.1, 0, 0])


class MotionTests(unittest.TestCase):
    def test_export_roundtrip_including_last_frame_and_quaternion_order(self):
        r = A00Retargeter()
        q = np.tile(r.configuration.q, (3, 1))
        q[-1, 0] = .123
        q[:, 3:7] = R.from_euler("z", [.1, .2, .3]).as_quat(scalar_first=True)
        with tempfile.TemporaryDirectory() as temp:
            for suffix in ("npz", "pkl", "csv"):
                p = Path(temp) / f"motion.{suffix}"
                save_motion(p, q, 50., r.joint_names)
                actual, fps, names = load_robot_motion(p)
                assert_allclose(actual, q)
                self.assertEqual(fps, 50.)
                self.assertEqual(names, r.joint_names)
                self.assertEqual(actual[-1, 0], .123)

    def test_resample_positions_and_quaternion_short_path(self):
        motion = HumanMotion([
            {"joint": ([0., 0., 0.], [1., 0., 0., 0.])},
            {"joint": ([1., 0., 0.], R.from_euler("z", 90, degrees=True).as_quat(scalar_first=True))},
        ], 1.)
        out = resample(motion, 2.)
        self.assertEqual(len(out.frames), 3)
        assert_allclose(out.frames[1]["joint"][0], [.5, 0, 0])
        assert_allclose(R.from_quat(out.frames[1]["joint"][1], scalar_first=True).as_rotvec(), [0, 0, np.pi/4])

    def test_bvh_per_joint_channel_order_and_frame_time(self):
        text = '''HIERARCHY
ROOT Hips
{
 OFFSET 10 20 30
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT LeftFoot
 {
  OFFSET 1 0 0
  CHANNELS 3 Yrotation Zrotation Xrotation
  JOINT LeftToeBase
  {
   OFFSET 1 0 0
   CHANNELS 3 Xrotation Yrotation Zrotation
  }
 }
 JOINT RightFoot
 {
  OFFSET -1 0 0
  CHANNELS 3 Xrotation Yrotation Zrotation
  JOINT RightToeBase
  {
   OFFSET -1 0 0
   CHANNELS 3 Xrotation Yrotation Zrotation
  }
 }
}
MOTION
Frames: 1
Frame Time: 0.02
0 0 0 90 0 0 90 0 0 0 0 0 0 0 0 0 0 0
'''
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "test.bvh"
            p.write_text(text)
            motion = load_bvh(p, unit_scale=1.)
        self.assertEqual(motion.fps, 50.)
        f = motion.frames[0]
        assert_allclose(f["Hips"][0], [0, 0, 0], atol=1e-10)
        assert_allclose(f["LeftFoot"][0], [0, 0, 1], atol=1e-10)
        assert_allclose(f["LeftToeBase"][0], [0, 1, 1], atol=1e-10)
        assert_allclose(f["LeftFootMod"][0], f["LeftFoot"][0])


if __name__ == "__main__":
    unittest.main()
