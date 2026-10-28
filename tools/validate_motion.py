"""Re-load an exported motion and verify it against XML FK and limits."""
import argparse
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jlj_gmr.motion import load_robot_motion
from jlj_gmr.retarget import DEFAULT_XML


def validate(path, xml, speed_limit=None):
    q, fps, names = load_robot_motion(path)
    model = mj.MjModel.from_xml_path(str(xml))
    data = mj.MjData(model)
    expected = [model.joint(i).name for i in range(1, model.njnt)]
    assert names == expected, "Joint order mismatch"
    assert q.shape[1] == model.nq and len(q), "Invalid frame shape"
    assert np.isfinite(q).all() and fps > 0, "Nonfinite data or nonpositive fps"
    quaternion_error = float(np.max(np.abs(np.linalg.norm(q[:, 3:7], axis=1) - 1)))
    assert quaternion_error < 1e-6, "Non-unit root quaternion"
    violations = {}
    for j in range(1, model.njnt):
        if model.jnt_limited[j]:
            x = q[:, model.jnt_qposadr[j]]
            lo, hi = model.jnt_range[j]
            count = int(np.sum((x < lo - 1e-8) | (x > hi + 1e-8)))
            if count:
                violations[model.joint(j).name] = count
    assert not violations, f"Joint limit violations: {violations}"
    foot_ids = [i for i in range(model.ngeom) if model.geom(i).name and "foot" in model.geom(i).name
                and model.geom_type[i] == mj.mjtGeom.mjGEOM_CAPSULE]
    lowest = np.inf
    for row in q:
        data.qpos[:] = row
        mj.mj_forward(model, data)
        for i in foot_ids:
            axis = data.geom_xmat[i].reshape(3, 3)[:, 2]
            ends = data.geom_xpos[i] + np.array([-1., 1.])[:, None] * model.geom_size[i, 1] * axis
            lowest = min(lowest, float(ends[:, 2].min() - model.geom_size[i, 0]))
    assert lowest > -1e-6, f"Foot penetration: {lowest}"
    speeds = np.max(np.abs(np.diff(q[:, 7:], axis=0)), axis=0) * fps if len(q) > 1 else np.zeros(len(names))
    if speed_limit is not None:
        assert speeds.max() <= speed_limit + 1e-8, "Output speed limit exceeded"
    return {"motion": str(Path(path).resolve()), "frames": len(q), "fps": fps,
            "duration_seconds": (len(q) - 1) / fps, "finite": True, "joint_limit_violations": violations,
            "max_root_quaternion_norm_error": quaternion_error, "min_foot_capsule_z_m": lowest,
            "max_joint_speed_rad_s": float(speeds.max()), "speed_limit_rad_s": speed_limit,
            "peak_joint_speed_rad_s": dict(zip(names, speeds.tolist())), "passed": True}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("motion", type=Path)
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--max-joint-speed", type=float)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = validate(args.motion, args.xml, args.max_joint_speed)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
