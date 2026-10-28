"""Portable arrays and explicit GMR output quaternion/joint conventions."""
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from .bvh import HumanMotion


def load_pose_npz(path):
    with np.load(path, allow_pickle=False) as a:
        names = a["body_names"].astype(str).tolist()
        p, q, fps = a["positions"], a["quaternions"], float(a["fps"])
        height = float(a["human_height"]) if "human_height" in a else None
    if p.ndim != 3 or p.shape[1:] != (len(names), 3) or q.shape != (*p.shape[:2], 4) or len(p) < 1 or fps <= 0:
        raise ValueError("Expected positions (T,J,3), quaternions (T,J,4 wxyz), body_names (J,), positive fps")
    return HumanMotion([{n: (p[t, j].copy(), q[t, j].copy()) for j, n in enumerate(names)}
                        for t in range(len(p))], fps, height)


def resample(motion, fps):
    if fps <= 0:
        raise ValueError("fps must be positive")
    if len(motion.frames) == 1 or np.isclose(fps, motion.fps):
        return HumanMotion(motion.frames, fps, motion.height)
    src = np.arange(len(motion.frames)) / motion.fps
    dst = np.arange(int(np.floor(src[-1] * fps + 1e-7)) + 1) / fps
    frames = [{} for _ in dst]
    for name in motion.frames[0]:
        p = np.stack([f[name][0] for f in motion.frames])
        q = np.stack([f[name][1] for f in motion.frames])
        p = np.stack([np.interp(dst, src, p[:, k]) for k in range(3)], axis=-1)
        q = Slerp(src, R.from_quat(q, scalar_first=True))(dst).as_quat(scalar_first=True)
        for i in range(len(dst)):
            frames[i][name] = (p[i], q[i])
    return HumanMotion(frames, fps, motion.height)


def save_motion(path, qpos, fps, joint_names, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    qpos = np.asarray(qpos)
    if qpos.ndim != 2 or len(qpos) < 1 or qpos.shape[1] != 7 + len(joint_names) or not np.isfinite(qpos).all() or fps <= 0:
        raise ValueError("Invalid qpos array/fps/joint_names")
    motion = {"fps": float(fps), "root_pos": qpos[:, :3],
              "root_rot": qpos[:, [4, 5, 6, 3]], "dof_pos": qpos[:, 7:],
              "joint_names": list(joint_names), "root_rot_order": "xyzw", "robot": "a00",
              "local_body_pos": None, "link_body_list": None}
    if path.suffix == ".npz":
        np.savez_compressed(path, qpos=qpos, fps=fps, joint_names=np.asarray(joint_names), root_pos=motion["root_pos"],
                            root_rot=motion["root_rot"], dof_pos=motion["dof_pos"], root_rot_order="xyzw")
    elif path.suffix == ".pkl":
        with path.open("wb") as f:
            pickle.dump(motion, f)
    elif path.suffix == ".csv":
        # CSV explicitly uses MuJoCo qpos: xyz, wxyz, XML-order joints (radians).
        header = ["root_x", "root_y", "root_z", "root_qw", "root_qx", "root_qy", "root_qz", *joint_names]
        np.savetxt(path, qpos, delimiter=",", header=",".join(header), comments="")
    else:
        raise ValueError("Output must be .pkl, .npz, or .csv")
    sidecar = {"fps": fps, "robot": "a00", "joint_names": list(joint_names),
               "qpos_root_rot_order": "wxyz", "gmr_root_rot_order": "xyzw",
               "position_unit": "m", "joint_unit": "rad", **(metadata or {})}
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n")


def load_robot_motion(path):
    path = Path(path)
    if path.suffix == ".pkl":
        # Pickle input is for locally trusted generated robot motions only.
        with path.open("rb") as f:
            a = pickle.load(f)
        q = np.concatenate([a["root_pos"], np.asarray(a["root_rot"])[:, [3, 0, 1, 2]], a["dof_pos"]], axis=1)
        return q, float(a["fps"]), a.get("joint_names")
    if path.suffix == ".csv":
        q = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        info = json.loads(path.with_suffix(".csv.json").read_text())
        return q, float(info["fps"]), info["joint_names"]
    with np.load(path, allow_pickle=False) as a:
        return a["qpos"].copy(), float(a["fps"]), a["joint_names"].astype(str).tolist()
