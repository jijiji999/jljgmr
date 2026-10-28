"""SMPL-X FK following local GMR utils/smpl.py (LICENSE-GMR).

Optional dependencies: torch, smplx, and separately obtained SMPL-X body models.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from .bvh import HumanMotion


def load_smplx(path, body_model_path, height=None):
    import torch
    import smplx
    from smplx.joint_names import JOINT_NAMES

    with np.load(path, allow_pickle=False) as data:
        required = {"pose_body", "root_orient", "trans", "betas", "gender"}
        if required - set(data):
            raise ValueError(f"SMPL-X npz missing: {sorted(required - set(data))}")
        pose = np.asarray(data["pose_body"], dtype=np.float32)
        orient = np.asarray(data["root_orient"], dtype=np.float32)
        trans = np.asarray(data["trans"], dtype=np.float32)
        betas = np.asarray(data["betas"], dtype=np.float32)
        gender = str(data["gender"].item())
        if gender.startswith("b'"):
            gender = data["gender"].item().decode()
        fps_key = next((k for k in ("mocap_frame_rate", "mocap_framerate", "fps") if k in data), None)
        if fps_key is None:
            raise ValueError("SMPL-X npz needs mocap_frame_rate/mocap_framerate/fps")
        fps = float(data[fps_key])
    if fps <= 0 or pose.ndim != 2 or pose.shape[1] != 63 or len(pose) < 1:
        raise ValueError("Expected nonempty pose_body (T,63) and positive fps")
    if orient.shape != (len(pose), 3) or trans.shape != (len(pose), 3):
        raise ValueError("root_orient and trans must have shape (T,3)")
    if betas.ndim == 1:
        betas = betas[None]
    model = smplx.create(str(Path(body_model_path)), "smplx", gender=gender,
                         use_pca=False, num_betas=betas.shape[-1])
    frames = []
    parents = model.parents.cpu().numpy()
    # Bound memory for longer clips; no gradients are needed for FK.
    with torch.no_grad():
        for start in range(0, len(pose), 256):
            stop = min(start + 256, len(pose))
            size = stop - start
            batch_betas = betas if len(betas) == 1 else betas[start:stop]
            if len(batch_betas) == 1:
                batch_betas = np.repeat(batch_betas, size, axis=0)
            zeros = torch.zeros(size, 3)
            output = model(betas=torch.from_numpy(batch_betas), global_orient=torch.from_numpy(orient[start:stop]),
                           body_pose=torch.from_numpy(pose[start:stop]), transl=torch.from_numpy(trans[start:stop]),
                           left_hand_pose=torch.zeros(size, 45), right_hand_pose=torch.zeros(size, 45),
                           jaw_pose=zeros, leye_pose=zeros, reye_pose=zeros,
                           expression=torch.zeros(size, model.num_expression_coeffs), return_full_pose=True)
            full = output.full_pose.cpu().numpy().reshape(size, -1, 3)
            points = output.joints.cpu().numpy()
            for t in range(size):
                rots, frame = [], {}
                for j, parent in enumerate(parents):
                    rot = R.from_rotvec(full[t, j])
                    if parent >= 0:
                        rot = rots[parent] * rot
                    rots.append(rot)
                    frame[JOINT_NAMES[j]] = (points[t, j].copy(), rot.as_quat(scalar_first=True))
                frames.append(frame)
    return HumanMotion(frames, fps, height)
