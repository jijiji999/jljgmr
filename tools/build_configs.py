"""Transfer the local GMR anatomical conventions into A00 body coordinates.

Only config generation needs the reference GMR checkout; retargeting is independent.
Run from the project root: python tools/build_configs.py --gmr /home/rx/project/GMR
"""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "A00_MuJoCo_XML/A00_MuJoCo_XML/A00_robot_mujoco.xml"


def generate(gmr):
    a = mj.MjModel.from_xml_path(str(MODEL))
    ad = mj.MjData(a)
    mj.mj_forward(a, ad)
    g = mj.MjModel.from_xml_path(str(gmr / "assets/unitree_g1/g1_mocap_29dof.xml"))
    gd = mj.MjData(g)
    mj.mj_forward(g, gd)
    mapping = {"pelvis": "base_link", "torso_link": "waist_pitch"}
    for side in ("left", "right"):
        for old, new in (("hip_roll_link", "hip_yaw"), ("hip_yaw_link", "hip_yaw"),
                         ("knee_link", "knee"), ("ankle_roll_link", "ankle_roll"),
                         ("toe_link", "ankle_roll"), ("shoulder_yaw_link", "shoulder_yaw"),
                         ("elbow_link", "elbow"), ("wrist_yaw_link", "wrist_pitch")):
            mapping[f"{side}_{old}"] = f"{side}_{new}"
    # Align each segment's outgoing direction as well as its reference body frame.
    # This removes A00's CAD mounting rotations, including mirrored knee frames.
    children = {}
    for side in ("left", "right"):
        for old, new, gc, ac in (
            ("hip_roll_link", "hip_yaw", "knee_link", "knee"),
            ("hip_yaw_link", "hip_yaw", "knee_link", "knee"),
            ("knee_link", "knee", "ankle_roll_link", "ankle_roll"),
            ("shoulder_yaw_link", "shoulder_yaw", "elbow_link", "elbow"),
            ("elbow_link", "elbow", "wrist_yaw_link", "wrist_pitch"),
            ("wrist_yaw_link", "wrist_pitch", "rubber_hand", "finger"),
        ):
            children[f"{side}_{old}"] = (f"{side}_{gc}", f"{side}_{ac}")
    for source, original in (("smplx", "smplx_to_g1"),
                             ("bvh_lafan1", "bvh_lafan1_to_g1"),
                             ("bvh_nokov", "bvh_nokov_to_g1")):
        config = json.loads((gmr / f"general_motion_retargeting/ik_configs/{original}.json").read_text())
        config["robot_root_name"] = "base_link"
        config["ground_height"] = 0.0
        for name in config["human_scale_table"]:
            config["human_scale_table"][name] = 0.85 if any(s in name.lower() for s in ("arm", "hand", "shoulder", "elbow", "wrist")) else 1.05
        for stage in (1, 2):
            table = {}
            for old, entry in config[f"ik_match_table{stage}"].items():
                new = mapping[old]
                gr = R.from_quat(gd.xquat[g.body(old).id], scalar_first=True)
                ar = R.from_quat(ad.xquat[a.body(new).id], scalar_first=True)
                align = R.identity()
                if old in children:
                    gc, ac = children[old]
                    # The G1 hand is fixed and points forward in its wrist frame.
                    gv = (gd.xpos[g.body(gc).id] - gd.xpos[g.body(old).id]
                          if gc in [g.body(i).name for i in range(g.nbody)]
                          else gr.apply([1., 0., 0.]))
                    av = ad.xpos[a.body(ac).id] - ad.xpos[a.body(new).id]
                    av, gv = av / np.linalg.norm(av), gv / np.linalg.norm(gv)
                    cross = np.cross(av, gv)
                    if np.linalg.norm(cross) > 1e-8:
                        align = R.from_rotvec(cross / np.linalg.norm(cross) * np.arccos(np.clip(av @ gv, -1., 1.)))
                offset = R.from_quat(entry[4], scalar_first=True) * gr.inv() * align * ar
                entry[4] = offset.as_quat(scalar_first=True).round(10).tolist()
                entry[3] = [0., 0., 0.]
                if new == "base_link":
                    entry[1:3] = [100, 20]
                elif "ankle_roll" in new:
                    entry[1:3] = [100, 30]
                elif "waist_pitch" == new:
                    entry[1:3] = [0, 30]
                elif "shoulder" in new:
                    entry[1:3] = [0 if stage == 1 else 5, 15]
                elif "wrist" in new:
                    entry[1:3] = [0 if stage == 1 else 60, 2]
                elif "elbow" in new:
                    entry[1:3] = [0 if stage == 1 else 15, 2]
                else:
                    entry[1:3] = [0 if stage == 1 else 10, 10]
                table[new] = entry
            config[f"ik_match_table{stage}"] = table
        config["foot_bodies"] = ["left_ankle_roll", "right_ankle_roll"]
        config["posture_cost"] = 0.05
        config["morphology_scaling"] = True
        config["position_anchors"] = {}
        config["position_chains"] = []
        for side in ("left", "right"):
            config["position_anchors"][f"{side}_hip_yaw"] = "base_link"
            config["position_anchors"][f"{side}_shoulder_yaw"] = "waist_pitch"
            entries = config["ik_match_table1"]
            for parent, child in (("hip_yaw", "knee"), ("knee", "ankle_roll"),
                                  ("shoulder_yaw", "elbow"), ("elbow", "wrist_pitch")):
                rp, rc = f"{side}_{parent}", f"{side}_{child}"
                config["position_chains"].append([entries[rp][0], entries[rc][0], rp, rc])
        # Hip-yaw origins sit 20 cm below the human hip landmarks; their position
        # is determined by proximal joints, so orientation dominates the task.
        for stage in (1, 2):
            for side in ("left", "right"):
                config[f"ik_match_table{stage}"][f"{side}_hip_yaw"][1] = 0
        config["provenance"] = f"Adapted from local GMR/{original}.json and A00 FK; initial calibration, tune on representative motions."
        (ROOT / f"jlj_gmr/configs/{source}_to_a00.json").write_text(json.dumps(config, indent=2) + "\n")
    names = [a.joint(i).name for i in range(a.njnt) if a.jnt_type[i] == mj.mjtJoint.mjJNT_HINGE]
    (ROOT / "jlj_gmr/configs/joint_order.json").write_text(json.dumps(names, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gmr", type=Path, required=True)
    generate(p.parse_args().gmr)
