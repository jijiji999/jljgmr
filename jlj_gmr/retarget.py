"""Two-stage constrained differential IK, adapted from GMR (LICENSE-GMR).

Targets are world poses in metres with scalar-first (wxyz) quaternions.
"""
import json
from pathlib import Path

import mink
import mujoco as mj
import numpy as np
from qpsolvers import available_solvers
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "A00_MuJoCo_XML/A00_MuJoCo_XML/A00_robot_mujoco.xml"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


class A00Retargeter:
    def __init__(self, src_human="bvh_nokov", *, xml_path=None, config_path=None,
                 actual_human_height=None, fps=60., solver="proxqp", damping=0.05,
                 max_iter=12, ground_mode="initial", max_joint_speed=None):
        if fps <= 0 or max_iter < 1 or damping < 0:
            raise ValueError("fps/max_iter must be positive; damping must be nonnegative")
        if ground_mode not in ("initial", "floor", "none"):
            raise ValueError("ground_mode must be initial, floor, or none")
        if max_joint_speed is not None and max_joint_speed <= 0:
            raise ValueError("max_joint_speed must be positive (rad/s)")
        if solver not in available_solvers:
            raise ValueError(f"QP solver {solver!r} unavailable; installed: {available_solvers}")
        self.xml_path = Path(xml_path or DEFAULT_XML).resolve()
        if not self.xml_path.is_file():
            raise FileNotFoundError(f"A00 XML not found: {self.xml_path}; pass xml_path/--xml")
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        if self.model.jnt_type[0] != mj.mjtJoint.mjJNT_FREE or self.model.nq != self.model.nv + 1:
            raise ValueError("Expected a root freejoint followed by scalar hinge/slide joints")
        self.config_path = Path(config_path or CONFIG_DIR / f"{src_human}_to_a00.json")
        self.config = json.loads(self.config_path.read_text())
        self.human_root_name = self.config["human_root_name"]
        self.robot_root_name = self.config["robot_root_name"]
        self.human_scale_table = dict(self.config["human_scale_table"])
        if actual_human_height is not None:
            if actual_human_height <= 0:
                raise ValueError("actual_human_height must be positive (metres)")
            # Taller subjects need less scaling to the same fixed robot size.
            ratio = self.config["human_height_assumption"] / actual_human_height
            self.human_scale_table = {k: v * ratio for k, v in self.human_scale_table.items()}
        self.dt = 1. / fps
        self.solver, self.damping, self.max_iter = solver, damping, max_iter
        self.ground_mode, self.max_joint_speed = ground_mode, max_joint_speed
        self.ground_offset = None
        self.configuration = mink.Configuration(self.model)
        reference = mj.MjData(self.model)
        mj.mj_forward(self.model, reference)
        self.reference_positions = reference.xpos.copy()
        self.reference_rotations = [R.from_quat(q, scalar_first=True) for q in reference.xquat]
        self.joint_ids = list(range(1, self.model.njnt))
        self.joint_names = [self.model.joint(i).name for i in self.joint_ids]
        self.joint_qpos_indices = self.model.jnt_qposadr[self.joint_ids]
        self.foot_geom_ids = [i for i in range(self.model.ngeom)
                              if self.model.geom(i).name and "foot" in self.model.geom(i).name
                              and self.model.geom_type[i] == mj.mjtGeom.mjGEOM_CAPSULE]
        if not self.foot_geom_ids:
            raise ValueError("No foot collision capsules found; A00 sole geometry is required")
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.stages = []
        self.required_bodies = {self.human_root_name}
        for stage in (1, 2):
            entries = []
            if self.config[f"use_ik_match_table{stage}"]:
                for robot, entry in self.config[f"ik_match_table{stage}"].items():
                    human, pc, rc, pos, quat = entry
                    if not pc and not rc:
                        continue
                    self.model.body(robot)  # fail early with the invalid name
                    if human not in self.human_scale_table:
                        raise ValueError(f"No scale configured for {human}")
                    task = mink.FrameTask(frame_name=robot, frame_type="body",
                                          position_cost=pc, orientation_cost=rc, lm_damping=1.)
                    entries.append((robot, human, task, np.asarray(pos, dtype=float),
                                    R.from_quat(quat, scalar_first=True)))
                    self.required_bodies.add(human)
            if entries:
                self.stages.append(entries)
        if not self.stages:
            raise ValueError("At least one IK stage must contain an active task")
        self.posture = mink.PostureTask(self.model, cost=self.config.get("posture_cost", 0.05))
        self.scaled_human_data = {}
        self.target_stages = []
        self.last_diagnostics = {}
        self.reset()

    def sole_min_z(self):
        """Exact lowest z of the ten A00 foot collision capsules at current qpos."""
        data = self.configuration.data
        values = []
        for i in self.foot_geom_ids:
            half = self.model.geom_size[i, 1]
            radius = self.model.geom_size[i, 0]
            values.append(data.geom_xpos[i, 2] - abs(data.geom_xmat[i, 8]) * half - radius)
        return float(min(values))

    def reset(self, qpos=None):
        q = self.model.qpos0.copy() if qpos is None else np.asarray(qpos, dtype=float).copy()
        if q.shape != (self.model.nq,) or not np.isfinite(q).all() or np.linalg.norm(q[3:7]) < 1e-8:
            raise ValueError(f"qpos must be finite, shape ({self.model.nq},), with a nonzero quaternion")
        q[3:7] /= np.linalg.norm(q[3:7])
        self.configuration.update(q)
        if qpos is None:
            q[2] -= self.sole_min_z()
            self.configuration.update(q)
        self.posture.set_target(q)
        self.ground_offset = None
        self.has_frame = qpos is not None
        self.last_diagnostics = {}

    def update_targets(self, human_data, offset_to_ground=None):
        missing = sorted(self.required_bodies - human_data.keys())
        if missing:
            raise ValueError(f"Missing human bodies: {', '.join(missing)}; check source format/config")
        poses = {}
        for name in self.required_bodies:
            p, q = (np.asarray(x, dtype=float).copy() for x in human_data[name])
            if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
                raise ValueError(f"Invalid pose for {name}: expected finite xyz and nonzero wxyz quaternion")
            poses[name] = (p, R.from_quat(q, scalar_first=True))
        root = poses[self.human_root_name][0]
        scaled_root = root * self.human_scale_table[self.human_root_name]
        targets = []
        for entries in self.stages:
            stage = {}
            for robot, human, task, offset, rot_offset in entries:
                p, rot = poses[human]
                rot = rot * rot_offset
                p = scaled_root + (p - root) * self.human_scale_table[human] + rot.apply(offset)
                stage[robot] = (p, rot)
            if self.config.get("morphology_scaling", False):
                # Keep GMR's pose targets; adapt landmark positions to A00's
                # actual segment lengths instead of scaling arms around pelvis.
                root_id = self.model.body(self.robot_root_name).id
                rp, rr = stage[self.robot_root_name]
                for robot, anchor in self.config["position_anchors"].items():
                    bid, aid = self.model.body(robot).id, self.model.body(anchor).id
                    local = self.reference_rotations[aid].inv().apply(
                        self.reference_positions[bid] - self.reference_positions[aid])
                    if anchor == self.robot_root_name:
                        ap, ar = rp, rr
                    else:
                        # The waist origin is anchored to the pelvis, while its
                        # orientation follows the source chest through GMR offsets.
                        ap = rp + rr.apply(self.reference_positions[aid] - self.reference_positions[root_id])
                        ar = stage[anchor][1]
                    stage[robot] = (ap + ar.apply(local), stage[robot][1])
                for hp, hc, robot_parent, robot_child in self.config["position_chains"]:
                    direction = poses[hc][0] - poses[hp][0]
                    length = np.linalg.norm(direction)
                    if length < 1e-8:
                        raise ValueError(f"Coincident human landmarks: {hp}, {hc}")
                    a, b = self.model.body(robot_parent).id, self.model.body(robot_child).id
                    robot_length = np.linalg.norm(self.reference_positions[b] - self.reference_positions[a])
                    p = stage[robot_parent][0] + direction * (robot_length / length)
                    stage[robot_child] = (p, stage[robot_child][1])
            targets.append(stage)
        mode = self.ground_mode if offset_to_ground is None else ("floor" if offset_to_ground else "none")
        # A foot target is its ankle body origin. Compute its support distance
        # from the actual capsule geometry in the desired orientation.
        lowest = np.inf
        for robot in self.config["foot_bodies"]:
            p, rot = targets[0][robot]
            body_id = self.model.body(robot).id
            for i in self.foot_geom_ids:
                if self.model.geom_bodyid[i] != body_id:
                    continue
                gp = p + rot.apply(self.model.geom_pos[i])
                gr = rot * R.from_quat(self.model.geom_quat[i], scalar_first=True)
                bottom = gp[2] - abs(gr.as_matrix()[2, 2]) * self.model.geom_size[i, 1] - self.model.geom_size[i, 0]
                lowest = min(lowest, bottom)
        if self.ground_offset is None:
            self.ground_offset = float(lowest - self.config.get("ground_height", 0.))
        shift = (lowest - self.config.get("ground_height", 0.) if mode == "floor" else
                 self.ground_offset if mode == "initial" else 0.)
        self.target_stages = targets
        self.scaled_human_data = {}
        for entries, stage in zip(self.stages, targets):
            for robot, human, task, _, _ in entries:
                p, rot = stage[robot]
                p[2] -= shift
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot.as_quat(scalar_first=True)), p))
                self.scaled_human_data[human] = (p.copy(), rot.as_quat(scalar_first=True))
        return targets

    def retarget(self, human_data, offset_to_ground=None):
        previous = self.configuration.q.copy()
        self.update_targets(human_data, offset_to_ground)
        self.posture.set_target(previous)
        iterations = []
        for entries in self.stages:
            tasks = [entry[2] for entry in entries] + [self.posture]
            count = 0
            for _ in range(self.max_iter):
                before = self.configuration.q.copy()
                vel = mink.solve_ik(self.configuration, tasks, self.dt, solver=self.solver,
                                    damping=self.damping, limits=self.limits)
                if not np.isfinite(vel).all():
                    self.configuration.update(previous)
                    raise RuntimeError("QP returned nonfinite joint velocity")
                self.configuration.integrate_inplace(vel, self.dt)
                count += 1
                if np.linalg.norm(self.configuration.q - before) < 1e-5:
                    break
            iterations.append(count)
        q = self.configuration.q.copy()
        if self.max_joint_speed is not None and self.has_frame:
            idx = self.joint_qpos_indices
            step = self.max_joint_speed * self.dt
            q[idx] = previous[idx] + np.clip(q[idx] - previous[idx], -step, step)
        # Numerical tolerance only; all optimization iterations already use limits.
        for j in self.joint_ids:
            if self.model.jnt_limited[j]:
                idx = self.model.jnt_qposadr[j]
                q[idx] = np.clip(q[idx], *self.model.jnt_range[j])
        self.configuration.update(q)
        mode = self.ground_mode if offset_to_ground is None else ("floor" if offset_to_ground else "none")
        penetration = max(0., self.config.get("ground_height", 0.) - self.sole_min_z())
        if mode != "none" and penetration:
            q[2] += penetration
            self.configuration.update(q)
        errors = []
        for robot, _, task, _, _ in self.stages[-1]:
            e = task.compute_error(self.configuration)
            pc, rc = self.config["ik_match_table2" if self.config["use_ik_match_table2"] else "ik_match_table1"][robot][1:3]
            errors.append({"body": robot, "position_cost": pc, "orientation_cost": rc,
                           "position_m": float(np.linalg.norm(e[:3])),
                           "rotation_rad": float(np.linalg.norm(e[3:]))})
        self.last_diagnostics = {"iterations": iterations, "sole_min_z": self.sole_min_z(),
                                 "ground_correction_m": penetration if mode != "none" else 0., "errors": errors}
        self.has_frame = True
        return self.configuration.q.copy()
