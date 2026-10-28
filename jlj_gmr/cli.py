import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description="A00 whole-body GMR retargeting")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Validate the XML and list joints/limits")
    inspect.add_argument("--xml", type=Path)
    retarget = commands.add_parser("retarget", help="Convert human motion to A00")
    retarget.add_argument("--input", type=Path, required=True)
    retarget.add_argument("--format", choices=["nokov", "lafan1", "smplx", "poses"], default="nokov")
    retarget.add_argument("--source", choices=["bvh_nokov", "bvh_lafan1", "smplx"], help="Required for generic pose npz")
    retarget.add_argument("--output", type=Path, required=True)
    retarget.add_argument("--xml", type=Path)
    retarget.add_argument("--config", type=Path)
    retarget.add_argument("--body-model", type=Path, help="SMPL-X model folder")
    retarget.add_argument("--human-height", type=float, help="Subject height in metres; default uses config assumption")
    retarget.add_argument("--fps", type=float, help="Output fps; resamples positions and rotations")
    retarget.add_argument("--unit-scale", type=float, default=.01, help="BVH length units to metres; default cm")
    retarget.add_argument("--start-frame", type=int, default=0)
    retarget.add_argument("--max-frames", type=int)
    retarget.add_argument("--max-iter", type=int, default=12)
    retarget.add_argument("--solver", default="proxqp")
    retarget.add_argument("--ground-mode", choices=["initial", "floor", "none"], default="initial")
    retarget.add_argument("--max-joint-speed", type=float, help="Optional output joint speed bound in rad/s")
    for command in ("view", "render"):
        p = commands.add_parser(command)
        p.add_argument("--motion", type=Path, required=True)
        p.add_argument("--xml", type=Path)
        if command == "view":
            p.add_argument("--loop", action="store_true")
        else:
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    if args.command == "render":
        os.environ.setdefault("MUJOCO_GL", "egl")
    from .retarget import A00Retargeter, DEFAULT_XML
    import numpy as np
    from .motion import load_robot_motion, save_motion, resample
    if args.command == "inspect":
        r = A00Retargeter(xml_path=args.xml)
        print(f"XML: {r.xml_path}\nnq={r.model.nq}, nv={r.model.nv}, actuators={r.model.nu}")
        print("qpos: root xyz + root wxyz + 29 joints in this order (rad):")
        for j in r.joint_ids:
            print(f"{r.model.jnt_qposadr[j]:2d}  {r.model.joint(j).name:25s} {r.model.jnt_range[j]}")
        return
    if args.command in ("view", "render"):
        q, fps, names = load_robot_motion(args.motion)
        from .viewer import view_motion, render_motion
        if args.command == "view":
            view_motion(args.xml or DEFAULT_XML, q, fps, names, args.loop)
        else:
            render_motion(args.xml or DEFAULT_XML, q, fps, names, args.output, args.frame)
            print(f"Saved {args.output}")
        return
    if args.output.suffix not in (".npz", ".pkl", ".csv"):
        parser.error("--output must end with .npz/.pkl/.csv")
    if args.start_frame < 0 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("--start-frame must be nonnegative; --max-frames must be positive")
    source = args.source or ("smplx" if args.format == "smplx" else f"bvh_{args.format}")
    if args.format == "poses":
        if not args.source:
            parser.error("--format poses requires --source")
        from .motion import load_pose_npz
        motion = load_pose_npz(args.input)
    elif args.format == "smplx":
        if args.body_model is None:
            parser.error("--format smplx requires --body-model")
        from .smpl import load_smplx
        motion = load_smplx(args.input, args.body_model, args.human_height)
    else:
        from .bvh import load_bvh
        motion = load_bvh(args.input, args.format, args.unit_scale, args.human_height)
    original_fps = motion.fps
    stop = args.start_frame + args.max_frames if args.max_frames else None
    motion.frames = motion.frames[args.start_frame:stop]
    if not motion.frames:
        parser.error("Selected frame range is empty")
    if args.fps is not None:
        motion = resample(motion, args.fps)
    r = A00Retargeter(source, xml_path=args.xml, config_path=args.config,
                      actual_human_height=args.human_height or motion.height, fps=motion.fps,
                      solver=args.solver, max_iter=args.max_iter, ground_mode=args.ground_mode,
                      max_joint_speed=args.max_joint_speed)
    output, diagnostics = [], []
    start = time.monotonic()
    for i, frame in enumerate(motion.frames):
        output.append(r.retarget(frame))
        diagnostics.append(r.last_diagnostics)
        if (i + 1) % 300 == 0:
            print(f"Retargeted {i + 1}/{len(motion.frames)} frames", flush=True)
    elapsed = time.monotonic() - start
    error_names = [e["body"] for e in diagnostics[0]["errors"]]
    errors = np.array([[e["position_m"] for e in d["errors"]] for d in diagnostics])
    q = np.asarray(output)
    near_limit = {}
    for j in r.joint_ids:
        if r.model.jnt_limited[j]:
            v = q[:, r.model.jnt_qposadr[j]]
            lo, hi = r.model.jnt_range[j]
            count = int(np.sum((v - lo < .01) | (hi - v < .01)))
            if count:
                near_limit[r.model.joint(j).name] = count
    summary = {"frames": len(q), "source_fps": original_fps, "fps": motion.fps,
               "elapsed_seconds": elapsed, "processing_fps": len(q) / elapsed,
               "min_sole_z_m": min(d["sole_min_z"] for d in diagnostics),
               "max_ground_correction_m": max(d["ground_correction_m"] for d in diagnostics),
               "mean_tracked_position_error_m": {n: float(v) for n, v, e in zip(error_names, errors.mean(axis=0), diagnostics[0]["errors"]) if e["position_cost"] > 0},
               "max_joint_step_rad": float(np.max(np.abs(np.diff(q[:, 7:], axis=0)))) if len(q) > 1 else 0.,
               "near_joint_limit_frame_counts": near_limit}
    metadata = {"input": str(args.input.resolve()), "xml": str(r.xml_path), "source": source,
                "config": str(r.config_path.resolve()), "ik_config": r.config, "human_height": args.human_height or motion.height,
                "start_source_frame": args.start_frame, "ground_mode": args.ground_mode,
                "max_joint_speed": args.max_joint_speed, "solver": args.solver, "summary": summary}
    save_motion(args.output, q, motion.fps, r.joint_names, metadata)
    report = args.output.with_suffix(args.output.suffix + ".report.json")
    report.write_text(json.dumps({"summary": summary, "frames": diagnostics}, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved {args.output}\nReport {report}")
