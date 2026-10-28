"""Kinematic playback and offscreen rendering. No robot hardware connection."""
import os
import time
from pathlib import Path
import numpy as np


def _scene(xml):
    import mujoco as mj
    spec = mj.MjSpec.from_file(str(xml))
    spec.worldbody.add_geom(name="retarget_floor", type=mj.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, .05], rgba=[.24, .27, .31, 1])
    spec.worldbody.add_light(pos=[2, -2, 4], dir=[-1, 1, -2], diffuse=[.7, .7, .7])
    return spec.compile()


def _check(model, qpos, names):
    actual = [model.joint(i).name for i in range(1, model.njnt)]
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not len(qpos) or not np.isfinite(qpos).all():
        raise ValueError(f"Motion must be finite, nonempty, shape (T,{model.nq})")
    if names is not None and list(names) != actual:
        raise ValueError("Motion joint order differs from the XML joint order")


def view_motion(xml, qpos, fps, names, loop=False):
    import mujoco as mj
    import mujoco.viewer
    model = _scene(xml)
    _check(model, qpos, names)
    data = mj.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[:] = [1, 1, 0, 0, 0, 0]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3., 135., -15.
        while viewer.is_running():
            for q in qpos:
                start = time.monotonic()
                if not viewer.is_running():
                    return
                data.qpos[:] = q
                mj.mj_forward(model, data)
                viewer.cam.lookat[:] = q[:3] + [0, 0, -.1]
                viewer.sync()
                time.sleep(max(0., 1. / fps - (time.monotonic() - start)))
            if not loop:
                return


def render_motion(xml, qpos, fps, names, output, frame=0, width=960, height=720):
    # The render command imports this before MuJoCo when used through the CLI.
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco as mj
    import imageio.v2 as imageio
    model = _scene(xml)
    _check(model, qpos, names)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mj.MjData(model)
    opt = mj.MjvOption()
    opt.geomgroup[:] = [1, 1, 0, 0, 0, 0]
    cam = mj.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.0, 135., -15.
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with mj.Renderer(model, height=height, width=width) as renderer:
        def draw(q):
            data.qpos[:] = q
            mj.mj_forward(model, data)
            cam.lookat[:] = q[:3] + [0, 0, -.1]
            renderer.update_scene(data, camera=cam, scene_option=opt)
            return renderer.render().copy()
        if output.suffix.lower() == ".png":
            if not 0 <= frame < len(qpos):
                raise ValueError(f"frame must be in [0,{len(qpos) - 1}]")
            imageio.imwrite(output, draw(qpos[frame]))
        elif output.suffix.lower() == ".mp4":
            with imageio.get_writer(output, fps=fps, codec="libx264", quality=8) as writer:
                for q in qpos:
                    writer.append_data(draw(q))
        else:
            raise ValueError("Render output must be .png or .mp4")
