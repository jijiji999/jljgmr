"""BVH channel parser and FK; GMR's Y-up/cm -> Z-up/m convention.

Reads each joint's actual channel order and the file's Frame Time.
"""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class HumanMotion:
    frames: list
    fps: float
    height: float | None = None


def load_bvh(path, format="nokov", unit_scale=0.01, height=None):
    if format not in ("nokov", "lafan1") or unit_scale <= 0:
        raise ValueError("format must be nokov/lafan1; unit_scale must be positive")
    text = Path(path).read_text()
    hierarchy, motion = re.split(r"\bMOTION\b", text, maxsplit=1)
    names, parents, offsets, channels = [], [], [], []
    stack, pending = [], None
    for line in hierarchy.splitlines():
        words = line.strip().split()
        if not words:
            continue
        if words[0] in ("ROOT", "JOINT"):
            name = words[1].split(":")[-1]
            if name in names:
                raise ValueError(f"Duplicate BVH joint name after namespace removal: {name}")
            names.append(name)
            parents.append(stack[-1] if stack else -1)
            offsets.append([0., 0., 0.])
            channels.append([])
            pending = len(names) - 1
        elif words[0] == "End":
            pending = -1
        elif words[0] == "{":
            stack.append(pending)
        elif words[0] == "}":
            stack.pop()
        elif words[0] == "OFFSET" and stack[-1] != -1:
            offsets[stack[-1]] = list(map(float, words[1:4]))
        elif words[0] == "CHANNELS":
            channels[stack[-1]] = words[2:]
            if int(words[1]) != len(words[2:]):
                raise ValueError("BVH channel count mismatch")
    frame_count = int(re.search(r"Frames:\s*(\d+)", motion).group(1))
    dt = float(re.search(r"Frame Time:\s*([\d.eE+\-]+)", motion).group(1))
    if frame_count < 1 or dt <= 0:
        raise ValueError("BVH must have at least one frame and positive Frame Time")
    values_text = re.split(r"Frame Time:\s*[\d.eE+\-]+", motion, maxsplit=1)[1]
    values = np.fromstring(values_text, sep=" ")
    nc = sum(map(len, channels))
    if len(values) != frame_count * nc:
        raise ValueError(f"BVH expected {frame_count * nc} channel values, found {len(values)}")
    values = values.reshape(frame_count, nc)
    positions, rotations = [], []
    cursor = 0
    for j, ch in enumerate(channels):
        data = values[:, cursor:cursor + len(ch)]
        cursor += len(ch)
        p = np.tile(offsets[j], (frame_count, 1)).astype(float)
        rot_indices, axes = [], ""
        for k, c in enumerate(ch):
            if c in ("Xposition", "Yposition", "Zposition"):
                # Match GMR/Nokov exports: root position channels are
                # absolute; child translation channels add to their offsets.
                if parents[j] < 0:
                    p[:, "XYZ".index(c[0])] = data[:, k]
                else:
                    p[:, "XYZ".index(c[0])] += data[:, k]
            elif c in ("Xrotation", "Yrotation", "Zrotation"):
                rot_indices.append(k)
                axes += c[0]
            else:
                raise ValueError(f"Unsupported BVH channel {c}")
        local = R.from_euler(axes, data[:, rot_indices], degrees=True) if axes else R.identity(frame_count)
        if parents[j] >= 0:
            parent = parents[j]
            p = positions[parent] + rotations[parent].apply(p)
            local = rotations[parent] * local
        positions.append(p)
        rotations.append(local)
    world = R.from_matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    positions = np.stack([world.apply(p) * unit_scale for p in positions], axis=1)
    quats = np.stack([(world * q).as_quat(scalar_first=True) for q in rotations], axis=1)
    frames = []
    toe = "ToeBase" if format == "nokov" else "Toe"
    required = [f"{side}{bone}" for side in ("Left", "Right") for bone in ("Foot", toe)]
    missing = set(required) - set(names)
    if missing:
        raise ValueError(f"BVH {format} missing joints: {sorted(missing)}")
    for i in range(frame_count):
        frame = {n: (positions[i, j].copy(), quats[i, j].copy()) for j, n in enumerate(names)}
        for side in ("Left", "Right"):
            frame[f"{side}FootMod"] = (frame[f"{side}Foot"][0].copy(), frame[f"{side}{toe}"][1].copy())
        frames.append(frame)
    return HumanMotion(frames, 1. / dt, height)
