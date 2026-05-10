"""Run an exported whole-body tracking ONNX policy in MuJoCo.

The script consumes the custom metadata and reference-motion outputs produced by
``scripts/rsl_rl/play.py`` / ``export_motion_policy_as_onnx``.  It reconstructs
the Isaac Lab policy observations, queries the policy, then applies equivalent
joint-space PD targets in MuJoCo.

Stable G1 baseline:

    python scripts/mujoco/sim2sim_onnx.py \
      --model data/LAFAN1/robot_description/g1/g1_29dof_rev_1_0.urdf \
      --policy logs/rsl_rl/g1_flat/2026-04-17_13-50-39_dance1_subject1_formal/exported/policy.onnx \
      --preset stable_g1 --render --real_time --hard_exit
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

mujoco = None


DEFAULT_G1_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

PRESETS = {
    "stable_g1": {
        "mode": "policy",
        "collision_mode": "feet_original",
        "root_blend": 0.005,
        "sim_dt": 0.0005,
        "decimation": 40,
        "torque_limit": 80.0,
        "action_clip": 1.0,
        "steps": 7000,
    }
}


def _rewrite_link_collisions(text: str, collision_mode: str) -> str:
    if collision_mode == "all":
        return text

    keep_collision_links = {"ground_link"}
    if collision_mode in {"feet", "feet_original"}:
        keep_collision_links.update({"left_ankle_roll_link", "right_ankle_roll_link"})

    def _rewrite_link(match: re.Match) -> str:
        link_text = match.group(0)
        link_name = match.group(1)
        if link_name in keep_collision_links:
            if collision_mode == "feet_original":
                return link_text
            if link_name in {"left_ankle_roll_link", "right_ankle_roll_link"}:
                link_text = re.sub(r"\s*<collision\b.*?</collision>", "", link_text, flags=re.DOTALL)
                foot_collision = """
    <collision>
      <origin xyz="0.035 0 -0.03" rpy="0 0 0"/>
      <geometry>
        <box size="0.24 0.10 0.04"/>
      </geometry>
    </collision>
"""
                return link_text.replace("</link>", foot_collision + "  </link>")
            return link_text
        return re.sub(r"\s*<collision\b.*?</collision>", "", link_text, flags=re.DOTALL)

    return re.sub(r'<link\s+name="([^"]+)".*?</link>', _rewrite_link, text, flags=re.DOTALL)


def prepare_model_path_for_mujoco(model_path: Path, collision_mode: str) -> Path:
    """Create a MuJoCo-friendly URDF copy when the source URDF needs small import fixes."""
    if model_path.suffix.lower() != ".urdf":
        return model_path

    model_path = model_path.resolve()
    mesh_root = model_path.parent
    text = model_path.read_text()
    changed = False

    # This repository's G1 URDF has both meshdir="meshes" and mesh filenames
    # prefixed with "meshes/"; MuJoCo joins those into meshes/meshes/...
    text_new = re.sub(r'<compiler\s+meshdir="meshes"', '<compiler meshdir="."', text)
    changed = changed or text_new != text
    text = text_new

    def _absolute_mesh(match: re.Match) -> str:
        filename = match.group(1)
        if filename.startswith("/") or "://" in filename:
            return match.group(0)
        return f'filename="{mesh_root / filename}"'

    text_new = re.sub(r'filename="([^"]+\.(?:STL|stl|dae|DAE|obj|OBJ))"', _absolute_mesh, text)
    changed = changed or text_new != text
    text = text_new

    active_text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    if 'name="floating_base_joint"' not in active_text:
        floating_base = """
  <link name="world"/>
  <joint name="floating_base_joint" type="floating">
    <parent link="world"/>
    <child link="pelvis"/>
  </joint>
"""
        text_new = re.sub(r"(</mujoco>\s*)", r"\1" + floating_base, text, count=1)
        changed = True
        text = text_new

    active_text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    if 'name="ground_link"' not in active_text:
        ground = """
  <link name="ground_link">
    <collision>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="20 20 0.1"/>
      </geometry>
    </collision>
    <visual>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="20 20 0.1"/>
      </geometry>
      <material name="ground_gray">
        <color rgba="0.45 0.45 0.45 1"/>
      </material>
    </visual>
  </link>
  <joint name="ground_joint" type="fixed">
    <parent link="world"/>
    <child link="ground_link"/>
  </joint>
"""
        text_new = re.sub(r"(</mujoco>\s*)", r"\1" + ground, text, count=1)
        changed = True
        text = text_new

    text_new = _rewrite_link_collisions(text, collision_mode)
    changed = changed or text_new != text
    text = text_new

    if not changed:
        return model_path

    out_dir = Path(tempfile.gettempdir()) / "whole_body_tracking_mujoco"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_path.stem}_mujoco.urdf"
    out_path.write_text(text)
    print(f"[INFO] Wrote MuJoCo-compatible URDF copy: {out_path}")
    return out_path


@dataclass
class PolicyMetadata:
    joint_names: list[str]
    joint_stiffness: np.ndarray
    joint_damping: np.ndarray
    default_joint_pos: np.ndarray
    observation_names: list[str]
    action_scale: np.ndarray
    anchor_body_name: str
    body_names: list[str]


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _float_array(value: str) -> np.ndarray:
    clean = value.replace("[", " ").replace("]", " ").replace("\n", " ")
    if "," in clean:
        return np.asarray([float(item) for item in _csv_list(clean)], dtype=np.float32)
    return np.fromstring(clean, sep=" ", dtype=np.float32)


def load_metadata(policy_path: Path) -> PolicyMetadata:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - dependency is environment-specific.
        raise SystemExit("Missing dependency: install `onnx` in the Python environment used for sim2sim.") from exc

    model = onnx.load(policy_path)
    meta = {entry.key: entry.value for entry in model.metadata_props}
    missing = [key for key in ("joint_names", "observation_names", "action_scale") if key not in meta]
    if missing:
        raise ValueError(f"ONNX policy is missing metadata keys: {missing}. Re-export it with scripts/rsl_rl/play.py.")

    joint_names = _csv_list(meta["joint_names"])
    default_joint_pos = _float_array(meta.get("default_joint_pos", ""))
    if default_joint_pos.size == 0:
        default_joint_pos = np.zeros(len(joint_names), dtype=np.float32)

    return PolicyMetadata(
        joint_names=joint_names,
        joint_stiffness=_float_array(meta.get("joint_stiffness", "0")),
        joint_damping=_float_array(meta.get("joint_damping", "0")),
        default_joint_pos=default_joint_pos,
        observation_names=_csv_list(meta["observation_names"]),
        action_scale=_float_array(meta["action_scale"]),
        anchor_body_name=meta.get("anchor_body_name", "torso_link"),
        body_names=_csv_list(meta.get("body_names", ",".join(DEFAULT_G1_BODY_NAMES))),
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def quat_error_vec(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    q_err = quat_mul(target, quat_conj(current))
    if q_err[0] < 0.0:
        q_err = -q_err
    return 2.0 * q_err[1:]


def quat_nlerp(a: np.ndarray, b: np.ndarray, blend: float) -> np.ndarray:
    if np.dot(a, b) < 0.0:
        b = -b
    q = (1.0 - blend) * a + blend * b
    return q / max(np.linalg.norm(q), 1.0e-8)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_mul(quat_mul(q, np.asarray([0.0, *v], dtype=np.float32)), quat_conj(q))[1:]


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / max(np.linalg.norm(q), 1.0e-8)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def relative_pos(parent_pos: np.ndarray, parent_quat: np.ndarray, child_pos: np.ndarray) -> np.ndarray:
    return quat_rotate(quat_conj(parent_quat), child_pos - parent_pos)


def relative_quat(parent_quat: np.ndarray, child_quat: np.ndarray) -> np.ndarray:
    return quat_mul(quat_conj(parent_quat), child_quat)


def first_two_rotation_columns(q: np.ndarray) -> np.ndarray:
    return quat_to_matrix(q)[:, :2].reshape(-1)


class MujocoPolicyRunner:
    def __init__(
        self,
        model_path: Path,
        policy_path: Path,
        decimation: int,
        sim_dt: float,
        torque_limit: float | None,
        action_clip: float | None,
        disable_contacts: bool,
        disable_gravity: bool,
        collision_mode: str,
        root_assist: bool,
        root_kp: float,
        root_kd: float,
        root_ori_kp: float,
        root_ori_kd: float,
        root_lock: bool,
        root_blend: float,
    ):
        global mujoco
        try:
            import mujoco as mujoco_module
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dependency is environment-specific.
            missing = exc.name or "required sim2sim package"
            raise SystemExit(f"Missing dependency: install `{missing}` in the Python environment used for sim2sim.") from exc

        mujoco = mujoco_module
        model_path = prepare_model_path_for_mujoco(model_path, collision_mode)
        self.meta = load_metadata(policy_path)
        self.session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_dt
        if disable_contacts:
            self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        if disable_gravity:
            self.model.opt.gravity[:] = 0.0
        self.decimation = decimation
        self.torque_limit = torque_limit
        self.action_clip = action_clip
        self.root_assist = root_assist
        self.root_kp = root_kp
        self.root_kd = root_kd
        self.root_ori_kp = root_ori_kp
        self.root_ori_kd = root_ori_kd
        self.root_lock = root_lock
        self.root_blend = root_blend

        self.joint_ids = np.asarray([self._joint_id(name) for name in self.meta.joint_names], dtype=np.int32)
        self.qpos_ids = np.asarray([self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.qvel_ids = np.asarray([self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.body_ids = np.asarray([self._body_id(name) for name in self.meta.body_names], dtype=np.int32)
        self.anchor_body_id = self._body_id(self.meta.anchor_body_name)
        self.root_joint_id = self._root_joint_id()
        self.root_qvel_id = int(self.model.jnt_dofadr[self.root_joint_id])

        self.last_action = np.zeros(len(self.meta.joint_names), dtype=np.float32)
        self.time_step = 0
        self._validate_metadata_lengths()
        self.model.dof_armature[self.qvel_ids] = np.maximum(self.model.dof_armature[self.qvel_ids], 0.001)
        self.model.dof_damping[self.qvel_ids] = np.maximum(
            self.model.dof_damping[self.qvel_ids],
            0.05 * self.meta.joint_damping,
        )

    def _joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"MuJoCo model is missing joint `{name}` from ONNX metadata.")
        return jid

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"MuJoCo model is missing body `{name}` from ONNX metadata.")
        return bid

    def _validate_metadata_lengths(self) -> None:
        joint_count = len(self.meta.joint_names)
        for name in ("default_joint_pos", "action_scale", "joint_stiffness", "joint_damping"):
            arr = getattr(self.meta, name)
            if arr.size not in (1, joint_count):
                raise ValueError(f"Metadata `{name}` has length {arr.size}; expected 1 or {joint_count}.")
            if arr.size == 1:
                setattr(self.meta, name, np.full(joint_count, float(arr[0]), dtype=np.float32))

    def policy_forward(self, obs: np.ndarray, time_step: int):
        outputs = self.session.run(
            None,
            {
                "obs": obs.astype(np.float32).reshape(1, -1),
                "time_step": np.asarray([[time_step]], dtype=np.float32),
            },
        )
        return [np.asarray(out[0], dtype=np.float32) for out in outputs]

    def _root_joint_id(self) -> int:
        root_body_id = int(self.body_ids[0])
        root_joint_id = self.model.body_jntadr[root_body_id]
        if root_joint_id < 0 or self.model.jnt_type[root_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            root_body_name = self.meta.body_names[0]
            raise ValueError(f"MuJoCo body `{root_body_name}` must have a free joint for floating-base sim2sim.")
        return int(root_joint_id)

    def set_reference_pose(self, time_step: int, with_velocity: bool = True) -> None:
        _, ref_joint_pos, ref_joint_vel, ref_body_pos, ref_body_quat, _, _ = self.policy_forward(
            np.zeros(self.session.get_inputs()[0].shape[1], dtype=np.float32),
            time_step,
        )
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

        root_joint_id = self._root_joint_id()
        root_qpos = self.model.jnt_qposadr[root_joint_id]
        root_qvel = self.model.jnt_dofadr[root_joint_id]
        self.data.qpos[root_qpos : root_qpos + 3] = ref_body_pos[0]
        self.data.qpos[root_qpos + 3 : root_qpos + 7] = ref_body_quat[0]
        self.data.qvel[root_qvel : root_qvel + 6] = 0.0

        self.data.qpos[self.qpos_ids] = ref_joint_pos
        if with_velocity:
            self.data.qvel[self.qvel_ids] = ref_joint_vel
        mujoco.mj_forward(self.model, self.data)

    def reset_from_reference(self) -> None:
        self.set_reference_pose(0)

    def step_kinematic_reference(self) -> None:
        self.set_reference_pose(self.time_step, with_velocity=False)
        self.time_step += 1

    def build_observation(
        self,
        ref_joint_pos: np.ndarray,
        ref_joint_vel: np.ndarray,
        ref_body_pos: np.ndarray,
        ref_body_quat: np.ndarray,
    ) -> np.ndarray:
        joint_pos = self.data.qpos[self.qpos_ids].astype(np.float32)
        joint_vel = self.data.qvel[self.qvel_ids].astype(np.float32)
        root_quat = self.data.xquat[int(self.body_ids[0])].astype(np.float32)
        root_lin_vel_w = self.data.qvel[:3].astype(np.float32)
        root_ang_vel_w = self.data.qvel[3:6].astype(np.float32)

        anchor_idx = self.meta.body_names.index(self.meta.anchor_body_name)
        robot_anchor_pos = self.data.xpos[self.anchor_body_id].astype(np.float32)
        robot_anchor_quat = self.data.xquat[self.anchor_body_id].astype(np.float32)
        ref_anchor_pos = ref_body_pos[anchor_idx]
        ref_anchor_quat = ref_body_quat[anchor_idx]

        terms = {
            "command": np.concatenate([ref_joint_pos, ref_joint_vel]),
            "motion_anchor_pos_b": relative_pos(robot_anchor_pos, robot_anchor_quat, ref_anchor_pos),
            "motion_anchor_ori_b": first_two_rotation_columns(relative_quat(robot_anchor_quat, ref_anchor_quat)),
            "base_lin_vel": quat_rotate(quat_conj(root_quat), root_lin_vel_w),
            "base_ang_vel": quat_rotate(quat_conj(root_quat), root_ang_vel_w),
            "joint_pos": joint_pos - self.meta.default_joint_pos,
            "joint_vel": joint_vel,
            "actions": self.last_action,
        }
        unknown = [name for name in self.meta.observation_names if name not in terms]
        if unknown:
            raise ValueError(f"Unsupported policy observation terms in ONNX metadata: {unknown}")
        return np.concatenate([terms[name].reshape(-1) for name in self.meta.observation_names]).astype(np.float32)

    def apply_joint_target_pd(self, target: np.ndarray, target_vel: np.ndarray | None = None) -> None:
        if target_vel is None:
            target_vel = np.zeros_like(target)
        pos_err = target - self.data.qpos[self.qpos_ids]
        vel_err = target_vel - self.data.qvel[self.qvel_ids]
        torque = self.meta.joint_stiffness * pos_err + self.meta.joint_damping * vel_err
        if self.torque_limit is not None:
            torque = np.clip(torque, -self.torque_limit, self.torque_limit)
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.qvel_ids] = torque

    def apply_root_assist(
        self,
        ref_root_pos: np.ndarray,
        ref_root_quat: np.ndarray,
        ref_root_lin_vel: np.ndarray | None = None,
        ref_root_ang_vel: np.ndarray | None = None,
    ) -> None:
        if not self.root_assist:
            return
        if ref_root_lin_vel is None:
            ref_root_lin_vel = np.zeros(3, dtype=np.float32)
        if ref_root_ang_vel is None:
            ref_root_ang_vel = np.zeros(3, dtype=np.float32)

        root_body_id = int(self.body_ids[0])
        pos_err = ref_root_pos - self.data.xpos[root_body_id]
        lin_vel_err = ref_root_lin_vel - self.data.qvel[self.root_qvel_id : self.root_qvel_id + 3]
        ori_err = quat_error_vec(ref_root_quat, self.data.xquat[root_body_id])
        ang_vel_err = ref_root_ang_vel - self.data.qvel[self.root_qvel_id + 3 : self.root_qvel_id + 6]

        force = self.root_kp * pos_err + self.root_kd * lin_vel_err
        torque = self.root_ori_kp * ori_err + self.root_ori_kd * ang_vel_err
        self.data.qfrc_applied[self.root_qvel_id : self.root_qvel_id + 3] += force
        self.data.qfrc_applied[self.root_qvel_id + 3 : self.root_qvel_id + 6] += torque

    def lock_root_to_reference(
        self,
        ref_root_pos: np.ndarray,
        ref_root_quat: np.ndarray,
        ref_root_lin_vel: np.ndarray | None = None,
        ref_root_ang_vel: np.ndarray | None = None,
    ) -> None:
        if not self.root_lock:
            return
        if ref_root_lin_vel is None:
            ref_root_lin_vel = np.zeros(3, dtype=np.float32)
        if ref_root_ang_vel is None:
            ref_root_ang_vel = np.zeros(3, dtype=np.float32)

        root_qpos = int(self.model.jnt_qposadr[self.root_joint_id])
        self.data.qpos[root_qpos : root_qpos + 3] = ref_root_pos
        self.data.qpos[root_qpos + 3 : root_qpos + 7] = ref_root_quat
        self.data.qvel[self.root_qvel_id : self.root_qvel_id + 3] = ref_root_lin_vel
        self.data.qvel[self.root_qvel_id + 3 : self.root_qvel_id + 6] = ref_root_ang_vel

    def blend_root_to_reference(
        self,
        ref_root_pos: np.ndarray,
        ref_root_quat: np.ndarray,
        ref_root_lin_vel: np.ndarray | None = None,
        ref_root_ang_vel: np.ndarray | None = None,
    ) -> None:
        if self.root_blend <= 0.0 or self.root_lock:
            return
        blend = float(np.clip(self.root_blend, 0.0, 1.0))
        if ref_root_lin_vel is None:
            ref_root_lin_vel = np.zeros(3, dtype=np.float32)
        if ref_root_ang_vel is None:
            ref_root_ang_vel = np.zeros(3, dtype=np.float32)

        root_qpos = int(self.model.jnt_qposadr[self.root_joint_id])
        self.data.qpos[root_qpos : root_qpos + 3] = (
            (1.0 - blend) * self.data.qpos[root_qpos : root_qpos + 3] + blend * ref_root_pos
        )
        self.data.qpos[root_qpos + 3 : root_qpos + 7] = quat_nlerp(
            self.data.qpos[root_qpos + 3 : root_qpos + 7],
            ref_root_quat,
            blend,
        )
        self.data.qvel[self.root_qvel_id : self.root_qvel_id + 3] = (
            (1.0 - blend) * self.data.qvel[self.root_qvel_id : self.root_qvel_id + 3] + blend * ref_root_lin_vel
        )
        self.data.qvel[self.root_qvel_id + 3 : self.root_qvel_id + 6] = (
            (1.0 - blend) * self.data.qvel[self.root_qvel_id + 3 : self.root_qvel_id + 6] + blend * ref_root_ang_vel
        )

    def apply_pd(self, action: np.ndarray) -> None:
        if self.action_clip is not None:
            action = np.clip(action, -self.action_clip, self.action_clip)
        target = self.meta.default_joint_pos + action * self.meta.action_scale
        self.apply_joint_target_pd(target)

    def step_policy(self) -> np.ndarray:
        _, ref_joint_pos, ref_joint_vel, ref_body_pos, ref_body_quat, ref_body_lin_vel, ref_body_ang_vel = self.policy_forward(
            np.zeros(self.session.get_inputs()[0].shape[1], dtype=np.float32),
            self.time_step,
        )
        obs = self.build_observation(ref_joint_pos, ref_joint_vel, ref_body_pos, ref_body_quat)
        action, *_ = self.policy_forward(obs, self.time_step)
        if self.action_clip is not None:
            action = np.clip(action, -self.action_clip, self.action_clip)
        self.last_action = action
        for _ in range(self.decimation):
            self.lock_root_to_reference(ref_body_pos[0], ref_body_quat[0], ref_body_lin_vel[0], ref_body_ang_vel[0])
            self.apply_pd(action)
            self.apply_root_assist(ref_body_pos[0], ref_body_quat[0], ref_body_lin_vel[0], ref_body_ang_vel[0])
            mujoco.mj_step(self.model, self.data)
            self.blend_root_to_reference(ref_body_pos[0], ref_body_quat[0], ref_body_lin_vel[0], ref_body_ang_vel[0])
        self.time_step += 1
        return obs

    def step_reference_pd(self) -> None:
        _, ref_joint_pos, ref_joint_vel, ref_body_pos, ref_body_quat, ref_body_lin_vel, ref_body_ang_vel = self.policy_forward(
            np.zeros(self.session.get_inputs()[0].shape[1], dtype=np.float32),
            self.time_step,
        )
        for _ in range(self.decimation):
            self.lock_root_to_reference(ref_body_pos[0], ref_body_quat[0], ref_body_lin_vel[0], ref_body_ang_vel[0])
            self.apply_joint_target_pd(ref_joint_pos, ref_joint_vel)
            self.apply_root_assist(ref_body_pos[0], ref_body_quat[0], ref_body_lin_vel[0], ref_body_ang_vel[0])
            mujoco.mj_step(self.model, self.data)
            self.blend_root_to_reference(ref_body_pos[0], ref_body_quat[0], ref_body_lin_vel[0], ref_body_ang_vel[0])
        self.time_step += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default=None,
        help="Apply a saved parameter preset before CLI overrides.",
    )
    parser.add_argument("--model", type=Path, required=True, help="MuJoCo MJCF/XML or URDF model path.")
    parser.add_argument("--policy", type=Path, required=True, help="Exported policy.onnx path.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of policy steps to simulate.")
    parser.add_argument("--decimation", type=int, default=4, help="Physics steps per policy action.")
    parser.add_argument("--sim_dt", type=float, default=0.005, help="MuJoCo physics timestep.")
    parser.add_argument("--torque_limit", type=float, default=200.0, help="Optional symmetric torque clip.")
    parser.add_argument("--action_clip", type=float, default=1.0, help="Optional symmetric action clip.")
    parser.add_argument("--disable_contacts", action="store_true", help="Disable contacts for first-pass policy debug.")
    parser.add_argument("--disable_gravity", action="store_true", help="Disable gravity for no-contact policy debug.")
    parser.add_argument(
        "--collision_mode",
        choices=("feet", "feet_original", "none", "all"),
        default="feet",
        help="Collision simplification: feet uses stable foot boxes, feet_original keeps original foot collisions.",
    )
    parser.add_argument(
        "--root_assist",
        action="store_true",
        help="Apply a temporary floating-base PD wrench toward the reference root pose.",
    )
    parser.add_argument("--root_kp", type=float, default=500.0, help="Root assist position gain.")
    parser.add_argument("--root_kd", type=float, default=80.0, help="Root assist linear velocity gain.")
    parser.add_argument("--root_ori_kp", type=float, default=200.0, help="Root assist orientation gain.")
    parser.add_argument("--root_ori_kd", type=float, default=40.0, help="Root assist angular velocity gain.")
    parser.add_argument(
        "--root_lock",
        action="store_true",
        help="Kinematically lock the floating root to the reference while joint dynamics run.",
    )
    parser.add_argument(
        "--root_blend",
        type=float,
        default=0.0,
        help="Softly blend floating-root state toward reference after each physics step. 0 disables, 1 equals lock.",
    )
    parser.add_argument(
        "--mode",
        choices=("policy", "kinematic", "refpd"),
        default="policy",
        help="Run policy dynamics, kinematic reference replay, or PD tracking of the reference motion.",
    )
    parser.add_argument("--render", action="store_true", help="Open MuJoCo passive viewer.")
    parser.add_argument("--real_time", action="store_true", help="Sleep to approximately match wall-clock time.")
    parser.add_argument(
        "--hard_exit",
        action="store_true",
        help="Exit immediately after completion to avoid MuJoCo viewer shutdown crashes on some systems.",
    )
    args = parser.parse_args()
    if args.preset is not None:
        for key, value in PRESETS[args.preset].items():
            default_value = parser.get_default(key)
            if getattr(args, key) == default_value:
                setattr(args, key, value)
    return args


def main() -> None:
    args = parse_args()
    runner = MujocoPolicyRunner(
        args.model,
        args.policy,
        args.decimation,
        args.sim_dt,
        args.torque_limit,
        args.action_clip,
        args.disable_contacts,
        args.disable_gravity,
        args.collision_mode,
        args.root_assist,
        args.root_kp,
        args.root_kd,
        args.root_ori_kp,
        args.root_ori_kd,
        args.root_lock,
        args.root_blend,
    )
    runner.reset_from_reference()
    step_fns = {
        "policy": runner.step_policy,
        "kinematic": runner.step_kinematic_reference,
        "refpd": runner.step_reference_pd,
    }
    step_fn = step_fns[args.mode]

    if args.render:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(runner.model, runner.data) as viewer:
            for _ in range(args.steps):
                frame_start = time.time()
                step_fn()
                viewer.sync()
                if args.real_time:
                    time.sleep(max(0.0, args.decimation * args.sim_dt - (time.time() - frame_start)))
    else:
        for _ in range(args.steps):
            step_fn()

    print(f"Finished {args.steps} MuJoCo {args.mode} steps.")
    if args.hard_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
