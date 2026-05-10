# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import copy
import torch

import onnx
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnv

from whole_body_tracking.tasks.tracking.mdp import MotionCommand


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(torch.nn.Module):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor = self._resolve_actor(actor_critic)
        self.uses_tensordict_obs = hasattr(self.actor, "obs_groups")
        self.obs_group_name = None
        if self.uses_tensordict_obs:
            if len(self.actor.obs_groups) != 1:
                raise ValueError(f"ONNX export only supports one actor obs group, got {self.actor.obs_groups}.")
            self.obs_group_name = self.actor.obs_groups[0]
            self.normalizer = torch.nn.Identity()
            self.obs_dim = int(self.actor.obs_dim)
        else:
            self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()
            self.obs_dim = self._resolve_obs_dim(self.actor)

        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    @staticmethod
    def _resolve_actor(policy):
        if hasattr(policy, "actor"):
            return copy.deepcopy(policy.actor)
        if hasattr(policy, "student"):
            return copy.deepcopy(policy.student)
        if hasattr(policy, "mlp"):
            return copy.deepcopy(policy)
        raise ValueError("Policy does not have an actor/student module or an exportable MLP model.")

    @staticmethod
    def _resolve_obs_dim(actor) -> int:
        if hasattr(actor, "obs_dim"):
            return int(actor.obs_dim)
        if hasattr(actor, "mlp") and hasattr(actor.mlp, "__getitem__"):
            return int(actor.mlp[0].in_features)
        if hasattr(actor, "__getitem__"):
            return int(actor[0].in_features)
        raise ValueError("Unable to infer actor observation dimension for ONNX export.")

    def _actor_forward(self, x):
        if self.uses_tensordict_obs:
            obs = TensorDict({self.obs_group_name: x}, batch_size=x.shape[:-1])
            return self.actor(obs)
        return self.actor(self.normalizer(x))

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self._actor_forward(x),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        obs = torch.zeros(1, self.obs_dim)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
