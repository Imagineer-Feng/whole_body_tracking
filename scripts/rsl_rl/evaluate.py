"""Batch evaluation script for trained RSL-RL policies.

This script evaluates one or more checkpoints across multiple seeds and reports
reproducible tracking metrics with confidence intervals.

What this script does:
- Evaluates one or many checkpoints (WandB or local files).
- Runs fixed episodes per seed for reproducible statistics.
- Reports success rate (timeout-finished episodes), returns, episode length,
    termination reason counts, and command-level motion metrics.
- Exports:
    - seed-level CSV
    - checkpoint-level summary CSV
    - full JSON summary

Evaluation presets:
- ID scene: keep training-time perturbations/randomization/noise.
    Use default arguments (no disable flags).
- Clean scene: remove perturbations/noise for upper-bound tracking quality.
    Use --disable_randomization --disable_obs_noise --disable_early_termination.

Quick examples:

1) ID scene, multiple checkpoints from a WandB run:

     python scripts/rsl_rl/evaluate.py \
         --task Tracking-Flat-G1-v0 \
         --wandb_path org/project/run_id \
         --checkpoint_names model_28000.pt model_28500.pt model_29000.pt model_29500.pt model_30000.pt \
         --num_envs 64 --num_episodes 200 --seeds 0 1 2 3 4 --headless

2) Clean scene, same checkpoints:

     python scripts/rsl_rl/evaluate.py \
         --task Tracking-Flat-G1-v0 \
         --wandb_path org/project/run_id \
         --checkpoint_names model_28000.pt model_28500.pt model_29000.pt model_29500.pt model_30000.pt \
         --num_envs 64 --num_episodes 200 --seeds 0 1 2 3 4 \
         --disable_randomization --disable_obs_noise --disable_early_termination --headless

3) Evaluate one explicit WandB checkpoint path:

     python scripts/rsl_rl/evaluate.py \
         --task Tracking-Flat-G1-v0 \
         --wandb_path org/project/run_id/model_30000.pt \
         --num_envs 64 --num_episodes 200 --seeds 0 1 2 3 4 --headless

4) Evaluate local checkpoints:

     python scripts/rsl_rl/evaluate.py \
         --task Tracking-Flat-G1-v0 \
         --checkpoint_paths /abs/path/model_29000.pt /abs/path/model_30000.pt \
         --num_envs 64 --num_episodes 200 --seeds 0 1 2 3 4 --headless
"""

"""Launch Isaac Sim Simulator first."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(
    description="Evaluate trained RSL-RL checkpoints for motion tracking.",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=(
        "Examples:\n"
        "  ID scene (default):\n"
        "    python scripts/rsl_rl/evaluate.py --task Tracking-Flat-G1-v0 "
        "--wandb_path org/project/run_id "
        "--checkpoint_names model_28000.pt model_28500.pt model_29000.pt model_29500.pt model_30000.pt "
        "--num_envs 64 --num_episodes 200 --seeds 0 1 2 3 4 --headless\n\n"
        "  Clean scene:\n"
        "    python scripts/rsl_rl/evaluate.py --task Tracking-Flat-G1-v0 "
        "--wandb_path org/project/run_id "
        "--checkpoint_names model_28000.pt model_28500.pt model_29000.pt model_29500.pt model_30000.pt "
        "--num_envs 64 --num_episodes 200 --seeds 0 1 2 3 4 "
        "--disable_randomization --disable_obs_noise --disable_early_termination --headless\n"
    ),
)
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=200, help="Episodes per seed for each checkpoint.")
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Seeds for evaluation.")

parser.add_argument(
    "--wandb_path",
    type=str,
    default=None,
    help=(
        "WandB run path or explicit model path. Examples: "
        "org/project/run_id or org/project/run_id/model_30000.pt"
    ),
)
parser.add_argument(
    "--checkpoint_names",
    type=str,
    nargs="*",
    default=None,
    help="Checkpoint names in the WandB run to evaluate, e.g. model_28000.pt model_30000.pt",
)
parser.add_argument(
    "--checkpoint_paths",
    type=str,
    nargs="*",
    default=None,
    help="Local checkpoint paths to evaluate (overrides wandb_path if provided).",
)
parser.add_argument(
    "--registry_name",
    type=str,
    default=None,
    help="Optional WandB motion registry name to force eval motion, e.g. org/wandb-registry-motions/name:latest",
)
parser.add_argument("--motion_file", type=str, default=None, help="Optional local motion npz to force eval motion.")

parser.add_argument(
    "--disable_randomization",
    action="store_true",
    default=False,
    help="Disable startup randomization and push perturbations for clean tracking eval.",
)
parser.add_argument(
    "--disable_early_termination",
    action="store_true",
    default=False,
    help="Disable non-timeout termination terms.",
)
parser.add_argument(
    "--disable_obs_noise",
    action="store_true",
    default=False,
    help="Disable policy observation corruption for clean tracking eval.",
)

parser.add_argument("--output_dir", type=str, default="logs/rsl_rl/eval", help="Directory to write evaluation outputs.")
parser.add_argument("--output_prefix", type=str, default="eval", help="Prefix for output files.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401


@dataclass
class SeedEvalSummary:
    checkpoint: str
    seed: int
    episodes: int
    success_rate: float
    return_mean: float
    return_std: float
    episode_length_mean: float
    reason_counts: dict[str, int]
    motion_metrics: dict[str, float]


def _as_flat_bool_tensor(x: Any, device: torch.device) -> torch.Tensor | None:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not torch.is_tensor(x):
        return None
    return x.to(device=device).view(-1).bool()


def _extract_timeout_mask(infos: Any, device: torch.device) -> torch.Tensor | None:
    if not isinstance(infos, dict):
        return None
    for key in ("time_outs", "timeouts", "truncated", "time_out"):
        if key in infos:
            mask = _as_flat_bool_tensor(infos[key], device)
            if mask is not None:
                return mask
    return None


def _extract_termination_masks(env_unwrapped: Any, num_envs: int, device: torch.device) -> dict[str, torch.Tensor]:
    manager = getattr(env_unwrapped, "termination_manager", None)
    if manager is None:
        return {}

    term_dict = None
    for attr in ("_term_dones", "term_dones"):
        if hasattr(manager, attr):
            candidate = getattr(manager, attr)
            if isinstance(candidate, dict):
                term_dict = candidate
                break

    if term_dict is None:
        return {}

    masks: dict[str, torch.Tensor] = {}
    for name, value in term_dict.items():
        tensor = _as_flat_bool_tensor(value, device)
        if tensor is None:
            continue
        if tensor.numel() != num_envs:
            continue
        masks[name] = tensor
    return masks


def _override_eval_cfg(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
) -> ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg:
    cfg = copy.deepcopy(env_cfg)
    cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else cfg.scene.num_envs
    cfg.sim.device = args_cli.device if args_cli.device is not None else cfg.sim.device

    if args_cli.motion_file is not None:
        cfg.commands.motion.motion_file = args_cli.motion_file

    if args_cli.registry_name is not None:
        import wandb

        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        artifact = wandb.Api().artifact(registry_name)
        cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    if args_cli.disable_randomization and hasattr(cfg, "events") and cfg.events is not None:
        for attr in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
            if hasattr(cfg.events, attr):
                setattr(cfg.events, attr, None)

    if args_cli.disable_early_termination and hasattr(cfg, "terminations") and cfg.terminations is not None:
        for attr in ("anchor_pos", "anchor_ori", "ee_body_pos"):
            if hasattr(cfg.terminations, attr):
                setattr(cfg.terminations, attr, None)

    if args_cli.disable_obs_noise and hasattr(cfg.observations, "policy") and cfg.observations.policy is not None:
        cfg.observations.policy.enable_corruption = False

    return cfg


def _resolve_checkpoints(
    agent_cfg: RslRlOnPolicyRunnerCfg,
) -> list[tuple[str, str]]:
    if args_cli.checkpoint_paths:
        paths = [os.path.abspath(p) for p in args_cli.checkpoint_paths]
        return [(os.path.basename(p), p) for p in paths]

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path
        explicit_model = None
        if "/model_" in run_path and run_path.endswith(".pt"):
            explicit_model = run_path.split("/")[-1]
            run_path = "/".join(run_path.split("/")[:-1])

        wandb_run = wandb.Api().run(run_path)
        if args_cli.checkpoint_names:
            model_files = args_cli.checkpoint_names
        elif explicit_model is not None:
            model_files = [explicit_model]
        else:
            model_files = [f.name for f in wandb_run.files() if f.name.startswith("model_") and f.name.endswith(".pt")]
            model_files = sorted(model_files, key=lambda name: int(name.split("_")[1].split(".")[0]))

        local_root = os.path.abspath(os.path.join("logs", "rsl_rl", "temp_eval", run_path.replace("/", "_")))
        os.makedirs(local_root, exist_ok=True)

        resolved = []
        for name in model_files:
            local_file = os.path.join(local_root, name)
            wandb_run.file(name).download(root=local_root, replace=True)
            resolved.append((name, local_file))
        return resolved

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    return [(os.path.basename(checkpoint_path), checkpoint_path)]


def _evaluate_single_seed(
    checkpoint_name: str,
    checkpoint_path: str,
    seed: int,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
) -> SeedEvalSummary:
    local_env_cfg = copy.deepcopy(env_cfg)
    local_env_cfg.seed = seed

    env = gym.make(args_cli.task, cfg=local_env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(checkpoint_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    num_envs = env.unwrapped.num_envs
    episode_returns = torch.zeros(num_envs, device=env.unwrapped.device)
    episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=env.unwrapped.device)

    returns_list: list[float] = []
    lengths_list: list[int] = []
    reason_counts: defaultdict[str, int] = defaultdict(int)

    motion_metric_sums: defaultdict[str, float] = defaultdict(float)
    motion_metric_steps = 0

    obs, _ = env.get_observations()

    max_total_steps = int(args_cli.num_episodes * max(1, num_envs) * 20)
    total_steps = 0

    while len(returns_list) < args_cli.num_episodes and total_steps < max_total_steps and simulation_app.is_running():
        total_steps += 1
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)

        rewards = rewards.view(-1)
        dones = dones.view(-1).bool()

        episode_returns += rewards
        episode_lengths += 1

        motion_command = env.unwrapped.command_manager.get_term("motion")
        for name, tensor in motion_command.metrics.items():
            motion_metric_sums[name] += float(tensor.mean().item())
        motion_metric_steps += 1

        if not torch.any(dones):
            continue

        timeout_mask = _extract_timeout_mask(infos, env.unwrapped.device)
        term_masks = _extract_termination_masks(env.unwrapped, num_envs, env.unwrapped.device)

        done_ids = torch.nonzero(dones, as_tuple=False).view(-1).tolist()
        for env_id in done_ids:
            if len(returns_list) >= args_cli.num_episodes:
                break

            returns_list.append(float(episode_returns[env_id].item()))
            lengths_list.append(int(episode_lengths[env_id].item()))

            is_timeout = bool(timeout_mask[env_id].item()) if timeout_mask is not None else False
            if is_timeout:
                reason_counts["time_out"] += 1
            else:
                reason = None
                for term_name, term_mask in term_masks.items():
                    if term_name == "time_out":
                        continue
                    if bool(term_mask[env_id].item()):
                        reason = term_name
                        break
                reason_counts[reason if reason is not None else "terminated_unknown"] += 1

            episode_returns[env_id] = 0.0
            episode_lengths[env_id] = 0

    env.close()

    if not returns_list:
        raise RuntimeError(
            f"No episodes finished for checkpoint={checkpoint_name}, seed={seed}. "
            "Try reducing num_envs, increasing simulation time, or checking termination settings."
        )

    success = reason_counts.get("time_out", 0)
    success_rate = success / len(returns_list)

    motion_metrics = {}
    if motion_metric_steps > 0:
        for key, value in motion_metric_sums.items():
            motion_metrics[key] = value / motion_metric_steps

    return SeedEvalSummary(
        checkpoint=checkpoint_name,
        seed=seed,
        episodes=len(returns_list),
        success_rate=success_rate,
        return_mean=float(np.mean(returns_list)),
        return_std=float(np.std(returns_list)),
        episode_length_mean=float(np.mean(lengths_list)),
        reason_counts=dict(reason_counts),
        motion_metrics=motion_metrics,
    )


def _mean_std_ci95(values: list[float]) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) > 0 else float("nan")
    return mean, std, ci95


def _write_seed_csv(path: str, rows: list[SeedEvalSummary]):
    all_reason_keys = sorted({k for r in rows for k in r.reason_counts.keys()})
    all_motion_keys = sorted({k for r in rows for k in r.motion_metrics.keys()})

    headers = [
        "checkpoint",
        "seed",
        "episodes",
        "success_rate",
        "return_mean",
        "return_std",
        "episode_length_mean",
    ] + [f"reason_{k}" for k in all_reason_keys] + [f"motion_{k}" for k in all_motion_keys]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            data = {
                "checkpoint": row.checkpoint,
                "seed": row.seed,
                "episodes": row.episodes,
                "success_rate": row.success_rate,
                "return_mean": row.return_mean,
                "return_std": row.return_std,
                "episode_length_mean": row.episode_length_mean,
            }
            for key in all_reason_keys:
                data[f"reason_{key}"] = row.reason_counts.get(key, 0)
            for key in all_motion_keys:
                data[f"motion_{key}"] = row.motion_metrics.get(key, float("nan"))
            writer.writerow(data)


def _write_checkpoint_csv(path: str, rows: list[dict[str, Any]]):
    if not rows:
        return
    headers = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg = _override_eval_cfg(env_cfg)

    checkpoints = _resolve_checkpoints(agent_cfg)
    if len(checkpoints) == 0:
        raise RuntimeError("No checkpoints found for evaluation.")

    print(f"[INFO] Evaluating {len(checkpoints)} checkpoints with seeds={args_cli.seeds}")

    seed_rows: list[SeedEvalSummary] = []
    checkpoint_rows: list[dict[str, Any]] = []

    for checkpoint_name, checkpoint_path in checkpoints:
        print(f"[INFO] Checkpoint: {checkpoint_name} ({checkpoint_path})")
        per_seed: list[SeedEvalSummary] = []

        for seed in args_cli.seeds:
            print(f"[INFO]   Seed {seed}: running {args_cli.num_episodes} episodes")
            summary = _evaluate_single_seed(checkpoint_name, checkpoint_path, seed, env_cfg, agent_cfg)
            per_seed.append(summary)
            seed_rows.append(summary)
            print(
                f"[INFO]   Seed {seed} done: success_rate={summary.success_rate:.4f}, "
                f"return_mean={summary.return_mean:.4f}"
            )

        sr_values = [x.success_rate for x in per_seed]
        ret_values = [x.return_mean for x in per_seed]
        len_values = [x.episode_length_mean for x in per_seed]

        sr_mean, sr_std, sr_ci95 = _mean_std_ci95(sr_values)
        ret_mean, ret_std, ret_ci95 = _mean_std_ci95(ret_values)
        len_mean, len_std, len_ci95 = _mean_std_ci95(len_values)

        checkpoint_rows.append(
            {
                "checkpoint": checkpoint_name,
                "num_seeds": len(per_seed),
                "episodes_per_seed": args_cli.num_episodes,
                "success_rate_mean": sr_mean,
                "success_rate_std": sr_std,
                "success_rate_ci95": sr_ci95,
                "return_mean": ret_mean,
                "return_std": ret_std,
                "return_ci95": ret_ci95,
                "episode_length_mean": len_mean,
                "episode_length_std": len_std,
                "episode_length_ci95": len_ci95,
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    seed_csv = os.path.join(out_dir, f"{args_cli.output_prefix}_seed_metrics_{timestamp}.csv")
    ckpt_csv = os.path.join(out_dir, f"{args_cli.output_prefix}_checkpoint_summary_{timestamp}.csv")
    summary_json = os.path.join(out_dir, f"{args_cli.output_prefix}_summary_{timestamp}.json")

    _write_seed_csv(seed_csv, seed_rows)
    _write_checkpoint_csv(ckpt_csv, checkpoint_rows)

    json_payload = {
        "args": vars(args_cli),
        "checkpoints": checkpoints,
        "seed_metrics": [
            {
                "checkpoint": x.checkpoint,
                "seed": x.seed,
                "episodes": x.episodes,
                "success_rate": x.success_rate,
                "return_mean": x.return_mean,
                "return_std": x.return_std,
                "episode_length_mean": x.episode_length_mean,
                "reason_counts": x.reason_counts,
                "motion_metrics": x.motion_metrics,
            }
            for x in seed_rows
        ],
        "checkpoint_summary": checkpoint_rows,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    print("[INFO] Evaluation finished.")
    print(f"[INFO] Seed metrics CSV: {seed_csv}")
    print(f"[INFO] Checkpoint summary CSV: {ckpt_csv}")
    print(f"[INFO] JSON summary: {summary_json}")


if __name__ == "__main__":
    main()
    simulation_app.close()
