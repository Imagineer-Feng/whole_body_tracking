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
         --wandb_path imagineer-feng-shanghai-jiao-tong-university/beyondmimic_repo/d1s1formal30000_bnrfmt \
         --checkpoint_names model_28000.pt model_28500.pt model_29000.pt model_29500.pt model_29999.pt \
         --motion_file /home/imagineerfeng/whole_body_tracking/artifacts/dance1_subject1:v0/motion.npz \
         --num_envs 64 --num_episodes 200 --seeds 0 1 2 3 4 --headless

2) Clean scene, same checkpoints:

     python scripts/rsl_rl/evaluate.py \
         --task Tracking-Flat-G1-v0 \
         --wandb_path imagineer-feng-shanghai-jiao-tong-university/beyondmimic_repo/d1s1formal30000_bnrfmt \
         --checkpoint_names model_28000.pt model_28500.pt model_29000.pt model_29500.pt model_29999.pt \
         --motion_file /home/imagineerfeng/whole_body_tracking/artifacts/dance1_subject1:v0/motion.npz \
         --num_envs 64 --num_episodes 200 --seeds 0 1 2 3 4 \
         --disable_randomization --disable_obs_noise --disable_early_termination --headless

3) Evaluate one explicit WandB checkpoint path:

     python scripts/rsl_rl/evaluate.py \
         --task Tracking-Flat-G1-v0 \
         --wandb_path imagineer-feng-shanghai-jiao-tong-university/beyondmimic_repo/d1s1formal30000_bnrfmt/model_30000.pt \
         --num_envs 64 --num_episodes 200 --seeds 0 1 2 3 4 --headless

4) Evaluate local checkpoints:

     python scripts/rsl_rl/evaluate.py \
         --task Tracking-Flat-G1-v0 \
         --checkpoint_paths /abs/path/model_29000.pt /abs/path/model_30000.pt \
         --num_envs 64 --num_episodes 200 --seeds 0 1 2 3 4 --headless
"""

from __future__ import annotations

# Launch Isaac Sim Simulator first.

import argparse
import copy
import csv
import dataclasses
import gc
import importlib
import json
import math
import os
import pathlib
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import Any

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

RAW_CLI_ARGS = sys.argv[1:]


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
    "--force_download_checkpoints",
    action="store_true",
    default=False,
    help="Force re-downloading WandB checkpoints even when cached local files already exist.",
)
parser.add_argument(
    "--checkpoint_download_retries",
    type=int,
    default=3,
    help="Number of WandB download attempts for each missing or forced checkpoint.",
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
parser.add_argument(
    "--seed_isolation",
    action="store_true",
    default=False,
    help="Force running each seed in a separate subprocess and aggregate outputs.",
)
parser.add_argument(
    "--allow_failed_seeds",
    action="store_true",
    default=False,
    help="Allow seed-isolated evaluation to finish when some checkpoint/seed workers fail.",
)
parser.add_argument(
    "--_seed_worker",
    action="store_true",
    default=False,
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--_worker_json_out",
    type=str,
    default=None,
    help=argparse.SUPPRESS,
)

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
from isaaclab.sim.converters import urdf_converter as urdf_converter_module
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
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


def _ensure_urdf_importer_available() -> None:
    """Ensure URDF importer extension is enabled before spawning URDF assets."""

    def _try_import() -> bool:
        try:
            importlib.import_module("isaacsim.asset.importer.urdf._urdf")
            return True
        except ModuleNotFoundError:
            return False

    if _try_import():
        return

    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    for ext_name in ("isaacsim.asset.importer.urdf", "omni.importer.urdf"):
        try:
            if not ext_manager.is_extension_enabled(ext_name):
                ext_manager.set_extension_enabled_immediate(ext_name, True)
        except Exception:
            continue

        if _try_import():
            return

    raise ModuleNotFoundError(
        "URDF importer module is unavailable. Please ensure extension "
        "'isaacsim.asset.importer.urdf' is installed and enabled in this Kit experience."
    )


def _patch_urdf_importer_api_compat() -> None:
    """Patch URDF converter API differences across Isaac Sim 4.5 builds."""

    if getattr(urdf_converter_module.UrdfConverter, "_wbt_compat_patched", False):
        return

    def _compat_get_urdf_import_config(self):
        import omni.kit.commands

        _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")

        import_config.set_distance_scale(1.0)
        import_config.set_make_default_prim(True)
        import_config.set_create_physics_scene(False)

        import_config.set_density(self.cfg.link_density)
        convex_decomp = self.cfg.collider_type == "convex_decomposition"
        import_config.set_convex_decomp(convex_decomp)
        import_config.set_collision_from_visuals(self.cfg.collision_from_visuals)
        import_config.set_merge_fixed_joints(self.cfg.merge_fixed_joints)
        if hasattr(import_config, "set_merge_fixed_ignore_inertia"):
            import_config.set_merge_fixed_ignore_inertia(self.cfg.merge_fixed_joints)

        import_config.set_fix_base(self.cfg.fix_base)
        import_config.set_self_collision(self.cfg.self_collision)
        import_config.set_parse_mimic(self.cfg.convert_mimic_joints_to_normal_joints)
        import_config.set_replace_cylinders_with_capsules(self.cfg.replace_cylinders_with_capsules)

        return import_config

    urdf_converter_module.UrdfConverter._get_urdf_import_config = _compat_get_urdf_import_config
    urdf_converter_module.UrdfConverter._wbt_compat_patched = True


def _get_rsl_rl_installed_version() -> str:
    """Resolve installed rsl-rl package version across naming variants."""
    for pkg_name in ("rsl-rl", "rsl_rl"):
        try:
            return pkg_version(pkg_name)
        except PackageNotFoundError:
            continue
    # Keep fallback deterministic when package metadata cannot be resolved.
    return "5.0.0"


def _as_flat_bool_tensor(x: Any, device: torch.device) -> torch.Tensor | None:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not torch.is_tensor(x):
        return None
    return x.to(device=device).view(-1).bool()


def _as_flat_float_tensor(x: Any, device: torch.device, expected_numel: int) -> torch.Tensor | None:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not torch.is_tensor(x):
        return None
    tensor = x.to(device=device, dtype=torch.float32).view(-1)
    if tensor.numel() == expected_numel:
        return tensor
    if tensor.numel() == 1:
        return tensor.expand(expected_numel)
    return None


def _extract_actor_observation(obs_like: Any) -> Any:
    """Normalize observation container into actor/policy observation tensor."""
    if isinstance(obs_like, (tuple, list)) and len(obs_like) > 0:
        obs_like = obs_like[0]

    if isinstance(obs_like, dict):
        for key in ("policy", "actor", "obs", "observation"):
            if key in obs_like:
                return obs_like[key]
        if len(obs_like) == 1:
            return next(iter(obs_like.values()))

    return obs_like


def _unpack_step_result(step_result: Any, device: torch.device) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Handle VecEnv step output variants: 4-tuple or Gymnasium 5-tuple."""
    if not isinstance(step_result, (tuple, list)):
        raise RuntimeError(f"Unexpected env.step output type: {type(step_result)}")

    if len(step_result) == 4:
        obs, rewards, dones, infos = step_result
    elif len(step_result) == 5:
        obs, rewards, terminated, truncated, infos = step_result
        terminated_t = torch.as_tensor(terminated, device=device).view(-1).bool()
        truncated_t = torch.as_tensor(truncated, device=device).view(-1).bool()
        dones = torch.logical_or(terminated_t, truncated_t)
    else:
        raise RuntimeError(f"Unexpected env.step output length: {len(step_result)}")

    rewards_t = torch.as_tensor(rewards, device=device).view(-1)
    dones_t = torch.as_tensor(dones, device=device).view(-1).bool()
    infos_dict = infos if isinstance(infos, dict) else {}
    return _extract_actor_observation(obs), rewards_t, dones_t, infos_dict


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


def _checkpoint_sort_key(name: str) -> tuple[int, int | str]:
    stem = os.path.basename(name)
    if stem.startswith("model_") and stem.endswith(".pt"):
        try:
            return (0, int(stem.split("_", 1)[1].split(".", 1)[0]))
        except ValueError:
            pass
    return (1, stem)


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

    # Fallback: infer motion file from run-level used motion artifact when evaluating from wandb run.
    if args_cli.motion_file is None and args_cli.registry_name is None and args_cli.wandb_path is not None:
        import wandb

        run_path = args_cli.wandb_path
        if "/model_" in run_path and run_path.endswith(".pt"):
            run_path = "/".join(run_path.split("/")[:-1])
        wandb_run = wandb.Api().run(run_path)
        motion_artifact = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if motion_artifact is not None:
            cfg.commands.motion.motion_file = str(pathlib.Path(motion_artifact.download()) / "motion.npz")

    motion_file = getattr(cfg.commands.motion, "motion_file", None)
    if motion_file is None or isinstance(motion_file, type(dataclasses.MISSING)):
        raise ValueError(
            "Motion file is not set for evaluation. The specified WandB run does not provide a linked "
            "'motions' artifact. Please pass either --motion_file /abs/path/motion.npz or "
            "--registry_name org/wandb-registry-motions/name:latest."
        )

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

    # Evaluation is headless/statistical; disable command debug visualizers to avoid lingering callbacks
    # during env teardown between seeds.
    if hasattr(cfg, "commands") and cfg.commands is not None:
        for term_name in dir(cfg.commands):
            if term_name.startswith("_"):
                continue
            try:
                term_cfg = getattr(cfg.commands, term_name)
            except Exception:
                continue
            if hasattr(term_cfg, "debug_vis"):
                try:
                    setattr(term_cfg, "debug_vis", False)
                except Exception:
                    pass

    return cfg


def _resolve_checkpoints(
    agent_cfg: RslRlOnPolicyRunnerCfg,
) -> list[tuple[str, str]]:
    if args_cli.checkpoint_paths:
        paths = [os.path.abspath(p) for p in args_cli.checkpoint_paths]
        return [(os.path.basename(p), p) for p in paths]

    if args_cli.wandb_path:
        run_path = args_cli.wandb_path
        explicit_model = None
        if "/model_" in run_path and run_path.endswith(".pt"):
            explicit_model = run_path.split("/")[-1]
            run_path = "/".join(run_path.split("/")[:-1])

        local_root = os.path.abspath(os.path.join("logs", "rsl_rl", "temp_eval", run_path.replace("/", "_")))
        os.makedirs(local_root, exist_ok=True)

        wandb_run = None
        if args_cli.checkpoint_names:
            model_files = args_cli.checkpoint_names
        elif explicit_model is not None:
            model_files = [explicit_model]
        else:
            import wandb

            wandb_run = wandb.Api().run(run_path)
            model_files = [f.name for f in wandb_run.files() if f.name.startswith("model_") and f.name.endswith(".pt")]
            model_files = sorted(model_files, key=_checkpoint_sort_key)

        resolved = []
        for name in model_files:
            local_file = os.path.join(local_root, name)
            if os.path.exists(local_file) and os.path.getsize(local_file) > 0 and not args_cli.force_download_checkpoints:
                print(f"[INFO] Using cached checkpoint: {local_file}", flush=True)
                resolved.append((name, local_file))
                continue

            last_error = None
            for attempt in range(1, max(args_cli.checkpoint_download_retries, 1) + 1):
                try:
                    if wandb_run is None:
                        import wandb

                        wandb_run = wandb.Api().run(run_path)
                    print(f"[INFO] Downloading checkpoint from WandB: {name} (attempt {attempt})", flush=True)
                    wandb_run.file(name).download(
                        root=local_root,
                        replace=args_cli.force_download_checkpoints or not os.path.exists(local_file),
                    )
                    break
                except Exception as err:
                    last_error = err
                    if attempt >= max(args_cli.checkpoint_download_retries, 1):
                        raise RuntimeError(
                            f"Failed to download checkpoint '{name}' from WandB after {attempt} attempt(s). "
                            f"If it was downloaded before, pass --checkpoint_paths {local_file} or retry without "
                            "--force_download_checkpoints so the local cache can be used."
                        ) from err
                    sleep_s = min(5 * attempt, 20)
                    print(
                        f"[WARNING] Download failed for {name}: {last_error}. Retrying in {sleep_s}s...",
                        flush=True,
                    )
                    time.sleep(sleep_s)

            if not os.path.exists(local_file) or os.path.getsize(local_file) == 0:
                raise RuntimeError(f"Checkpoint download finished but local file is missing or empty: {local_file}")
            resolved.append((name, local_file))
        return resolved

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    return [(os.path.basename(checkpoint_path), checkpoint_path)]


def _disable_env_debug_vis(env_unwrapped: Any) -> None:
    """Best-effort shutdown of debug visualization callbacks before env teardown."""
    command_manager = getattr(env_unwrapped, "command_manager", None)
    if command_manager is None:
        return

    # Manager-level switch (if available).
    set_mgr_debug_vis = getattr(command_manager, "set_debug_vis", None)
    if callable(set_mgr_debug_vis):
        try:
            set_mgr_debug_vis(False)
        except Exception:
            pass

    # Per-term switch for compatibility with different IsaacLab manager APIs.
    active_terms = getattr(command_manager, "active_terms", [])
    for term_name in active_terms:
        try:
            term = command_manager.get_term(term_name)
        except Exception:
            continue
        set_term_debug_vis = getattr(term, "set_debug_vis", None)
        if callable(set_term_debug_vis):
            try:
                set_term_debug_vis(False)
            except Exception:
                pass


def _evaluate_single_seed(
    checkpoint_name: str,
    checkpoint_path: str,
    seed: int,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
) -> SeedEvalSummary:
    setup_t0 = time.time()
    print(f"[INFO]   Seed {seed}: setup start (checkpoint={checkpoint_name})", flush=True)

    local_env_cfg = copy.deepcopy(env_cfg)
    local_env_cfg.seed = seed

    returns_list: list[float] = []
    lengths_list: list[int] = []
    reason_counts: defaultdict[str, int] = defaultdict(int)
    motion_episode_values: defaultdict[str, list[float]] = defaultdict(list)

    env = None
    ppo_runner = None
    policy = None
    try:
        print(f"[INFO]   Seed {seed}: creating env via gym.make...", flush=True)
        env_create_done = threading.Event()

        def _env_create_heartbeat() -> None:
            while not env_create_done.wait(15.0):
                elapsed = time.time() - setup_t0
                print(f"[INFO]   Seed {seed}: still creating env... elapsed={elapsed:.1f}s", flush=True)

        heartbeat_thread = threading.Thread(target=_env_create_heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            env = gym.make(args_cli.task, cfg=local_env_cfg, render_mode=None)
        finally:
            env_create_done.set()

        print(f"[INFO]   Seed {seed}: env created in {time.time() - setup_t0:.1f}s", flush=True)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env)
        print(f"[INFO]   Seed {seed}: env wrapped in {time.time() - setup_t0:.1f}s", flush=True)

        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO]   Seed {seed}: runner constructed in {time.time() - setup_t0:.1f}s", flush=True)
        ppo_runner.load(checkpoint_path)
        print(f"[INFO]   Seed {seed}: checkpoint loaded in {time.time() - setup_t0:.1f}s", flush=True)
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        print(f"[INFO]   Seed {seed}: inference policy ready in {time.time() - setup_t0:.1f}s", flush=True)

        num_envs = env.unwrapped.num_envs
        device = env.unwrapped.device
        episode_returns = torch.zeros(num_envs, device=device)
        episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
        motion_metric_sums: dict[str, torch.Tensor] = {}

        obs = _extract_actor_observation(env.get_observations())
        print(f"[INFO]   Seed {seed}: initial observations ready in {time.time() - setup_t0:.1f}s", flush=True)

        max_total_steps = int(args_cli.num_episodes * max(1, num_envs) * 20)
        total_steps = 0
        start_time = time.time()
        last_progress_log_time = start_time

        while len(returns_list) < args_cli.num_episodes and total_steps < max_total_steps and simulation_app.is_running():
            total_steps += 1
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, infos = _unpack_step_result(env.step(actions), device)

            episode_returns += rewards
            episode_lengths += 1

            motion_command = env.unwrapped.command_manager.get_term("motion")
            for name, tensor in motion_command.metrics.items():
                metric_values = _as_flat_float_tensor(tensor, device, num_envs)
                if metric_values is None:
                    continue
                if name not in motion_metric_sums:
                    motion_metric_sums[name] = torch.zeros(num_envs, device=device)
                motion_metric_sums[name] += metric_values

            now = time.time()
            if now - last_progress_log_time >= 15.0:
                elapsed = max(now - start_time, 1e-6)
                steps_per_sec = total_steps / elapsed
                print(
                    f"[INFO]   Seed {seed} progress: episodes={len(returns_list)}/{args_cli.num_episodes}, "
                    f"sim_steps={total_steps}, elapsed={elapsed:.1f}s, steps_per_sec={steps_per_sec:.1f}",
                    flush=True,
                )
                last_progress_log_time = now

            if not torch.any(dones):
                continue

            timeout_mask = _extract_timeout_mask(infos, device)
            term_masks = _extract_termination_masks(env.unwrapped, num_envs, device)
            if timeout_mask is None:
                timeout_mask = term_masks.get("time_out")

            done_ids = torch.nonzero(dones, as_tuple=False).view(-1).tolist()
            for env_id in done_ids:
                if len(returns_list) >= args_cli.num_episodes:
                    break

                episode_len = max(int(episode_lengths[env_id].item()), 1)
                returns_list.append(float(episode_returns[env_id].item()))
                lengths_list.append(episode_len)

                for name, sums in motion_metric_sums.items():
                    motion_episode_values[name].append(float((sums[env_id] / episode_len).item()))

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
                for sums in motion_metric_sums.values():
                    sums[env_id] = 0.0
    finally:
        if env is not None:
            print(f"[INFO]   Seed {seed}: closing env...", flush=True)
            try:
                _disable_env_debug_vis(env.unwrapped)
            except Exception:
                pass
            try:
                env.close()
            except Exception:
                pass

        del policy
        del ppo_runner
        del env

        # Flush pending simulator teardown work before creating the next seed environment.
        for _ in range(3):
            try:
                simulation_app.update()
            except Exception:
                break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[INFO]   Seed {seed}: cleanup finished in {time.time() - setup_t0:.1f}s", flush=True)

    if not returns_list:
        raise RuntimeError(
            f"No episodes finished for checkpoint={checkpoint_name}, seed={seed}. "
            "Try reducing num_envs, increasing simulation time, or checking termination settings."
        )
    if len(returns_list) < args_cli.num_episodes:
        raise RuntimeError(
            f"Only {len(returns_list)}/{args_cli.num_episodes} episodes finished for "
            f"checkpoint={checkpoint_name}, seed={seed}. The evaluation stopped before collecting "
            "the requested fixed episode count."
        )

    success = reason_counts.get("time_out", 0)
    success_rate = success / len(returns_list)

    motion_metrics = {
        key: float(np.mean(values))
        for key, values in motion_episode_values.items()
        if len(values) > 0
    }

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
    extra_headers = sorted({key for row in rows for key in row.keys() if key not in headers})
    headers.extend(extra_headers)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _build_checkpoint_rows(seed_rows: list[SeedEvalSummary], episodes_per_seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[SeedEvalSummary]] = defaultdict(list)
    for row in seed_rows:
        grouped[row.checkpoint].append(row)

    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint_name, per_seed in grouped.items():
        sr_values = [x.success_rate for x in per_seed]
        ret_values = [x.return_mean for x in per_seed]
        len_values = [x.episode_length_mean for x in per_seed]
        total_episodes = sum(x.episodes for x in per_seed)
        reason_keys = sorted({k for x in per_seed for k in x.reason_counts.keys()})
        motion_keys = sorted({k for x in per_seed for k in x.motion_metrics.keys()})

        sr_mean, sr_std, sr_ci95 = _mean_std_ci95(sr_values)
        ret_mean, ret_std, ret_ci95 = _mean_std_ci95(ret_values)
        len_mean, len_std, len_ci95 = _mean_std_ci95(len_values)

        data: dict[str, Any] = {
            "checkpoint": checkpoint_name,
            "num_seeds": len(per_seed),
            "episodes_per_seed": episodes_per_seed,
            "total_episodes": total_episodes,
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

        for key in reason_keys:
            total = sum(x.reason_counts.get(key, 0) for x in per_seed)
            data[f"reason_{key}_total"] = total
            data[f"reason_{key}_rate"] = total / total_episodes if total_episodes > 0 else float("nan")

        for key in motion_keys:
            values = [x.motion_metrics[key] for x in per_seed if key in x.motion_metrics]
            mean, std, ci95 = _mean_std_ci95(values)
            data[f"motion_{key}_mean"] = mean
            data[f"motion_{key}_std"] = std
            data[f"motion_{key}_ci95"] = ci95

        checkpoint_rows.append(data)

    checkpoint_rows.sort(key=lambda x: _checkpoint_sort_key(x["checkpoint"]))
    return checkpoint_rows


def _strip_flag_with_values(args: list[str], flag: str) -> list[str]:
    """Remove one flag and its following non-flag values from a CLI arg list."""
    out: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token == flag:
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                i += 1
            continue
        out.append(token)
        i += 1
    return out


def _write_outputs(seed_rows: list[SeedEvalSummary], checkpoints: list[tuple[str, str]]) -> tuple[str, str, str]:
    checkpoint_rows = _build_checkpoint_rows(seed_rows, args_cli.num_episodes)
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

    if args_cli._seed_worker and args_cli._worker_json_out:
        with open(args_cli._worker_json_out, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2)
        return "", "", args_cli._worker_json_out

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    seed_csv = os.path.join(out_dir, f"{args_cli.output_prefix}_seed_metrics_{timestamp}.csv")
    ckpt_csv = os.path.join(out_dir, f"{args_cli.output_prefix}_checkpoint_summary_{timestamp}.csv")
    summary_json = os.path.join(out_dir, f"{args_cli.output_prefix}_summary_{timestamp}.json")

    _write_seed_csv(seed_csv, seed_rows)
    _write_checkpoint_csv(ckpt_csv, checkpoint_rows)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    return seed_csv, ckpt_csv, summary_json


def _run_seed_isolation_subprocesses(agent_cfg: RslRlOnPolicyRunnerCfg) -> None:
    checkpoints = _resolve_checkpoints(agent_cfg)
    if len(checkpoints) == 0:
        raise RuntimeError("No checkpoints found for evaluation.")

    print(f"[INFO] Isolated evaluation: checkpoints={len(checkpoints)}, seeds={args_cli.seeds}")

    base_args = list(RAW_CLI_ARGS)
    base_args = _strip_flag_with_values(base_args, "--seeds")
    base_args = _strip_flag_with_values(base_args, "--checkpoint_names")
    base_args = _strip_flag_with_values(base_args, "--checkpoint_paths")
    base_args = _strip_flag_with_values(base_args, "--seed_isolation")
    base_args = _strip_flag_with_values(base_args, "--_seed_worker")
    base_args = _strip_flag_with_values(base_args, "--_worker_json_out")

    seed_rows: list[SeedEvalSummary] = []
    failed_jobs: list[tuple[str, int]] = []
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_dir = os.path.abspath(os.path.join(args_cli.output_dir, f".seed_workers_{run_stamp}"))
    os.makedirs(tmp_dir, exist_ok=True)

    for checkpoint_name, checkpoint_path in checkpoints:
        for seed in args_cli.seeds:
            worker_json = os.path.join(tmp_dir, f"ckpt_{checkpoint_name}_seed_{seed}.json")
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                *base_args,
                "--checkpoint_paths",
                checkpoint_path,
                "--seeds",
                str(seed),
                "--_seed_worker",
                "--_worker_json_out",
                worker_json,
            ]
            print(f"[INFO] Launching worker: checkpoint={checkpoint_name}, seed={seed}", flush=True)
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(
                    f"[ERROR] Worker failed: checkpoint={checkpoint_name}, seed={seed}, "
                    f"exit_code={result.returncode}"
                )
                failed_jobs.append((checkpoint_name, seed))
                continue

            if not os.path.exists(worker_json):
                print(
                    f"[ERROR] Worker missing output JSON: checkpoint={checkpoint_name}, "
                    f"seed={seed}, path={worker_json}"
                )
                failed_jobs.append((checkpoint_name, seed))
                continue

            with open(worker_json, "r", encoding="utf-8") as f:
                payload = json.load(f)

            for row in payload.get("seed_metrics", []):
                seed_rows.append(
                    SeedEvalSummary(
                        checkpoint=row["checkpoint"],
                        seed=int(row["seed"]),
                        episodes=int(row["episodes"]),
                        success_rate=float(row["success_rate"]),
                        return_mean=float(row["return_mean"]),
                        return_std=float(row["return_std"]),
                        episode_length_mean=float(row["episode_length_mean"]),
                        reason_counts=dict(row.get("reason_counts", {})),
                        motion_metrics={k: float(v) for k, v in row.get("motion_metrics", {}).items()},
                    )
                )

    if failed_jobs and not args_cli.allow_failed_seeds:
        raise RuntimeError(f"Some seed workers failed: {failed_jobs}")

    if not seed_rows:
        raise RuntimeError(f"All seed workers failed. Failed jobs: {failed_jobs}")

    seed_csv, ckpt_csv, summary_json = _write_outputs(seed_rows, checkpoints)
    print("[INFO] Seed-isolated evaluation finished.")
    print(f"[INFO] Seed metrics CSV: {seed_csv}")
    print(f"[INFO] Checkpoint summary CSV: {ckpt_csv}")
    print(f"[INFO] JSON summary: {summary_json}")
    if failed_jobs:
        print(f"[WARNING] Some checkpoint/seed workers failed and were skipped: {failed_jobs}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _get_rsl_rl_installed_version())

    has_multi_checkpoints = (
        (args_cli.checkpoint_names is not None and len(args_cli.checkpoint_names) > 1)
        or (args_cli.checkpoint_paths is not None and len(args_cli.checkpoint_paths) > 1)
    )
    if (args_cli.seed_isolation or len(args_cli.seeds) > 1 or has_multi_checkpoints) and not args_cli._seed_worker:
        _run_seed_isolation_subprocesses(agent_cfg)
        return

    env_cfg = _override_eval_cfg(env_cfg)

    # Ensure URDF importer works across Isaac Sim 4.5 variants before creating environments.
    _ensure_urdf_importer_available()
    _patch_urdf_importer_api_compat()

    checkpoints = _resolve_checkpoints(agent_cfg)
    if len(checkpoints) == 0:
        raise RuntimeError("No checkpoints found for evaluation.")

    print(f"[INFO] Evaluating {len(checkpoints)} checkpoints with seeds={args_cli.seeds}")

    seed_rows: list[SeedEvalSummary] = []

    for checkpoint_name, checkpoint_path in checkpoints:
        print(f"[INFO] Checkpoint: {checkpoint_name} ({checkpoint_path})")

        for seed in args_cli.seeds:
            print(f"[INFO]   Seed {seed}: running {args_cli.num_episodes} episodes")
            summary = _evaluate_single_seed(checkpoint_name, checkpoint_path, seed, env_cfg, agent_cfg)
            seed_rows.append(summary)
            print(
                f"[INFO]   Seed {seed} done: success_rate={summary.success_rate:.4f}, "
                f"return_mean={summary.return_mean:.4f}"
            )

    seed_csv, ckpt_csv, summary_json = _write_outputs(seed_rows, checkpoints)

    print("[INFO] Evaluation finished.")
    print(f"[INFO] Seed metrics CSV: {seed_csv}")
    print(f"[INFO] Checkpoint summary CSV: {ckpt_csv}")
    print(f"[INFO] JSON summary: {summary_json}")


if __name__ == "__main__":
    main()
    simulation_app.close()
