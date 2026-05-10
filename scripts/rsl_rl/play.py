"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import importlib
import os
import pathlib
import torch
from importlib.metadata import PackageNotFoundError, version as pkg_version

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.sim.converters import urdf_converter as urdf_converter_module
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _get_rsl_rl_installed_version() -> str:
    """Resolve installed rsl-rl package version across naming variants."""
    for pkg_name in ("rsl-rl", "rsl_rl"):
        try:
            return pkg_version(pkg_name)
        except PackageNotFoundError:
            continue
    # Fallback keeps compatibility helper path deterministic.
    return "5.0.0"


def _get_policy_and_normalizer(runner: OnPolicyRunner):
    """Fetch policy and normalizer compatibly across rsl_rl versions."""
    policy = None
    normalizer = None

    alg = getattr(runner, "alg", None)
    if alg is not None:
        policy = getattr(alg, "policy", None)
        if policy is None and hasattr(alg, "get_policy"):
            policy = alg.get_policy()

    if policy is not None:
        normalizer = getattr(policy, "obs_normalizer", None)

    return policy, normalizer


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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _get_rsl_rl_installed_version())
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if args_cli.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    # Ensure URDF importer works across Isaac Sim 4.5 variants before env creation.
    _ensure_urdf_importer_available()
    _patch_urdf_importer_api_compat()

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    # Get policy and normalizer with version compatibility
    export_policy, export_normalizer = _get_policy_and_normalizer(ppo_runner)
    if export_policy is None:
        print("[WARN]: Policy not found for ONNX export, skipping export.")
    else:
        try:
            export_motion_policy_as_onnx(
                env.unwrapped,
                export_policy,
                normalizer=export_normalizer,
                path=export_model_dir,
                filename="policy.onnx",
            )
            attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
        except Exception as err:
            # Playback should continue even when exporter internals differ across rsl_rl versions.
            print(f"[WARN]: ONNX export failed, continuing playback without export: {err}")
    # reset environment
    obs_result = env.get_observations()
    obs = obs_result[0] if isinstance(obs_result, tuple) else obs_result
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
