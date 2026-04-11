"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --motion_file source/whole_body_tracking/whole_body_tracking/assets/g1/motions/lafan_walk_short.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")
parser.add_argument(
    "--playback_speed",
    type=float,
    default=1.0,
    help="Playback speed multiplier for motion replay (e.g., 0.5=half speed, 2.0=double speed).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.converters import urdf_converter as urdf_converter_module
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


def _ensure_urdf_importer_available() -> None:
    """Ensures URDF importer extension is loaded before spawning URDF assets."""

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
    """Patches IsaacLab URDF converter for older URDF importer builds."""

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


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    registry_name = args_cli.registry_name
    if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
        registry_name += ":latest"
    import pathlib

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    frame_accumulator = 0.0
    motion_fps = float(motion.fps.item()) if hasattr(motion.fps, 'item') else float(motion.fps)
    sim_dt_val = float(sim_dt.item()) if hasattr(sim_dt, 'item') else float(sim_dt)
    frames_per_loop = motion_fps * float(args_cli.playback_speed) * sim_dt_val

    print(
        f"[INFO]: Replay speed={args_cli.playback_speed:.3f}x, motion_fps={motion_fps:.2f}, "
        f"sim_dt={sim_dt_val:.4f}, expected_frame_advance_per_loop={frames_per_loop:.4f}"
    )

    # Simulation loop
    while simulation_app.is_running():
        frame_accumulator += frames_per_loop
        frame_step = int(frame_accumulator)
        if frame_step > 0:
            time_steps += frame_step
            frame_accumulator -= frame_step
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    _ensure_urdf_importer_available()
    _patch_urdf_importer_api_compat()

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
