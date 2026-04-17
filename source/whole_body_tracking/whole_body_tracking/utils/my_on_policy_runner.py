import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _is_wandb_logging_enabled(runner: OnPolicyRunner) -> bool:
    """Return True when current run uses wandb logger in old/new rsl_rl APIs."""
    if getattr(runner, "logger_type", None) == "wandb":
        return True
    logger = getattr(runner, "logger", None)
    if logger is not None and getattr(logger, "logger_type", None) == "wandb":
        return True
    return wandb.run is not None


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


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if not _is_wandb_logging_enabled(self):
            return

        # Link the motion artifact to this run first. This keeps play.py compatible even if export hooks fail.
        if self.registry_name is not None and wandb.run is not None:
            try:
                wandb.run.use_artifact(self.registry_name)
            except Exception as err:
                print(f"[WARN]: Failed to link motion artifact to run: {err}")
            finally:
                self.registry_name = None

        policy_path = path.split("model")[0]
        filename = policy_path.split("/")[-2] + ".onnx"
        policy, normalizer = _get_policy_and_normalizer(self)
        if policy is None:
            print("[WARN]: Skip ONNX export because policy handle is unavailable.")
            return

        try:
            export_policy_as_onnx(policy, normalizer=normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name if wandb.run else "none", path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
        except Exception as err:
            print(f"[WARN]: ONNX export hook failed during save, continue training: {err}")


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if not _is_wandb_logging_enabled(self):
            return

        policy_path = path.split("model")[0]
        filename = policy_path.split("/")[-2] + ".onnx"
        policy, normalizer = _get_policy_and_normalizer(self)
        if policy is None:
            print("[WARN]: Skip ONNX export because policy handle is unavailable.")
            return

        try:
            export_motion_policy_as_onnx(
                self.env.unwrapped, policy, normalizer=normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name if wandb.run else "none", path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
        except Exception as err:
            print(f"[WARN]: ONNX export hook failed during save, continue training: {err}")
