import os
import sys
from collections.abc import Mapping

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from isaacgymenvs.utils.reformat import omegaconf_to_dict

# Add parent directory for package imports.
par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if par_dir not in sys.path:
    sys.path.insert(0, par_dir)

if not OmegaConf.has_resolver('eq'):
    OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
    OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
    OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
    OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)


CONFIG_DIR = os.path.join(os.path.dirname(__file__), "cfg")
TASK_CONFIG_ALIASES = {
    "MetaWorldV2": "meta-world-v2",
}


def _compose_task_config(task: str) -> dict:
    """Load the task config without depending on the legacy trainer."""
    task_config_name = TASK_CONFIG_ALIASES.get(task, task)
    if HydraConfig.initialized() or GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=[f"task={task_config_name}"])
    return omegaconf_to_dict(cfg.task)


def _task_config_from_cfg(cfg, task: str) -> dict:
    if cfg is None:
        return _compose_task_config(task)
    if isinstance(cfg, DictConfig) and "task" in cfg:
        return omegaconf_to_dict(cfg.task)
    if isinstance(cfg, Mapping) and "task" in cfg:
        return omegaconf_to_dict(OmegaConf.create(cfg["task"]))
    if hasattr(cfg, "task"):
        return omegaconf_to_dict(cfg.task)
    return _compose_task_config(task)


def _apply_env_overrides(
    task_config: dict,
    *,
    seed: int,
    num_envs: int,
    tasks,
    task_env_count,
):
    env_cfg = task_config["env"]
    env_cfg["numEnvs"] = num_envs
    env_cfg["tasks"] = tasks if tasks is not None else env_cfg["tasks"]
    env_cfg["taskEnvCount"] = (
        task_env_count if task_env_count is not None else env_cfg["taskEnvCount"]
    )
    env_cfg["seed"] = seed

    for key in (
        "episodeLength",
        "fixed",
        "same_init_config_per_task",
        "termination_on_success",
        "reward_scale",
        "sparse_reward",
    ):
        if key in task_config:
            env_cfg[key] = task_config[key]

    if "record_videos" in task_config and "enableDebugVis" in env_cfg:
        env_cfg["enableDebugVis"] = bool(task_config["record_videos"])
    if "exempted_tasks" in task_config:
        env_cfg["exemptedInitAtRandomProgressTasks"] = task_config["exempted_tasks"]


def _rank_devices(sim_device: str, rl_device: str, multi_gpu: bool, task_config: dict):
    if not multi_gpu:
        return sim_device, rl_device

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    global_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    print(f"global_rank = {global_rank} local_rank = {local_rank} world_size = {world_size}")

    rank_device = f"cuda:{local_rank}"
    task_config["rank"] = local_rank
    task_config["rl_device"] = rank_device
    return rank_device, rank_device


def make(
    seed: int, 
    task: str, 
    num_envs: int, 
    sim_device: str,
    rl_device: str,
    graphics_device_id: int = -1,
    headless: bool = False,
    multi_gpu: bool = False,
    virtual_screen_capture: bool = False,
    force_render: bool = True,
    cfg: DictConfig = None,

    tasks: list = None,
    taskEnvCount: list = None,
    custom_args = None
): 
    from isaacgymenvs.tasks import isaacgym_task_map

    cfg_dict = _task_config_from_cfg(cfg, task)

    if custom_args is not None:
        custom_args_dict = custom_args.__dict__ if not isinstance(custom_args, dict) else custom_args
        cfg_dict.update(custom_args_dict)

    _apply_env_overrides(
        cfg_dict,
        seed=seed,
        num_envs=num_envs,
        tasks=tasks,
        task_env_count=taskEnvCount,
    )

    sim_device, rl_device = _rank_devices(sim_device, rl_device, multi_gpu, cfg_dict)
    return isaacgym_task_map[cfg_dict["name"]](
        cfg=cfg_dict,
        rl_device=rl_device,
        sim_device=sim_device,
        graphics_device_id=graphics_device_id,
        headless=headless,
        virtual_screen_capture=virtual_screen_capture,
        force_render=force_render,
    )
