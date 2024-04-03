import time
import wandb
import random
import numpy as np
import ray
from ray.rllib.agents.marwil import MARWILTrainer
from CustomCallbacks import *
from models import *
from typing import Dict, Tuple
import multiprocessing as mp
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog
from ray.tune.utils import flatten_dict
from ray.rllib.utils.framework import try_import_tf
import pathlib

tf1, tf, tfv = try_import_tf()

_exclude_results = ["done", "should_checkpoint", "config"]

# Use these result keys to update `wandb.config`
_config_results = [
    "trial_id", "experiment_tag", "node_ip", "experiment_id", "hostname",
    "pid", "date",
]

def _handle_result(result: Dict) -> Tuple[Dict, Dict]:
    config_update = result.get("config", {}).copy()
    log = {}
    flat_result = flatten_dict(result, delimiter="/")

    for k, v in flat_result.items():
        if any(
                k.startswith(item + "/") or k == item
                for item in _config_results):
            config_update[k] = v
        elif any(
                k.startswith(item + "/") or k == item
                for item in _exclude_results):
            continue
        elif not wandb.util.is_allowed_value(v):  # Use wandb.util.is_allowed_value instead
            continue
        else:
            log[k] = v

    config_update.pop("callbacks", None)  # Remove callbacks
    return log, config_update

def train_func():
    current_path = pathlib.Path(__file__).parent
    instance_path = current_path.parent / "JSS" / "instances" / "ta01"
    if not instance_path.exists():
        raise FileNotFoundError(f"Path {instance_path} does not exist")
    
    default_config = {
        'env': 'JSSEnv:jss-v1',
        'seed': 0,
        'framework': 'tf',
        'log_level': 'WARN',
        'env_config': {
            'instance_path': str(instance_path.absolute())
        },
        'num_gpus': 1,
        'evaluation_interval': None,
        'metrics_smoothing_episodes': 2000,
        'gamma': 1.0,
        'num_workers': mp.cpu_count(),
        'rollout_fragment_length': 704,  # TO TUNE
        'lr': 0.0006861,  # TO TUNE
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "_fake_gpus": False,
    }

    wandb.init(project='JSS_MARWIL_server', config=default_config, name="ta01")
    ray.init()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = wandb.config

    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

    config['model'] = {
        "fcnet_activation": "relu",
        "custom_model": "fc_masked_model_tf",
        'fcnet_hiddens': [319 for _ in range(2)],  # Using 319 as a placeholder
        "vf_share_layers": False,
    }
    # Create env_config if it doesn't exist
    if 'env_config' not in config:
        config['env_config'] = {}
    
    config['env_config']['instance_path'] = str(instance_path.absolute())
    
    config = with_common_config(config)
    config['seed'] = 0
    config['train_batch_size'] = 704 * 32  # Placeholder value

    stop = {
        "time_total_s": 2 * 60, # The training loop runs for a total time of 3 minutes
    }

    start_time = time.time()
    trainer = MARWILTrainer(config=config)
    while start_time + stop['time_total_s'] > time.time():
        result = trainer.train()
        log, config_update = _handle_result(result)
        wandb.log(log)
        # wandb.config.update(config_update, allow_val_change=True)
    
    ray.shutdown()

if __name__ == "__main__":
    train_func()


