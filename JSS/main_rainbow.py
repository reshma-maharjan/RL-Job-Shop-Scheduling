import time
import wandb
import random
import numpy as np
import ray
from ray.rllib.agents import RainbowTrainer
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
        elif not wandb.util.is_allowed_value(v):
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
        'num_gpus': 1,
        'evaluation_interval': None,
        'metrics_smoothing_episodes': 2000,
        'gamma': 0.99,  # Rainbow uses a different gamma value
        'num_workers': mp.cpu_count(),
        'buffer_size': 50000,  # Rainbow uses a replay buffer
        'train_batch_size': 32,  # Rainbow uses smaller batch sizes
        'rollout_fragment_length': 50,  # Rainbow uses smaller rollout fragment lengths
        'learning_starts': 1000,  # Rainbow starts learning after 1000 steps
        'target_network_update_freq': 500,  # Rainbow updates target network less frequently
        'prioritized_replay': True,  # Rainbow typically uses prioritized replay
        'prioritized_replay_alpha': 0.6,  # Rainbow prioritized replay alpha
        'prioritized_replay_beta': 0.4,  # Rainbow prioritized replay beta
        'final_prioritized_replay_beta': 0.4,  # Rainbow final prioritized replay beta
        'prioritized_replay_eps': 1e-6,  # Rainbow prioritized replay epsilon
        'dueling': True,  # Rainbow uses dueling architecture
        'double_q': True,  # Rainbow uses double Q-learning
        'n_step': 3,  # Rainbow uses n-step returns
        'noisy': False,  # Rainbow does not use noisy networks by default
        'v_min': -10.0,  # Rainbow value distribution minimum
        'v_max': 10.0,  # Rainbow value distribution maximum
        'num_atoms': 51,  # Rainbow number of atoms in value distribution
        'lr': 0.0001,  # Rainbow typically uses lower learning rates
        'adam_epsilon': 1e-6,  # Rainbow Adam epsilon
        'grad_clip': 0.5,  # Rainbow gradient clipping
        'entropy_coeff': 0.01,  # Rainbow entropy coefficient
        'model': {
        "fcnet_activation": "relu",
        "custom_model": "fc_masked_model_tf",
        'fcnet_hiddens': [319 for _ in range(2)],  # Using 319 as a placeholder
        "vf_share_layers": False,
        },
        'env_config': {
        'instance_path': str(instance_path.absolute())
        },
        'seed': 0,
        }

    wandb.init(project='JSS_Rainbow_server', config=default_config, name="ta01")
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
    config['env_config'] = {
        'instance_path': str(instance_path.absolute())
    }

    config = with_common_config(config)
    config['seed'] = 0
    config['train_batch_size'] = 704 * 32  # Placeholder value

    stop = {
        "time_total_s": 3 * 60, # The training loop runs for a total time of 3 minutes
    }

    start_time = time.time()
    trainer = RainbowTrainer(config=config)
    while start_time + stop['time_total_s'] > time.time():
        result = trainer.train()
        log, config_update = _handle_result(result)
        wandb.log(log)
        # wandb.config.update(config_update, allow_val_change=True)
    
    ray.shutdown()

if __name__ == "__main__":
    train_func()
