import time
import ray
import wandb
import random
import numpy as np
import ray.tune.integration.wandb as wandb_tune
from ray.rllib.agents.dqn.simple_q import SimpleQTrainer
from CustomCallbacks import *
import multiprocessing as mp
from ray.rllib.agents import with_common_config
from ray.rllib.utils.framework import try_import_tf
import pathlib
from typing import Dict, Tuple

tf1, tf, tfv = try_import_tf()

# Set up W&B logging
wandb.login()

# List of keys to exclude from logging to W&B
_exclude_results = ["done", "should_checkpoint", "config"]
_config_results = [
    "trial_id", "experiment_tag", "node_ip", "experiment_id", "hostname",
    "pid", "date",
]

# Function to handle logging results
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
        elif not wandb_tune._is_allowed_type(v):
            continue
        else:
            log[k] = v

    config_update.pop("callbacks", None)  # Remove callbacks
    return log, config_update

def train_func():
    current_path = pathlib.Path(__file__).parent
    instance_path = current_path.parent / "JSS" / "instances" / "ta10"
    if not instance_path.exists():
        raise FileNotFoundError(f"Path {instance_path} does not exist")

    default_config = {
        'env': 'JSSEnv:jss-v1',
        'seed': 0,
        'framework': 'tf',
        'log_level': 'WARN',
        'num_gpus': 1,
        'instance_path': str(instance_path.absolute()),
        'evaluation_interval': None,
        'metrics_smoothing_episodes': 2000,
        'gamma': 0.99,  # Discount factor
        'num_workers': mp.cpu_count(),
        'rollout_fragment_length': 50,  # Length of fragments to train on
        'train_batch_size': 128,  # Batch size for each training iteration
        'learning_starts': 1000,  # Number of steps before learning starts
        'buffer_size': 10000,  # Size of the replay buffer
        'target_network_update_freq': 500,  # How often to update the target network
        'lr_start': 0.001,  # Starting learning rate
        'lr_end': 0.0001,  # Ending learning rate
     }

    wandb.init(project='JSS_SimpleQTrainer_server', config=default_config, name="ta10")
    ray.init()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = wandb.config

    config = with_common_config(config)
    config['seed'] = 0
    config['callbacks'] = CustomCallbacks
    #config['train_batch_size'] = config['sgd_minibatch_size']

    config['lr'] = config['lr_start']
    config['lr_schedule'] = [[0, config['lr_start']], [15000000, config['lr_end']]]

  

    config.pop('instance_path', None)
    config.pop('lr_start', None)
    config.pop('lr_end', None)
   
    stop = {
        "time_total_s": 10 * 60,  # The training loop runs for a total time of 10 minutes
    }

    start_time = time.time()
    agent = SimpleQTrainer(config=config, env=config["env"])
    while start_time + stop['time_total_s'] > time.time():
        result = agent.train()
        log, config_update = _handle_result(result)
        wandb.log(log)

    ray.shutdown()


if __name__ == "__main__":
    train_func()
