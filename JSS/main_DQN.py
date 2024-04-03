import time
import ray
import wandb
import random
import numpy as np
import ray.tune.integration.wandb as wandb_tune
from ray.rllib.agents.dqn import DQNTrainer
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

# Main training function
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
        'instance_path': str(instance_path.absolute()),
        'evaluation_interval': None,
        'metrics_smoothing_episodes': 2000,
        'gamma': 0.99,
        'num_workers': mp.cpu_count(),
        'buffer_size': 1000000,
        'learning_starts': 10000,
        'lr_start': 0.0006861,  # TO TUNE
        'lr_end': 0.00007783,  # TO TUNE
        'train_batch_size': mp.cpu_count() * 4 * 64,
        'layer_size': 256,  # Default size for each layer
        'layer_nb': 2,
        'timesteps_per_iteration': 1000,
        'lr': 1e-4,
        'entropy_start': 0.0002458,
        'entropy_end': 0.002042,
        #'entropy_coeff': 0.0002458,  # TUNE LATER
        #'entropy_coeff_schedule': None,
        'exploration_config': {
            'type': 'EpsilonGreedy',
            'initial_epsilon': 1.0,
            'final_epsilon': 0.02,
            'epsilon_timesteps': 10000,
        },
        # DQN-specific configurations
        'target_network_update_freq': 10000,
        'dueling': False,
        'double_q': False,
        'n_step': 1,
        'noisy': False,
        'prioritized_replay': False,
        'model': {
            'fcnet_activation': 'relu',
            'fcnet_hiddens': [256, 256],  # Adjust to your desired architecture
            'custom_model': 'fc_masked_model_tf',
            'vf_share_layers': False,
            'num_outputs': 2,
        }
    }

    # Initialize W&B
    wandb.init(project='JSS_DQN_server', config=default_config, name="ta01")
    ray.init()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Get the current config
    config = wandb.config

    # Register custom model
    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

    # Update model config
    new_model_config= {
    "fcnet_activation": "relu",
    "custom_model": "fc_masked_model_tf",
    'fcnet_hiddens': [32, 16], 
    #'no_final_linear':True, # Adjust to your desired architecture
    "vf_share_layers": False,
   # "num_outputs": 2,  # Number of discrete actions
}
    #config['model'] = new_model_config
    config.update({'model': new_model_config}, allow_val_change=True)
 

    config['env_config'] = {
        'env_config': {'instance_path': config['instance_path']}
    }
    
    config = with_common_config(config)
    config['seed'] = 0
    config['callbacks'] = CustomCallbacks

 # Set train_batch_size to default if sgd_minibatch_size is not specified
    if 'sgd_minibatch_size' not in config:
        config['train_batch_size'] = default_config['train_batch_size']
    else:
        config['train_batch_size'] = config['sgd_minibatch_size']

    
    config['lr'] = config['lr_start']
    config['lr_schedule'] = [[0, config['lr_start']], [15000000, config['lr_end']]]
    config.pop('instance_path', None)
    config.pop('layer_size', None)
    config.pop('layer_nb', None)
    config.pop('lr_start', None)
    config.pop('lr_end', None)
    config.pop('entropy_start', None)
    config.pop('entropy_end', None)

    

    # Stop criteria
    stop = {
        "time_total_s": 10 * 60,  # The training loop runs for a total time of 10 minutes
    }

    start_time = time.time()
    trainer = DQNTrainer(config=config)
    while start_time + stop['time_total_s'] > time.time():
        result = trainer.train()
        result = wandb_tune._clean_log(result)
        log, config_update = _handle_result(result)
        wandb.log(log)

    # Clean up
    ray.shutdown()

# Call the training function
if __name__ == "__main__":
    train_func()
