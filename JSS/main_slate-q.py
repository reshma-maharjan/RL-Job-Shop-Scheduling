import time
import ray
import wandb
import random
import numpy as np
from ray.rllib.agents.qmix import QMixTrainer
from CustomCallbacks import CustomCallbacks
from models import FCMaskedActionsModelTF
from typing import Dict, Tuple
import multiprocessing as mp
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog
from ray.tune.utils import flatten_dict
import pathlib

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
        elif not wandb.util.is_allowed_value(v):
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
        # SlateQ (QMix) specific configurations
        'n_rollouts': 2,  # Number of rollouts
        'num_atoms': 1,  # Number of atoms in QMix
        'seq_len': 32,  # Length of rollout sequence
        'update_steps': 2,  # Update frequency for target network
        'batch_size': 64,  # Batch size for QMix
        'minibatch_size': 32,  # Minibatch size for QMix
        'buffer_size': 1000000,  # Buffer size for replay buffer
        'exploration_final_eps': 0.02,  # Final epsilon for exploration
        'exploration_fraction': 0.1,  # Fraction of total steps for exploration decay
        'prioritized_replay': False,  # Prioritized replay
        'prioritized_replay_alpha': 0.6,  # Alpha for prioritized replay
        'prioritized_replay_beta': 0.4,  # Beta for prioritized replay
        'final_prioritized_replay_beta': 0.4,  # Final beta for prioritized replay
        'prioritized_replay_eps': 1e-6,  # Epsilon for prioritized replay
        'model': {
            'fcnet_activation': 'relu',
            'fcnet_hiddens': [256, 256],  # Adjust to your desired architecture
            'custom_model': 'fc_masked_model_tf',
            'vf_share_layers': False,
            'num_outputs': 2,  # Number of discrete actions
        }
    }

    # Initialize W&B
    wandb.init(project='JSS_SlateQ_server', config=default_config, name="ta01")
    ray.init()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Get the current config
    config = wandb.config

    # Register custom model
    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

    # Update model config
    new_model_config = {
        "fcnet_activation": "relu",
        "custom_model": "fc_masked_model_tf",
        'fcnet_hiddens': [32, 16],
        'vf_share_layers': False,
        'num_outputs': 2,  # Number of discrete actions
    }
    config.update({'model': new_model_config}, allow_val_change=True)

    config['env_config'] = {
        'env_config': {'instance_path': config['instance_path']}
    }

    config = with_common_config(config)
    config['seed'] = 0
    config['callbacks'] = CustomCallbacks

    # Set train_batch_size to default if minibatch_size is not specified
    if 'minibatch_size' not in config:
        config['train_batch_size'] = default_config['train_batch_size']
    else:
        config['train_batch_size'] = config['minibatch_size']

    # Stop criteria
    stop = {
        "time_total_s": 10 * 60,  # The training loop runs for a total time of 10 minutes
    }

    start_time = time.time()
    trainer = QMixTrainer(config=config)
    while start_time + stop['time_total_s'] > time.time():
        result = trainer.train()
        log, config_update = _handle_result(result)
        wandb.log(log)

    # Clean up
    ray.shutdown()

# Call the training function
if __name__ == "__main__":
    train_func()
