import time

import ray
import wandb

import random

import numpy as np

import ray.tune.integration.wandb as wandb_tune

#from ray.rllib.agents.dqn import R2D2Trainer

from CustomCallbacks import *
from models import *

from typing import Dict, Tuple

import multiprocessing as mp
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog

from ray.tune.utils import flatten_dict
from ray.rllib.utils.framework import try_import_tf
import pathlib

from ray.rllib.agents.r2d2 import R2D2Trainer, DEFAULT_CONFIG as R2D2_CONFIG
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf

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
        'target_network_update_freq': 500,  # How often to update the target network
        'buffer_size': 10000,  # Size of the replay buffer
        'prioritized_replay_alpha': 0.6,  # Alpha parameter for prioritized replay
        'prioritized_replay_beta': 0.4,  # Beta parameter for prioritized replay
        'final_prioritized_replay_beta': 0.4,  # Final value of beta
        'prioritized_replay_beta_annealing_timesteps': 20000,  # Annealing steps for beta
        'n_step': 3,  # N-step updates
        'dueling': False,  # Dueling architecture
        'noisy': False,  # Noisy nets
        'double_q': True,  # Double Q-learning
        'num_atoms': 1,  # Distributional Q function
        'v_min': -10.0,  # Distributional Q function min value
        'v_max': 10.0,  # Distributional Q function max value
        'noisy_std': 0.1,  # Std for noisy nets
    }

    class LSTMPolicy(RecurrentTFModelV2):
        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            super(LSTMPolicy, self).__init__(obs_space, action_space, num_outputs, model_config, name)
            self.cell_size = 256  # You can change this as needed

            self.model = FullyConnectedNetwork(
                obs_space,
                action_space,
                num_outputs,
                model_config,
                name,
                fcnet_hiddens=[256],
                fcnet_activation="relu",
                use_lstm=True,
                lstm_cell_size=self.cell_size
            )

            self.register_variables(self.model.variables())

        def forward_rnn(self, inputs, state, seq_lens):
            return self.model.forward_rnn(inputs, state, seq_lens)

        def get_initial_state(self):
            return self.model.get_initial_state()

    wandb.init(project='JSS_R2D2_server', config=default_config, name="ta10")
    ray.init()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = wandb.config

    # Register the custom LSTM model
    ModelCatalog.register_custom_model("LSTMPolicy", LSTMPolicy)

    config['model'] = {
        "custom_model": "LSTMPolicy",
    }

    config['num_workers'] = 2  # Set the number of workers to 2 or more for R2D2

    config.update(R2D2_CONFIG)  # Update config with R2D2 default settings

    wandb.init(project='JSS_PPO_server', config=config, name="ta43")
    ray.init()
    tf1.set_random_seed(0)
    np.random.seed(0)
    random.seed(0)
    config['callbacks'] = CustomCallbacks
    config['train_batch_size'] = config['sgd_minibatch_size']

    config['lr'] = config['lr_start']
    config['lr_schedule'] = [[0, config['lr_start']], [15000000, config['lr_end']]]

    config['entropy_coeff'] = config['entropy_start']
    config['entropy_coeff_schedule'] = [[0, config['entropy_start']], [15000000, config['entropy_end']]]

    config.pop('instance_path', None)
    config.pop('layer_size', None)
    config.pop('layer_nb', None)
    config.pop('lr_start', None)
    config.pop('lr_end', None)
    config.pop('entropy_start', None)
    config.pop('entropy_end', None)

    stop = {
        "time_total_s": 10 * 60,  # The training loop runs for a total time of 10 minutes
    }

    start_time = time.time()
    trainer = R2D2Trainer(config=config)
    while start_time + stop['time_total_s'] > time.time():
        result = trainer.train()
        result = wandb_tune._clean_log(result)
        log, config_update = _handle_result(result)
        wandb.log(log)

    ray.shutdown()


if __name__ == "__main__":
    train_func()