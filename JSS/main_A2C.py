import time
import ray
import wandb
import random
import numpy as np
import ray.tune.integration.wandb as wandb_tune
from ray.rllib.agents.a3c import A2CTrainer
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
        elif not wandb_tune._is_allowed_type(v):
            continue
        else:
            log[k] = v

    config_update.pop("callbacks", None)  # Remove callbacks
    return log, config_update

def train_func(instance_ind, project_name=None):
    assert instance_ind is not None, "instance_ind must be provided"

    current_path = pathlib.Path(__file__).parent
    instance_path = current_path.parent / "JSS" / "instances" / f"la{instance_ind}"

    if not instance_path.exists():
        raise FileNotFoundError(f"Path {instance_path} does not exist")
    default_config = {
        'env': 'JSSEnv:jss-v1',
        'seed': 0,
        'framework': 'tf',
        'log_level': 'WARN',
        'num_gpus': 6,
        'instance_path': str(instance_path.absolute()),#'./instances/ta01',
        'evaluation_interval': None,
        'metrics_smoothing_episodes': 2000,
        'gamma': 1.0,
        'num_workers': mp.cpu_count(),
        'layer_nb': 2,
        'num_envs_per_worker': 4,
        'rollout_fragment_length': 704,  # TO TUNE
        'layer_size': 319,
        'lr': 0.0006861,  # TO TUNE
        'lr_start': 0.0006,  # TO TUNE
        'lr_end': 0.000078,  # TO TUNE
        "vf_loss_coeff": 0.7918,
        'lambda': 1.0,
        'entropy_coeff': 0.0002,  # TUNE LATER
        'entropy_start': 0.00025,
        'entropy_end': 0.002042,
        "batch_mode": "truncate_episodes",
        "grad_clip": 40.0,
        "use_critic": True,
        "use_gae": True,
        "observation_filter": "NoFilter",
        "_fake_gpus": False,
    }

    wandb.init(project=project_name, config=default_config, name=f"la{instance_ind}", group="JSS", job_type=f"training_{instance_ind}")
    ray.init()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = wandb.config

    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

    config['model'] = {
        "fcnet_activation": "relu",
        "custom_model": "fc_masked_model_tf",
        'fcnet_hiddens': [config['layer_size'] for k in range(config['layer_nb'])],
        "vf_share_layers": False,
    }
    config['env_config'] = {
        'env_config': {'instance_path': config['instance_path']}
    }

    config = with_common_config(config)
    config['seed'] = 0
    config['callbacks'] = CustomCallbacks
    config['train_batch_size'] = config['rollout_fragment_length'] * config['num_workers']

    config['lr'] = config['lr_start']
    config['lr_schedule'] = [[0, config['lr_start']], [15000000, config['lr_end']]]

    config['entropy_coeff'] = config['entropy_start']
 

    config.pop('instance_path', None)
    config.pop('layer_size', None)
    config.pop('layer_nb', None)
    config.pop('lr_start', None)
    config.pop('lr_end', None)
    config.pop('entropy_start', None)
    config.pop('entropy_end', None)

    iterations = 60  # Change this to run for 80 iterations


    trainer = A2CTrainer(config=config)
    for _ in range(iterations):
        result = trainer.train()
        result = wandb_tune._clean_log(result)
        log, config_update = _handle_result(result)
        wandb.log(log)

    wandb.finish()

    ray.shutdown()

if __name__ == "__main__":

    for ind in [20,25,30,35,40]:
    
        ind = str(ind).zfill(2)
        print(f"Training instance {ind}")
        train_func(instance_ind=ind, project_name="A2C_la_60")
      
        time.sleep(10)