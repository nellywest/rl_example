import yaml
from ray.tune import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPO


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return f"{agent_id}_policy"


def load_agent(train_config, env_creator, model, checkpoint):
    # Important: you have to register env and model before loading agent!
    env_name = train_config['environment']
    register_env(env_name, lambda env_config: ParallelPettingZooEnv(env_creator(env_config)))
    ModelCatalog.register_custom_model(train_config['training']['model']['custom_model'], model)
    return PPO.from_checkpoint(checkpoint)