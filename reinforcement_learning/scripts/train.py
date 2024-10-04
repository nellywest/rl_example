import ray
import argparse
from ray.tune import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from pathlib import Path
import yaml

from reinforcement_learning.models.custom_model import CustomModel
from prisoner_pettingzoo_env.prisoner_env import env_creator
from reinforcement_learning.scripts.evaluate import evaluate_parallel_env

# Load configuration from YAML file
with open('reinforcement_learning/configs/train_config.yaml', 'r') as file:
    train_config = yaml.safe_load(file)

parser = argparse.ArgumentParser(description='Train or evaluate custom example')
parser.add_argument("--checkpoint", type=Path)


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return f"{agent_id}_policy"

def train_model():
    args = parser.parse_args()

    env_name = train_config['environment']
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model(train_config['training']['model']['custom_model'], CustomModel)
    
    if args.checkpoint:
        # Just evaluate!

        PPOagent = PPO.from_checkpoint(args.checkpoint)
        env = env_creator({})
        evaluate_parallel_env(env, policy_mapping_fn, PPOagent, num_episodes=10, max_steps=100)
    else:
        # Train!

        ray.init()

        resources = ray.available_resources()
        num_gpus = resources.get("GPU", 0)

        config = (
            PPOConfig()
            .environment(train_config['environment'])
            .framework(train_config['framework'])
            .multi_agent(
                policies={
                    "prisoner_policy": (
                        None,
                        env_creator({}).observation_space("prisoner"),
                        env_creator({}).action_space("prisoner"),
                        {},
                    ),
                    "guard_policy": (
                        None,
                        env_creator({}).observation_space("guard"),
                        env_creator({}).action_space("guard"),
                        {},
                    ),
                },
                policy_mapping_fn=policy_mapping_fn
            )
            .training(model={"custom_model": train_config['training']['model']['custom_model']})
            .resources(num_gpus=num_gpus)
        )

        tune.run(
            "PPO",
            name="PPO",
            stop={"timesteps_total": train_config['stop']['timesteps_total']},
            checkpoint_freq=train_config['checkpoint_freq'],
            storage_path=train_config['storage_path'],
            config=config.to_dict(),
        )