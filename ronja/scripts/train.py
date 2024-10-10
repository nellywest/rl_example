import ray
import argparse
from ray.tune import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.rllib.models import ModelCatalog
from ronja.models.custom_model import PrisonerGuardModel
from prisoner_pettingzoo_env.prisoner_env import env_creator
from ronja.scripts.utils import policy_mapping_fn, load_yaml


def train_model(env_config, train_config, env_creator, model, policy_mapping_fn):
    env_name = train_config['environment']
    register_env(env_name, lambda env_config: ParallelPettingZooEnv(env_creator(env_config)))

    ModelCatalog.register_custom_model(train_config['training']['model']['custom_model'], model)
    
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
                    env_creator(env_config).observation_space("prisoner"),
                    env_creator(env_config).action_space("prisoner"),
                    {},
                ),
                "guard_policy": (
                    None,
                    env_creator(env_config).observation_space("guard"),
                    env_creator(env_config).action_space("guard"),
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


def main():
    parser = argparse.ArgumentParser(description="Ronja training script")
    parser.add_argument('-e', '--env', type=str, required=True, help="Path to environment YAML config")
    parser.add_argument('-t', '--train', type=str, required=True, help="Path to training YAML config")
    
    args = parser.parse_args()
    
    env_config = load_yaml(args.env)
    train_config = load_yaml(args.train)
    
    train_model(env_config, train_config, env_creator, PrisonerGuardModel, policy_mapping_fn)


if __name__ == "__main__":
    main()