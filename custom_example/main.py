import ray
import argparse
from ray.tune import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from custom_model import CustomModel
from custom_env import env_creator
from evaluation import evaluate_parallel_env
from pathlib import Path


parser = argparse.ArgumentParser(description ='Train or evaluate custom example')
parser.add_argument("--checkpoint", type=Path)


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return f"{agent_id}_policy"


if __name__ == "__main__":
    args = parser.parse_args()

    env_name = "custom_env"
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("CustomModel", CustomModel)
    
    if args.checkpoint:
        # Just evaluate!

        agent = PPO.from_checkpoint(args.checkpoint)
        env = env_creator({})
        evaluate_parallel_env(env, policy_mapping_fn, agent, num_episodes=2, max_steps=100)
    else:
        # Train!

        ray.init()

        resources = ray.available_resources()
        num_gpus = resources.get("GPU", 0)

        config = (
            PPOConfig()
            .environment("custom_env")
            .framework("torch")
            .multi_agent(
                # (policy_cls, obs_space, act_space, config)
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
            .training(model={"custom_model": "CustomModel"})
            .resources(num_gpus=num_gpus)
        )

        tune.run(
            "PPO",
            name="PPO",
            stop={"timesteps_total": 100000},
            checkpoint_freq=1,
            storage_path="~/ray_results/" + env_name,
            config=config.to_dict(),
        )