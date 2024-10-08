# Single-agent example

import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import train, tune


if __name__ == "__main__":
    ray.init()

    resources = ray.available_resources()
    num_gpus = resources.get("GPU", 0)

    env_name = "CartPole-v1"

    config = (
        PPOConfig()
        .environment(env_name)
        .resources(num_gpus=num_gpus)
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=train.RunConfig(
            stop={"timesteps_total": 10000},
            storage_path="~/ray_results/" + env_name
        ),
    )

    tuner.fit()

    ray.shutdown()