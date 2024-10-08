import time
import argparse
from train import load_yaml, policy_mapping_fn
from ray.tune import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from prisoner_pettingzoo_env.prisoner_env import env_creator
from ray.rllib.algorithms.ppo import PPO
from ronja.models.custom_model import CustomModel


def evaluate_parallel_env(env, policy_mapping_fn, agent, num_episodes=10, max_steps=100):
    # Store total rewards for all agents across episodes
    total_rewards = {agent_id: 0 for agent_id in env.possible_agents}
    
    # Loop over episodes
    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}/{num_episodes}")
        
        # Reset the environment at the beginning of each episode
        observations, _ = env.reset()
        
        # Store rewards for this episode
        episode_rewards = {agent_id: 0 for agent_id in env.possible_agents}
        
        # Loop over steps in the episode
        for step in range(max_steps):
            actions = {}
            for agent_id, obs in observations.items():
                policy_id = policy_mapping_fn(agent_id, None, None)
                policy = agent.get_policy(policy_id)
                action, _, _ = policy.compute_single_action(obs)
                actions[agent_id] = action
            
            # Step the environment with the actions
            observations, rewards, terminations, truncations, _ = env.step(actions)

            print(observations)
            time.sleep(1)
            env.render()

            # Accumulate rewards for each agent
            for agent_id in env.possible_agents:
                episode_rewards[agent_id] += rewards[agent_id]
                
            # If all agents are done, break the loop
            if any(terminations.values()) or all(truncations.values()):
                break
        
            time.sleep(1)

        # Print rewards for the episode
        print(f"Episode {episode + 1} rewards: {episode_rewards}")
        
        # Add episode rewards to total rewards
        for agent_id in env.possible_agents:
            total_rewards[agent_id] += episode_rewards[agent_id]

        time.sleep(1)
    
    # Print the final total rewards after all episodes
    print("Evaluation complete.")
    print(f"Total rewards after {num_episodes} episodes: {total_rewards}")


def main():
    parser = argparse.ArgumentParser(description="Ronja evaluation script")
    parser.add_argument('-e', '--env', type=str, required=True, help="Path to environment YAML config")
    parser.add_argument('-t', '--train', type=str, required=True, help="Path to training YAML config")
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help="Path to training YAML config")

    args = parser.parse_args()
    
    env_config = load_yaml(args.env)
    train_config = load_yaml(args.train)
    
    env_name = train_config['environment']
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model(train_config['training']['model']['custom_model'], CustomModel)

    agent = PPO.from_checkpoint(args.checkpoint)
    env = env_creator({})
    evaluate_parallel_env(env, policy_mapping_fn, agent, num_episodes=2, max_steps=100)


if __name__ == "__main__":
    main()