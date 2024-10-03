import numpy as np


def evaluate_parallel_env(env, policy_mapping_fn, PPOagent, num_episodes=10, max_steps=100):
    # Store total rewards for all agents across episodes
    total_rewards = {agent: 0 for agent in env.possible_agents}
    
    # Loop over episodes
    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}/{num_episodes}")
        
        # Reset the environment at the beginning of each episode
        observations, _ = env.reset()
        
        # Store rewards for this episode
        episode_rewards = {agent: 0 for agent in env.agents}
        
        # Loop over steps in the episode
        for step in range(max_steps):
            actions = {}
            for agent_id, obs in observations.items():
                policy_id = policy_mapping_fn(agent_id, None, None)
                policy = PPOagent.get_policy(policy_id)
                obs = np.array([obs])
                action, state_out, info = policy.compute_actions(obs)
                actions[agent] = action
            
            # Step the environment with the actions
            observations, rewards, dones, infos = env.step(actions)
            
            # Accumulate rewards for each agent
            for agent in env.agents:
                episode_rewards[agent] += rewards[agent]
                
            # If all agents are done, break the loop
            if all(dones.values()):
                break
        
        # Print rewards for the episode
        print(f"Episode {episode + 1} rewards: {episode_rewards}")
        
        # Add episode rewards to total rewards
        for agent in env.agents:
            total_rewards[agent] += episode_rewards[agent]
    
    # Print the final total rewards after all episodes
    print("Evaluation complete.")
    print(f"Total rewards after {num_episodes} episodes: {total_rewards}")