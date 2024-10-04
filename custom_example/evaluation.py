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
            env.render()

            # Accumulate rewards for each agent
            for agent_id in env.possible_agents:
                episode_rewards[agent_id] += rewards[agent_id]
                
            # If all agents are done, break the loop
            if any(terminations.values()) or all(truncations.values()):
                break
        
        # Print rewards for the episode
        print(f"Episode {episode + 1} rewards: {episode_rewards}")
        
        # Add episode rewards to total rewards
        for agent_id in env.possible_agents:
            total_rewards[agent_id] += episode_rewards[agent_id]
    
    # Print the final total rewards after all episodes
    print("Evaluation complete.")
    print(f"Total rewards after {num_episodes} episodes: {total_rewards}")