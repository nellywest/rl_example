import functools
import random
import numpy as np
from gymnasium.spaces import Box, Discrete, Dict
from pettingzoo import ParallelEnv
import random


class PrisonerGuardEnv(ParallelEnv):
    """The metadata holds environment constants."""
    metadata = {
        "name": "prisoner_guard_env_v0",
    }
    
    def __init__(self, config):
        """The init method takes in environment arguments from a YAML configuration file.

        Should define the following attributes:
        - escape x and y coordinates
        - guard x and y coordinates
        - prisoner x and y coordinates
        - timestamp
        - possible_agents

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.escape_y = config.get('escape_y', None)
        self.escape_x = config.get('escape_x', None)
        self.guard_y = config.get('guard_y', None)
        self.guard_x = config.get('guard_x', None)
        self.prisoner_y = config.get('prisoner_y', None)
        self.prisoner_x = config.get('prisoner_x', None)
        self.timestep = config.get('timestep', None)
        self.possible_agents = config.get('possible_agents', ["prisoner", "guard"])
        self.grid_size = config.get('grid_size', 7)

    def reset(self, seed=None, options=None):
        """Reset the environment to the starting state."""
        self.agents = self.possible_agents.copy()
        self.timestep = 0

        # Initial positions of the agents
        self.prisoner_x = 0
        self.prisoner_y = 0

        self.guard_x = self.grid_size - 1
        self.guard_y = self.grid_size - 1

        # Random escape position
        self.escape_x = random.randint(2, 5)
        self.escape_y = random.randint(2, 5)

        observation = np.array([
            self.prisoner_x,
            self.prisoner_y,
            self.guard_x,
            self.guard_y,
            self.escape_x,
            self.escape_y
        ])

        # Observations are represented as a flat numpy array
        observations = {
            "prisoner": observation,
            "guard": observation
        }

        # Dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        """Perform actions and update the environment state."""
        # Execute actions
        prisoner_action = actions["prisoner"]
        guard_action = actions["guard"]

        # Update prisoner position based on action
        if prisoner_action == 0 and self.prisoner_x > 0:
            self.prisoner_x -= 1
        elif prisoner_action == 1 and self.prisoner_x < self.grid_size - 1:
            self.prisoner_x += 1
        elif prisoner_action == 2 and self.prisoner_y > 0:
            self.prisoner_y -= 1
        elif prisoner_action == 3 and self.prisoner_y < self.grid_size - 1:
            self.prisoner_y += 1

        # Update guard position based on action
        if guard_action == 0 and self.guard_x > 0:
            self.guard_x -= 1
        elif guard_action == 1 and self.guard_x < self.grid_size - 1:
            self.guard_x += 1
        elif guard_action == 2 and self.guard_y > 0:
            self.guard_y -= 1
        elif guard_action == 3 and self.guard_y < self.grid_size - 1:
            self.guard_y += 1

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
            print("Guard catches the prisoner!", self.timestep)
            rewards = {"prisoner": -1, "guard": 1}
            terminations = {a: True for a in self.agents}
        elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
            print("Prisoner escapes!", self.timestep)
            rewards = {"prisoner": 1, "guard": -1}
            terminations = {a: True for a in self.agents}

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > 100:
            rewards = {"prisoner": 0, "guard": 0}
            truncations = {"prisoner": True, "guard": True}
        self.timestep += 1

        observation = np.array([
            self.prisoner_x,
            self.prisoner_y,
            self.guard_x,
            self.guard_y,
            self.escape_x,
            self.escape_y
        ])

        # Get observations as numpy arrays
        observations = {
            "prisoner": observation,
            "guard": observation,
        }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        # TO DO: This may lead to bugs
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""
        grid = np.full((self.grid_size, self.grid_size), " ")
        grid[self.prisoner_y, self.prisoner_x] = "P"
        grid[self.guard_y, self.guard_x] = "G"
        grid[self.escape_y, self.escape_x] = "E"
        print(f"{grid} \n")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Define the observation space using Box."""
        return Box(low=0, high=6, shape=(6,), dtype=np.int64)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """Define the action space as a Discrete space."""
        return Discrete(4)  # 4 possible actions: left, right, up, down
    

def env_creator(config):
    env = PrisonerGuardEnv(config)
    return env