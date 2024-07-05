from graphviz import Source
from IPython.display import display
from collections import defaultdict

import gymnasium as gym
import numpy as np
import random

def visualize_gen(filename):
    """Visualize the graph for a particular generation"""
    # Path to your .dot file
    file_path = f"{filename}.dot"
    source = Source.from_file(file_path)
    display(source)
    output_path = source.render(filename=file_path, format='png')
    return file_path

class GymnasiumCCEnv:

    def __init__(self, environment_name, configuration, obs_space_dict, action_map_dict):
        self.env_name = environment_name
        self.obs_map = obs_space_dict
        self.action_dict = action_map_dict
        self.env = gym.make(environment_name)
        self.evaluation_steps = 999
        self.observation_steps = 999
        self.configuration = configuration
        self.obs_vector = ["xpos", "ypos", "xvel", "yvel", "l1", "angl", "angl_vel", "l2"]

    def map_obs8(self, observation):
        """Creates 8-dimensional vector from observation map to feed into NN"""
        # defaults to 0 if the key is not present in a particular envioronment's observation space
        return [observation[self.obs_map[x]] if x in self.obs_map.keys() else 0 for x in self.obs_vector]


    def map_action (self, nn_output):
        """Maps output of NN to a discrete number for Gymanisum"""

        if nn_output < -10:
            return self.action_dict["left"]
        elif nn_output > 10:
            return self.action_dict["right"]
        elif (np.abs(nn_output) <= 10) & (np.abs(nn_output) > 3 ):
            return self.action_dict["up"]
        else:
            return self.action_dict["noop"]

    def observe(self, population, idx):
        self.env = gym.make(self.env_name, render_mode="human")
        agent_idx = idx
        observation, info = self.env.reset()

        for _ in range(self.evaluation_steps):
            population.agent_fwd(agent_idx, list(observation))
            action_raw = population.agent_out(agent_idx)
            action = self.map_action(action_raw[0])
            observation, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                observation, info = self.env.reset()

        self.env.close()

    def write_agent(self, population, agent_idx, file_save_path):
        population.agent_checkpt(agent_idx, file_save_path)

class CartPoleEnvironment(GymnasiumCCEnv):

    def __init__(self, configuration):
        obs_space_dictionary = {
            "x1": 0,
            "vel": 1,
            "angle": 2
        }

        action_map_dictionary = {
            "left": 0,
            "right": 1
        }
        cartpole_action = defaultdict(lambda: 0, action_map_dictionary)
        super().__init__("CartPole-v1", configuration, obs_space_dictionary, cartpole_action)

    def evaluate_agent(self, population, agent_idx):
        observation, info = self.env.reset()
        fitness = 0
        termination_penalty = min([100 * population.generation + 1, 1000])
        step = 0

        for _ in range(self.evaluation_steps):
            step += 1
            mapped_observation = self.map_obs8(observation)
            population.agent_fwd(agent_idx, mapped_observation)
            action_raw = population.agent_out(agent_idx)
            action_discrete = self.map_action(action_raw[0])
            observation, reward, terminated, truncated, info = self.env.step(action_discrete)

            reward_delta = reward #sometimes there are NaNs, which crash the program

            if (reward_delta == reward_delta) & (step > 9):
                fitness += reward_delta #forward progress + clipped reward penalty - complexity penalty

            if fitness != fitness:
                print(f"Error! Observing NaN Fitness")

            if terminated or truncated:
                #penalize every termination with a constant penalty
                step = 0
                new_fitness = fitness - termination_penalty
                fitness = max([new_fitness, 0])

                observation, info = self.env.reset()

        #print(f"Setting agent {agent_idx} fitness to {fitness}")

        fitness -= 0.1 * population.agent_complexity(agent_idx)
        
        if fitness < 0:
            population.set_agent_fitness(agent_idx, random.uniform(0, 1))
        else:
            population.set_agent_fitness(agent_idx, fitness)

    
class MountainCarEnvironment(GymnasiumCCEnv):

    def __init__(self, configuration):
        obs_space_dictionary = {
            "x1": 0,
            "vel": 1,
        }
        action_map_dictionary = {
            "left": 0,
            "noop": 1,
            "right": 2,
        }
        mc_action = defaultdict(lambda: 0, action_map_dictionary)
        super().__init__("MountainCar-v0", configuration, obs_space_dictionary, mc_action)

    def evaluate_agent(self, population, agent_idx):
        observation, info = self.env.reset()
        fitness = 0

        for _ in range(self.evaluation_steps):
            mapped_observation = self.map_obs8(observation)
            population.agent_fwd(agent_idx, mapped_observation)
            action_raw = population.agent_out(agent_idx)
            action_discrete = self.map_action(action_raw[0])
            observation, reward, terminated, truncated, info = self.env.step(action_discrete)

            reward_delta = max([0, mapped_observation[0]])*10 + reward #sometimes there are NaNs, which crash the program

            if reward_delta == reward_delta:
                fitness += reward_delta #forward progress + clipped reward penalty - complexity penalty

            if fitness != fitness:
                print(f"Error! Observing NaN Fitness")
                
            if terminated or truncated:
                observation, info = self.env.reset()

        #print(f"Setting agent {agent_idx} fitness to {fitness}")

        fitness -= 0.1 * population.agent_complexity(agent_idx)
        
        if fitness < 0:
            population.set_agent_fitness(agent_idx, random.uniform(0, 1))
        else:
            population.set_agent_fitness(agent_idx, fitness)

    
