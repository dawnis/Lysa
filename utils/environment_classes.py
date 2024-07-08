from graphviz import Source
from IPython.display import display
from collections import defaultdict

import gymnasium as gym
import numpy as np
import random

global obs_vector
obs_vector = ["xpos", "ypos", "xvel", "yvel", "l1", "angle", "angular_vel", "l2"]

def visualize_gen(filename):
    """Visualize the graph for a particular generation"""
    # Path to your .dot file
    file_path = f"{filename}.dot"
    source = Source.from_file(file_path)
    display(source)
    output_path = source.render(filename=file_path, format='png')
    return file_path

def map_observation_8d(observation, obs_map):
    """Creates 8-dimensional vector from observation map to feed into NN"""
    # defaults to 0 if the key is not present in a particular envioronment's observation space
    return [observation[obs_map[x]] if x in obs_map.keys() else 0 for x in obs_vector]

class GymnasiumCCEnv:

    def __init__(self, environment_name, configuration, obs_space_dict, action_map_dict, agent_fitness_default=0):
        self.env_name = environment_name
        self.agent_fitness_default = agent_fitness_default
        self.obs_map = obs_space_dict
        self.action_dict = action_map_dict
        self.env = gym.make(environment_name)
        self.evaluation_steps = 999
        self.observation_steps = 999
        self.configuration = configuration

    def calculate_reward_delta(self, observation, reward):
        return reward

    def termination_penalty(self):
        return 0

    def assign_fitness(self, population, agent_idx, fitness):
        fitness -= 0.1 * population.agent_complexity(agent_idx)
        if fitness < 0:
            population.set_agent_fitness(agent_idx, random.uniform(0, 1))
        else:
            population.set_agent_fitness(agent_idx, fitness)
        return

    def evaluate_agent(self, population, agent_idx):
        observation, info = self.env.reset()

        # In this task, there is a negative penalty for each step so start with the fitness goal
        fitness = self.agent_fitness_default

        for step in range(self.evaluation_steps):
            mapped_observation = map_observation_8d(observation, self.obs_map)
            population.agent_fwd(agent_idx, mapped_observation)
            action_raw = population.agent_out(agent_idx)
            action_discrete = self.map_action(action_raw[0])
            observation, reward, terminated, truncated, info = self.env.step(action_discrete)

            # if step % 10 == 0:
            #     print(f"Observation: {observation}")
            #     print(f"Mapped Observation: {mapped_observation}")
            #     print(f"Raw output {action_raw}, chosen action: {action_discrete}")

            reward_delta = self.calculate_reward_delta(observation, reward)

            # if step % 10 == 0:
            #     print(f"Observed reward is {reward_delta} for {observation}")

            if reward_delta == reward_delta:
                fitness += reward_delta #forward progress + clipped reward penalty - complexity penalty

            if fitness != fitness:
                print(f"Error! Observing NaN Fitness")

            if terminated or truncated:
                self.assign_fitness(population, agent_idx, fitness)
                break

        #print(f"Setting agent {agent_idx} fitness to {fitness}")
        self.assign_fitness(population, agent_idx, fitness)

    def map_action (self, nn_output):
        """Maps output of NN to a discrete number for Gymanisum"""

        if nn_output < -0.5:
            return self.action_dict["left"]
        elif nn_output > 0.5:
            return self.action_dict["right"]
        elif (np.abs(nn_output) <= 0.5) & (np.abs(nn_output) > 0.1):
            return self.action_dict["up"]
        else:
            return self.action_dict["noop"]

    def observe(self, population, idx):
        self.env = gym.make(self.env_name, render_mode="human")
        agent_idx = idx
        observation, info = self.env.reset()

        for _ in range(self.evaluation_steps):
            mapped_observation = map_observation_8d(observation, self.obs_map)
            population.agent_fwd(agent_idx, mapped_observation)
            action_raw = population.agent_out(agent_idx)
            action_discrete = self.map_action(action_raw[0])
            observation, reward, terminated, truncated, info = self.env.step(action_discrete)

            if terminated or truncated:
                observation, info = self.env.reset()

        self.env.close()

    def write_agent(self, population, agent_idx, file_save_path):
        population.agent_checkpt(agent_idx, file_save_path)

class AcrobotEnvironment(GymnasiumCCEnv):

    def __init__(self, configuration):
        obs_space_dictionary = {
            "xpos": 0,
            "ypos": 1,
            "xvel": 2,
            "yvel": 3,
            "angle": 4,
            "angular_vel": 5,
        }

        action_map_dictionary = {
            "left": 0,
            "noop": 1,
            "right": 2
        }
        acrobat_action = defaultdict(lambda: 1, action_map_dictionary)

        super().__init__("Acrobot-v1", configuration, obs_space_dictionary, acrobat_action, agent_fitness_default=500)

class PendulumEnvironment(GymnasiumCCEnv):

    def __init__(self, configuration):
        obs_space_dictionary = {
            "xpos": 0,
            "ypos": 1,
            "angular_vel": 2,
        }

        action_map_dictionary = {}
        pendulum_action = defaultdict(lambda: 0, action_map_dictionary)

        super().__init__("Pendulum-v1", configuration, obs_space_dictionary, pendulum_action, agent_fitness_default=1000)

    def map_action(self, nn_output):
        return [2*nn_output]

class CartPoleEnvironment(GymnasiumCCEnv):

    def __init__(self, configuration):
        obs_space_dictionary = {
            "xpos": 0,
            "xvel": 1,
            "angle": 2
        }

        action_map_dictionary = {
            "left": 0,
            "right": 1
        }
        cartpole_action = defaultdict(lambda: 0, action_map_dictionary)
        super().__init__("CartPole-v1", configuration, obs_space_dictionary, cartpole_action)

class MountainCarEnvironment(GymnasiumCCEnv):

    def __init__(self, configuration):
        obs_space_dictionary = {
            "xpos": 0,
            "xvel": 1,
        }
        action_map_dictionary = {
            "left": 0,
            "noop": 1,
            "right": 2,
        }
        mc_action = defaultdict(lambda: 1, action_map_dictionary)
        super().__init__("MountainCar-v0", configuration, obs_space_dictionary, mc_action, agent_fitness_default=200)

    def calculate_reward_delta(self, observation, reward):
        postional_reward = max([0, observation[0] + 0.4])
        fwd_velocity_reward = max([0, observation[1]])
        return (postional_reward + fwd_velocity_reward)*10 + float(reward)

