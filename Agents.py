import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment import AllocationEnv

class Config():

    def __init__(self, total_attention_units, n_envs):
        self.empirical_lambda = [15.44662005, 17.28904429, 18.81165501, 19.28857809, 19.2979021, 19.46666667,
                                19.65314685, 19.99207459, 20.02937063, 21.31888112, 21.82983683, 22.48904429,
                                22.92634033, 23.28018648, 23.34312354, 23.82610723, 24.44242424, 26.77902098,
                                27.74545455, 28.79160839, 32.48018648] ## TODO: change this from hardcoded
        self.total_attention_units = total_attention_units
        self.n_envs = n_envs

class RL_Agent():
    
    def __init__():
        pass


class greedy_Agent():

    def __init__(self,total_attention_units=1000):
        self.env_config = Config(total_attention_units=total_attention_units, n_envs=1)
        self.env = AllocationEnv(total_attention_units=self.env_config.total_attention_units,
                                  incident_rate_by_areas=self.env_config.empirical_lambda, timesteps=5)
        


    def take_action(self):
        #get observation of number of incidents
        obs, info = self.env.reset()
        # print(obs)
        curr_incidents = obs[:,0]
        total_attention_available = self.env.total_attention_units
        total_incidents = sum(curr_incidents)
        # print(f"total incidents: {total_incidents}")
        # print(curr_incidents)
        
        #allocate officers proportionally
        actions = []
        for location in curr_incidents:
            action = (location/total_incidents) * total_attention_available
            # print(f"location: {location} action{action}")
            actions.append(action)
        return actions
    

def simulate():
    Agent = greedy_Agent()
    config = Config(total_attention_units=1000, n_envs=1)
    env = AllocationEnv(total_attention_units=config.total_attention_units, incident_rate_by_areas=config.empirical_lambda, timesteps=5)
    done = False
    while not done:
        actions = Agent.take_action()
        obs, reward, done, truncated, info = env.step(actions=actions)
        print(f"Fairness: {info["Fairness"]}\n Total_incidents: {info["Total_incidents"]}")
        # print(f"Allocated units: {actions}, Reward: {reward}")


def main():
    simulate()

if __name__ == "__main__":
    main()