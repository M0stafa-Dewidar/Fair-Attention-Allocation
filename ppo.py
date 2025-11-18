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

def main():
    #ENVIRONMENT SETUP
    expirement_config = Config(total_attention_units=400, n_envs=4)
    vec_env = make_vec_env(lambda: AllocationEnv(incident_rate_by_areas = expirement_config.empirical_lambda, total_attention_units=expirement_config.total_attention_units, timesteps=1000), n_envs=expirement_config.n_envs)

    #MODEL SETUP
    policy_kwargs = dict(net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128]))   #set up 3-layer NN for policy and value networks
    model = PPO(policy="MlpPolicy", env=vec_env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=10000) ##TODO: change the number of time steps!!

    env = AllocationEnv(incident_rate_by_areas = expirement_config.empirical_lambda, total_attention_units=expirement_config.total_attention_units)
    obs, info = env.reset()
    
    done = False

    #Simulate?
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Allocated units: {action}, Reward: {reward}")



if __name__ == "__main__":
    main()


