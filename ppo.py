import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from environment import AllocationEnv  
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO





class Config():

    def __init__(self, total_attention_units, n_envs):
        self.empirical_lambda = [15.44662005, 17.28904429, 18.81165501, 19.28857809, 19.2979021, 19.46666667,
                                19.65314685, 19.99207459, 20.02937063, 21.31888112, 21.82983683, 22.48904429,
                                22.92634033, 23.28018648, 23.34312354, 23.82610723, 24.44242424, 26.77902098,
                                27.74545455, 28.79160839, 32.48018648]
        self.total_attention_units = total_attention_units
        self.n_envs = n_envs

def main():
    #ENVIRONMENT SETUP
    expirement_config = Config(total_attention_units=450, n_envs=1)
    vec_env = make_vec_env(lambda: AllocationEnv(incident_rate_by_areas = expirement_config.empirical_lambda, total_attention_units=expirement_config.total_attention_units, timesteps=10000), n_envs=expirement_config.n_envs)

    #MODEL SETUP
    policy_kwargs = dict(net_arch=dict(pi=[16, 16, 16], vf=[16, 16, 16]))   #set up 3-layer NN for policy and value networks
    model = PPO(policy="MlpPolicy", env=vec_env, ent_coef=0.05, policy_kwargs=policy_kwargs, verbose=1)
    # model = RecurrentPPO("MlpLstmPolicy", env=vec_env, verbose=1)
    # model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=1e-5, ent_coef=0.01)
    model.learn(total_timesteps=10000) ##TODO: change the number of time steps

    env = AllocationEnv(incident_rate_by_areas = expirement_config.empirical_lambda, total_attention_units=expirement_config.total_attention_units, timesteps=1000)
    obs, info = env.reset()
    
    done = False

    reward_time_series = []
    timestep = 0
    timestep_series = []
    total_reward = 0
    #Simulate?
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Allocated units: {action}, Reward: {reward}")
        if timestep % 1 == 0:
            reward_time_series.append(reward)
            timestep_series.append(timestep/100)
        total_reward += reward
        timestep += 1
    
    average_reward = total_reward / timestep
    print(f"Average Reward: {average_reward}")
    plt.plot(timestep_series, reward_time_series)
    plt.show()






if __name__ == "__main__":
    main()


