import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment import AllocationEnv  

empirical_lambda = np.array([2.1, 0.7, 5.3, ...]) ## TODO: change this vector into the empirical lambdas we get from dataset
vec_env = make_vec_env(lambda: AllocationEnv(incident_rate_by_areas = empirical_lambda, total_attention_units=400), n_envs=4)
policy_kwargs = dict(net_arch=[dict(policyNet=[128, 128, 128], valueNet=[128, 128, 128])])   #set up 3-layer NN for policy and value networks

model = PPO(policy="MlpPolicy", env=vec_env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=5000) ##TODO: change the number of time steps!!

env = AllocationEnv(incidence_rate_by_area=empirical_lambda, total_attention_unit=400)
obs, info = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated = env.step(action)
    print(f"Allocated units: {action}, Reward: {reward}")

      
    
  """def GAE(self, rewards, values, dones, last_value):
      advantages = []
      gae = 0
      l = len(rewards)
      for t in reversed(range(l)):
          delta = rewards[t] + self.gamma * (1 - dones[t]) * last_value - values[t]
          gae = delta + self.gamma * self.lambda * (1 - dones[t]) * gae
          advantages.insert(0, gae)
          last_value = values[t]
      return torch.tensor(advantages, dtype=torch.float32)


