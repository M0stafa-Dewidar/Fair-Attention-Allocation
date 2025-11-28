import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment import AllocationEnv
from torch import nn
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

  
LAPD_AREA_NAMES = ["Foothill", "Hollenbeck", "Mission", "Topanga", "Harbor", "Devonshire",
                    "West Valley", "Van Nuys", "Northeast", "West LA", "Rampart", "Wilshire",
                      "Newton", "Southeast", "Olympic", "North Hollywood",
                    "West Hollywood", "Southwest", "Pacific", "7th Street", "Central"]



empirical_lambda = [15.44662005, 17.28904429, 18.81165501, 19.28857809, 19.2979021, 19.46666667,
                                19.65314685, 19.99207459, 20.02937063, 21.31888112, 21.82983683, 22.48904429,
                                22.92634033, 23.28018648, 23.34312354, 23.82610723, 24.44242424, 26.77902098,
                                27.74545455, 28.79160839, 32.48018648]github

# empirical_lambda = [1, 2, 4, 6, 8, 10, 12,
#                                 14, 16, 18, 20, 22, 24,
#                                 26, 28, 30, 32, 34, 36,
#                                 38, 40]


class Config():

    def __init__(self, total_attention_units, n_envs):
        self.empirical_lambda = empirical_lambda 
        self.total_attention_units = total_attention_units
        self.n_envs = n_envs

TIMESTEPS = 200
TOTAL_ATTENTION_UNITS = 400

config = Config(total_attention_units=TOTAL_ATTENTION_UNITS, n_envs=1)
env = AllocationEnv(total_attention_units=config.total_attention_units, incident_rate_by_areas=config.empirical_lambda, timesteps=TIMESTEPS)

for i in range(21):
    LAPD_AREA_NAMES[i] = LAPD_AREA_NAMES[i] + "  lam=" + str(int(empirical_lambda[i])) + "  val=" + str(int(env.locations[i].curr_value/1000))

class RL_Agent():
    
    def __init__():
        pass


class greedy_Agent():

    def __init__(self,total_attention_units=20):
        self.total_attention_units = total_attention_units


    def take_action(self, env):
        self.env = env
        #get observation of number of incidents
        obs, info = self.env.reset()
        curr_incidents = obs
        total_attention_available = self.env.total_attention_units
        total_incidents = sum(curr_incidents)
        
        #allocate officers proportionally
        print(f"current incidents:   {curr_incidents}")
        actions = []
        for location in curr_incidents:
            action = (location/total_incidents)
            actions.append(action)
        return actions


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 21),
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



class NNAgent():
    def __init__(self):
        self.model = NeuralNetwork()
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.epochs = 5
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def take_action(self, env):
        obs = env._get_obs()
        # obs = obs/sum(obs)
        obs = torch.tensor(obs)
        self.model.train()
        action_pred = self.model(obs)
        print(f"Action Pred: {action_pred}")
        loss = self.loss_fn(action_pred, obs)

        loss.backward()
        CLIP_VALUE = 0.5
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=CLIP_VALUE)

        self.optimizer.step()
        self.optimizer.zero_grad()

        loss = loss.item()
        print(f"loss: {loss:>7f}")
        return action_pred


def simulate():
    Agent = greedy_Agent(total_attention_units=TOTAL_ATTENTION_UNITS)
    done = False
    timestep = 0
    timestep_timeseries = []
    Fairness_time_series = []
    reward_time_series = []
    Discovery_time_series = []
    Discovery_by_location = np.zeros(shape=(TIMESTEPS,21))
    while not done:
        actions = Agent.take_action(env)
        obs, reward, done, truncated, info = env.step(actions=actions)
        print(f"Fairness: {info["Fairness"]}\n Total_incidents: {info["Total_incidents"]}")
        reward_time_series.append(float(reward))
        Fairness_time_series.append(float(info["Fairness"]))
        Discovery_time_series.append(int(info["Discovered"]))
        Discovery_by_location[timestep:] = [float(x) for x in info["coverage_rate_array"]]
        timestep_timeseries.append(timestep)
        timestep += 1
    
    for i in range(21):
        location_discovery_time_series = Discovery_by_location[:,i]
        rolling_average = location_discovery_time_series[0]
        total = location_discovery_time_series[0]
        for j in range(1,len(location_discovery_time_series)):
            rolling_average = (0.999 * total / (j)) + (0.001 * location_discovery_time_series[j])
            total += location_discovery_time_series[j]
            location_discovery_time_series[j] = rolling_average
        Discovery_by_location[:,i] = location_discovery_time_series
    
    plt.plot(timestep_timeseries, Discovery_by_location, label=LAPD_AREA_NAMES)
    plt.xlabel("Timestep")
    plt.ylabel("Coverage rate")
    plt.title("Coverage rate by area")
    plt.legend()
    plt.show()

    # plt.plot(timestep_timeseries, Discovery_by_location[:,5:10], label=LAPD_AREA_NAMES[5:10])

    # plt.title("Discovery rate by area")
    # plt.legend()
    # plt.show()

    # plt.plot(timestep_timeseries, Discovery_by_location[:,10:15], label=LAPD_AREA_NAMES[10:15])

    # plt.title("Discovery rate by area")
    # plt.legend()
    # plt.show()

    # plt.plot(timestep_timeseries, Discovery_by_location[:,15:21], label=LAPD_AREA_NAMES[15:21])

    # plt.title("Discovery rate by area")
    # plt.legend()
    # plt.show()

    # plt.plot(timestep_timeseries, Fairness_time_series)
    # plt.show()
    # plt.plot(timestep_timeseries, reward_time_series)
    # plt.show()
    # plt.plot(timestep_timeseries, Discovery_time_series)
    # plt.show()



def main():
    simulate()

if __name__ == "__main__":
    main()