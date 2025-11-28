'''
This Module defines the environment for the simulation

'''

from typing import Optional
import numpy as np
import gymnasium as gym
import pandas as pd
from gymnasium.spaces import Dict, Box, Discrete
import matplotlib.pyplot as plt
from scipy.special import softmax
import torch

LAPD_AREA_NAMES = ["Foothill", "Hollenbeck", "Mission", "Topanga", "Harbor", "Devonshire",
                    "West Valley", "Van Nuys", "Northeast", "West LA", "Rampart", "Wilshire",
                      "Newton", "Southeast", "Olympic", "North Hollywood",
                    "West Hollywood", "Southwest", "Pacific", "7th Street", "Central"]

#controlling randomness of numpy generators
np.random.seed(13)

class Location():
    def __init__(self, name="default name", curr_incident_rate=0, curr_value=10000):
        self.location_name = name
        self.curr_incident_rate = curr_incident_rate
        self.curr_incidents = self.produce_incidents(self.curr_incident_rate)
        self.curr_attention = self.get_curr_attention(100) #TODO change this hardcoded value
        self.curr_value = curr_value

    def produce_incidents(self, incident_rate):
        return np.random.poisson(incident_rate)
    
    def get_curr_attention(self, range):
        return np.random.uniform(low=0, high=range)
    
    def set_attention(self, attention):
        self.curr_attention = attention

    def get_location_info(self):
        print(f"\
Location Name:         {self.location_name}\n\
Current Incident Rate: {self.curr_incident_rate}\n\
Current Incidents:     {self.curr_incidents}\n\
Current Attention:     {self.curr_attention}\n\
Current Value:         {self.curr_value}\n\
        ")

class AllocationEnv(gym.Env):

    '''
    Like all environments, our custom environment will inherit from gymnasium.Env
    that defines the structure all environments must follow. One of the requirements
    is defining the observation and action spaces, which declare what inputs (actions)
    and outputs (observations) are valid for this environment. 
    '''
    def __init__(self, incident_rate_by_areas, total_attention_units=100, timesteps=100):
        
        self.incident_rate_by_areas = incident_rate_by_areas
        self.time_step = 0
        self.timesteps = timesteps
        self.total_attention_units = total_attention_units
        self.locations = self.initialize_locations_random()

        #Observation Space
        self.observation_space = gym.spaces.Box(low=0, high=self.total_attention_units, shape=(21,), dtype=np.float32)

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(21,), dtype=np.float32)



    def initialize_locations_random(self):
        locations = []
        for i in range(len(LAPD_AREA_NAMES)):
            location = LAPD_AREA_NAMES[i]
            incident_rate = np.random.poisson(self.incident_rate_by_areas[i])
            curr_location_value = np.random.uniform(low=150000, high=750000)
            new_location = Location(name=location, curr_incident_rate=incident_rate, curr_value=curr_location_value)
            locations.append(new_location)
        
        #set initial attention for new locations randomly
        total_remaining_attention = self.total_attention_units
        allocations = np.zeros(21)
        for attention_unit in range(self.total_attention_units):
            locate_to = int(np.floor(np.random.uniform(low=0, high=20.99))) #RANDOM ALLOCATION
            allocations[locate_to] += 1
        
        for i in range(len(locations)):
            location = locations[i]
            location.set_attention(allocations[i])

        return locations

    '''
    Since we need to compute observations in both Env.reset() and Env.step(),
    it is convenient to have a helper method _get_obs that translates the environment
    internal state into the observation format. This keeps our code DRY 
    (Dont Repeat Yourself) and makes it easier to modify the observation format later. 
    '''
    def _get_obs(self):
        observation_map = np.zeros((21,),dtype=np.float32)
        for i in range(len(self.locations)):
            location = self.locations[i]
            observation_map[i] = location.curr_incidents
            # observation_map[i,1] = location.curr_value
            # observation_map[i,2] = location.curr_attention
        print(observation_map)
        return observation_map

    '''
    We can also implement a similar method for auxiliary information returned by
    Env.reset() and Env.step(). this can be useful for debugging and understanding
    agent progress, but shouldnt be used by the learning algorithm itself.
    '''
    def _get_info(self, verbose=1):
        #Compute the Value and fairness metrics
        Value = 0
        Fairness = 0
        Discovered = 0
        Total_incidents = 0
        Highest_Discovery_Rate = 0
        Lowest_Discovery_Rate = 100
        coverage_rate_arr = []
        for i in range(len(self.locations)):
            location = self.locations[i]
            Discovered += min(location.curr_attention, location.curr_incidents)
            Total_incidents += location.curr_incidents
            Value += location.curr_value
            coverage_rate_i = min(location.curr_attention / (location.curr_incidents + 0.0000001),1)
            coverage_rate_arr.append(coverage_rate_i)
            if coverage_rate_i > Highest_Discovery_Rate:
                Highest_Discovery_Rate = coverage_rate_i
            if coverage_rate_i < Lowest_Discovery_Rate:
                Lowest_Discovery_Rate = coverage_rate_i
        Fairness = Highest_Discovery_Rate - Lowest_Discovery_Rate
        if verbose:
            print(f"\
{'LOCATION NAME':<17}           {'INCIDENTS_NOW':<17}           {'ATTENTION_NOW':<17}\n")
            for location in self.locations:
                print(f"\
{location.location_name:<17}                    {location.curr_incidents:<17}                   {location.curr_attention:<17}")

            print(f"\
Current Fariness:     {Fairness}\n\
Current Value:        {Value}\n\
Current Discovered:   {Discovered}\n\
Total Incidents:      {Total_incidents}\n\
Highest Discovery %   {Highest_Discovery_Rate}\n\
Lowest Discovery %    {Lowest_Discovery_Rate}\n")
               
        return {"Fairness": Fairness,
                "Value": Value,
                "Discovered": Discovered,
                "Total_incidents": Total_incidents,
                "coverage_rate_array": coverage_rate_arr}
    

    '''
    The reset() method starts a new episode. 
    It takes two optional parameters: seed for reproducible 
    random generation and options for additional configuration.
    On the first line, you must call super().reset(seed=seed) 
    to properly initialize the random number generator. 
    '''
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        # self.time_step = 0
        # self.locations = self.initialize_locations_random()
        observations = self._get_obs()
        info = self._get_info()
        return observations, info


    '''
    The step() method contains the core environment logic. 
    It takes an action, updates the environment state, and returns the results.
    This is where the world rules, and reward logic live 
    '''
    def step(self, actions):
        # allocations = actions
        # total_action_norm = sum(actions)
        # m = torch.nn.Softmax()
        print(f"actions:    {actions}")
        action_probs = actions
        # scaled_actions = actions / total_action_norm * self.total_attention_units
        for i in range(len(action_probs)):
            # print(f"i:    {i}")
            curr_action = action_probs[i] * self.total_attention_units
            curr_location = self.locations[i]
            curr_location.curr_attention = curr_action
            self.locations[i] = curr_location
            self.locations[i].curr_attention = curr_action
        
        self.time_step += 1

        terminated = True if self.time_step == self.timesteps else False
        
        truncated = False

        observations = self._get_obs()
        info = self._get_info()

        # coverage_arr_np = info["coverage_rate_arr"]
        # print(type(coverage_arr_np[0]))
        # coverage_arr_np = torch.tensor(coverage_arr_np)

        # safety_std_dev = torch.std(coverage_arr_np)
        # safety_mean = torch.mean(coverage_arr_np)
        safety_threshold = 0.8

        for location in self.locations:
            location.curr_incidents = location.produce_incidents(location.curr_incident_rate)
            # location.curr_value = min(location.curr_value - 0.1 * location.curr_value * (safety_mean - (max(1,location.curr_attention / location.curr_incidents))) / safety_std_dev,1000000)
            if location.curr_attention / location.curr_incidents < safety_threshold:
                self.total_attention_units = self.total_attention_units - (0.01 * (location.curr_value / 200000))
            else:
                self.total_attention_units = self.total_attention_units + (0.01 * (location.curr_value / 200000))
            self.total_attention_units = min(600,max(200, self.total_attention_units))


        
        # new_total_attention = 0
        # Cost = 35000
        # for location in self.locations:
            # new_total_attention += (location.curr_value / Cost)
        
        # print("NEW ATTENTION TOTAL")
        # print(new_total_attention)
        # 
        # self.total_attention_units = new_total_attention

        penalty = 0
        for loc in self.locations:
            missed = max(0, loc.curr_incidents - loc.curr_attention)
            # Squaring the missed incidents makes big failures hurt MUCH more
        penalty -= (missed ** 2)
        
        Utility_weight = 1
        fairness_weight = 0
        penalty_weight = 0

        # loc0_attention = self.locations[0].curr_attention
        # reward = 100 * loc0_attention - (self.total_attention_units - loc0_attention)
        reward = (Utility_weight * info["Discovered"] / info["Total_incidents"]) - (fairness_weight * info["Fairness"]) + penalty_weight * penalty
        print(f"Current Reward:     {reward}")
        return observations, reward, terminated, truncated, info

def main():
    TestHarness()

def TestHarness():
    # test1()
    # test2()
    # create_env()
    # get_env_obs()
    # get_env_info()
    # make_env_step()
    # run_random_allocation_simulation()
    pass


'''
Create a location object and print its state
'''
def test1():
    North_Hollywood = Location(name="N Hollywood", curr_incident_rate=20)
    North_Hollywood.get_location_info()


def test2():
    Center = Location(name="Center", curr_incident_rate=35)
    Center.get_location_info()
    Center.set_attention(attention=35)
    Center.get_location_info()

def create_env():
    env = AllocationEnv(incident_rate_by_areas=10, total_attention_units=100)

def get_env_obs():
    env = AllocationEnv(incident_rate_by_areas=10, total_attention_units=100)
    obs = env._get_obs() 

def get_env_info():
    env = AllocationEnv(incident_rate_by_areas=10, total_attention_units=100)
    info = env._get_info()

def make_env_step():
    env = AllocationEnv(incident_rate_by_areas=10, total_attention_units=100)
    actions = np.zeros(21)
    for attention_unit in range(300):
        # allocation = int(np.random.uniform(low=0, high=21))
        allocation = min(int(np.random.uniform(low=0, high=20)),20)
        actions[allocation] += 1
    env.step(actions=actions)

def random_unifrom_allocation(actions, total_attention_units):
    for attention_unit in range(total_attention_units):
        allocation = int(np.random.uniform(low=0, high=len(actions)-1))
        actions[allocation] += 1
    return actions

def run_random_allocation_simulation(total_attention_units=300):
    env = AllocationEnv(incident_rate_by_areas=10, total_attention_units=total_attention_units)
    timestep = 0
    Fairness_time_series = []
    Discovered_time_series = []
    time_step_series = []
    while timestep < 20:
        time_step_series.append(timestep)
        actions = np.zeros(21)
        new_actions = random_unifrom_allocation(actions, total_attention_units)
        env.step(actions=new_actions)
        Fairness, _ , Discovered = env._get_info()
        Fairness_time_series.append(Fairness)
        Discovered_time_series.append(Discovered)
        timestep += 1
    plt.plot(time_step_series, Fairness_time_series, label="Fairness")
    # plt.plot(time_step_series, Discovered_time_series, label="Discovered")
    # plt.plot(time_step_series, Fairness, label="Fairness")
    plt.show()

    

if __name__ == "__main__":
    main()

