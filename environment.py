'''
This Module defines the environment for the simulation

'''

from typing import Optional
import numpy as np
import gymnasium as gym
import pandas as pd
from gymnasium.spaces import Dict, Box, Discrete
import matplotlib.pyplot as plt

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
        
        self.time_step = 0
        self.timesteps = timesteps
        self.total_attention_units = total_attention_units
        self.locations = self.initialize_locations_random()

        #Observation Space
        observation_space = Dict({
            "1": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "2": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "3": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "4": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "5": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "6": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "7": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "8": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "9": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "10": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "11": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "12": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "13": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "14": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "15": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "16": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "17": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "18": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "19": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "20": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
            "21": Dict({
                "incidents": Box(low=0, high=100, shape=(), dtype=np.float32),
                "incident_rate": Box(low=0, high=100, shape=(), dtype=np.float32),
                "curr_value": Box(low=0, high=1000000, shape=(), dtype=np.float32),
                "curr_attention_units": Box(low=0, high=400, shape=(), dtype=np.float32),
            }),
        })

        self.action_space = gym.spaces.Box(low=0, high=100, shape=(21,), dtype=np.float32)



    def initialize_locations_random(self):
        locations = []
        for location in LAPD_AREA_NAMES:
            incident_rate = np.random.poisson(np.random.uniform(low=15, high=40))
            curr_location_value = np.random.uniform(low=150000, high=750000)
            new_location = Location(name=location, curr_incident_rate=incident_rate, curr_value=curr_location_value)
            locations.append(new_location)
        
        #set initial attention for new locations randomly
        total_remaining_attention = self.total_attention_units
        for location in locations:
            attention_allocated = np.random.uniform(low=0, high=total_remaining_attention) #RANDOM ALLOCATION
            location.set_attention(attention_allocated)
            # location.get_location_info()
            total_remaining_attention -= attention_allocated

        return locations

    '''
    Since we need to compute observations in both Env.reset() and Env.step(),
    it is convenient to have a helper method _get_obs that translates the environment
    internal state into the observation format. This keeps our code DRY 
    (Dont Repeat Yourself) and makes it easier to modify the observation format later. 
    '''
    def _get_obs(self):
        curr_map_observation = {}
        for location in self.locations:
            curr_map_observation[location.location_name] = {"incidents": location.curr_incidents,
                                                   "incident_rate": location.curr_incident_rate,
                                                   "curr_value": location.curr_value,
                                                   "curr_attention_units":location.curr_attention}
        return curr_map_observation

    '''
    We can also implement a similar method for auxiliary information returned by
    Env.reset() and Env.step(). this can be useful for debugging and understanding
    agent progress, but shouldnt be used by the learning algorithm itself.
    '''
    def _get_info(self):
        #Compute the Value and fairness metrics
        Value = 0
        Fairness = 0
        Discovered = 0
        Total_incidents = 0
        Highest_Discovery_Rate = 0
        Lowest_Discovery_Rate = 100
        for i in range(len(self.locations) - 1):
            location = self.locations[i]
            Discovered += min(location.curr_attention, location.curr_incidents)
            Total_incidents += location.curr_incidents
            Value += location.curr_value
            coverage_rate_i = min(location.curr_attention / location.curr_incidents,1)
            if coverage_rate_i > Highest_Discovery_Rate:
                Highest_Discovery_Rate = coverage_rate_i
            if coverage_rate_i < Lowest_Discovery_Rate:
                Lowest_Discovery_Rate = coverage_rate_i
        Fairness = Highest_Discovery_Rate - Lowest_Discovery_Rate    
        Value += self.locations[len(self.locations)-1].curr_value

        print(f"\
Current Fariness:     {Fairness}\n\
Current Value:        {Value}\n\
Current Discovered:   {Discovered}\n\
Total Incidents:      {Total_incidents}\n\
Highest Discovery %   {Highest_Discovery_Rate}\n\
Lowest Discovery %    {Lowest_Discovery_Rate}\n")
               
        return Fairness, Value, Discovered
    

    '''
    The reset() method starts a new episode. 
    It takes two optional parameters: seed for reproducible 
    random generation and options for additional configuration.
    On the first line, you must call super().reset(seed=seed) 
    to properly initialize the random number generator. 
    '''
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=43)
        observations = self._get_obs()
        info = self._get_info()
        return observations, info


    '''
    The step() method contains the core environment logic. 
    It takes an action, updates the environment state, and returns the results.
    This is where the world rules, and reward logic live 
    '''
    def step(self, actions):
        for i in range(len(actions)):
            curr_action = actions[i]
            curr_location = self.locations[i]
            curr_location.curr_attention = curr_action
            self.locations[i] = curr_location
        
        self.time_step += 1

        terminated = True if self.time_step == self.timesteps else False
        
        truncated = False

        observations = self._get_obs()
        info = self._get_info()

        Utility_weight = 1
        fairness_weight = self.total_attention_units / 2
        reward = (Utility_weight * info[2]) - (fairness_weight * info[0])
        print(f"Current Reward:     {reward}")
        return observations, reward, terminated, truncated

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

