'''
This Module defines the environment for the simulation

'''

from typing import Optional
import numpy as np
import gymnasium as gym
import pandas as pd

LAPD_AREA_NAMES = ["Foothill", "Hollenbeck", "Mission", "Topanga", "Harbor", "Devonshire",
                    "West Valley", "Van Nuys", "Northeast", "West LA", "Rampart", "Wilshire",
                      "Newton", "Southeast", "Olympic", "North Hollywood",
                    "West Hollywood", "Southwest", "Pacific", "7th Street", "Central"]

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
    def __init__(self, incident_rate_by_areas, total_attention_units=400):
        
        self.total_attention_units = total_attention_units
        self.locations = self.initialize_locations_random()


    def initialize_locations_random(self):
        locations = []
        for location in LAPD_AREA_NAMES:
            incident_rate = np.random.uniform(low=10, high=40)
            curr_location_value = np.random.uniform(low=150000, high=750000)
            new_location = Location(name=location, curr_incident_rate=incident_rate, curr_value=curr_location_value)
            new_location.get_location_info()
            locations.append(new_location)
        
        #set initial attention for new locations randomly
        total_remaining_attention = self.total_attention_units
        for location in locations:
            attention_allocated = np.random.uniform(low=0, high=total_remaining_attention) #RANDOM ALLOCATION
            location.set_attention(attention_allocated)
            location.get_location_info()
            total_remaining_attention -= attention_allocated

        return locations

    '''
    Since we need to compute observations in both Env.reset() and Env.step(),
    it is convenient to have a helper method _get_obs that translates the environment
    internal state into the observation format. This keeps our code DRY 
    (Dont Repeat Yourself) and makes it easier to modify the observation format later. 
    '''
    def _get_obs(self):
        pass

    '''
    We can also implement a similar method for auxiliary information returned by
    Env.reset() and Env.step(). this can be useful for debugging and understanding
    agent progress, but shouldnt be used by the learning algorithm itself.
    '''
    def _get_info(self):
        pass
    

    '''
    The reset() method starts a new episode. 
    It takes two optional parameters: seed for reproducible 
    random generation and options for additional configuration.
    On the first line, you must call super().reset(seed=seed) 
    to properly initialize the random number generator. 
    '''
    def reset(self):
        pass


    '''
    The step() method contains the core environment logic. 
    It takes an action, updates the environment state, and returns the results.
    This is where the world rules, and reward logic live 
    '''
    def step(self):
        pass


def main():
    TestHarness()

def TestHarness():
    # test1()
    # test2()
    create_env()


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
    env = AllocationEnv(incident_rate_by_areas=10, total_attention_units=400)


if __name__ == "__main__":
    main()

