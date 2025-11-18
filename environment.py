'''
This Module defines the environment for the simulation

'''

from typing import Optional
import numpy as np
import gymnasium as gym
import pandas as pd



class Location():
    def __init__(self, curr_incident_rate, curr_incidents, curr_attention, curr_value):
        self.curr_incident_rate = curr_incident_rate
        self.curr_incidents = np.random.poisson


class AllocationEnv(gym.Env):

    '''
    Like all environments, our custom environment will inherit from gymnasium.Env
    that defines the structure all environments must follow. One of the requirements
    is defining the observation and action spaces, which declare what inputs (actions)
    and outputs (observations) are valid for this environment. 
    '''
    def __init__(self, incident_rate_by_areas, total_attention_units=400):
        
        self.total_attention_units = total_attention_units
        self.locations = incident_rate_by_areas
        self.attention

        
        pass


    def get_incident_rate_by_area(self):
        df = pd.read_csv("Crime_data_from_2020_to_Present_20251116.csv")
        slice = df[["AREA NAME", "AREA", "DATE OCC"]]
        incident_rate_by_areas = slice.groupby("AREA NAME").size()
        incident_rate_by_areas /= 2145
        return incident_rate_by_areas

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
