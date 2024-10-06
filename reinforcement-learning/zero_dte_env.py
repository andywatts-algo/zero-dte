import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import spaces
import pandas as pd
import data as data

class ZeroDteEnv(gym.Env):
    def __init__(self, config=None):
        super(ZeroDteEnv, self).__init__()
        self.config = config
        self.date = "20240102"
        self.quotes = data.get_option_quotes(self.date)

        # Observation space TODO
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=(3,))  # Set the minimum and maximum values of the observation space to be more realistic.
        self.action_space = spaces.Discrete(5)  # Three discrete actions: Role down, hold, and roll up
        self.reset()

    def step(self, action):
        step_quotes = self.quotes[self.quotes['ms_of_day'] == self.ms]
        atm_strike = step_quotes.iloc[10]
        call_mid = atm_strike['call_mid']
        put_mid = atm_strike['put_mid']

# TODO single P/C position

# self.positions['call']

        # Buy ATM Call
        if action == 0:  
            if self.positions['call'] < 0:  # BTC
                # TODO compute PnL
                self.positions['call'] = 0 
                self.balance += call_mid
            else:   # BTO
                self.positions['call'] += call_mid
                self.balance -= call_mid


        elif action == 1:  # Buy ATM Put
            pass  # Do nothing
        elif action == 2:  # Sell ATM Call
            pass  # Do nothing
        elif action == 3:  # Sell ATM Put
            pass  # Do nothing
        elif action == 4:  # Hold
            pass  # Do nothing
        

        reward = 0.0
        self.ms += 1000
        done = self.ms > 58200000 or self.balance <= 0  # Note. 4:10PM
        return observation, reward, done, False, {}

    def get_observation(self):
        step_quotes = self.quotes[self.quotes['ms_of_day'] == self.ms]
        atm_strike = step_quotes.iloc[10]
        
        observation = step_quotes[['strike', 'call_mid', 'put_mid']].values.flatten()
        observation = np.append(observation, self.ms) # Add ms_of_day
        observation = np.append(observation, atm_strike['call_mid']) # Add atm call mid
        observation = np.append(observation, atm_strike['put_mid']) # Add atm put mid
        observation = np.append(observation, self.positions['put']) # Add put possitions
        observation = np.append(observation, self.positions['call']) # Add call possitions
        observation = np.append(observation, self.balance) # Add balance 
        return observation

    # Called after every episode;  Should return first observation
    def reset(self, seed=None, options=None, **kwargs):
        super().reset(seed=seed)
        self.balance = 1000.0
        self.ms = 34200000
        self.positions = {'call': 0, 'put': 0}  # Track call and put positions
        return self.get_observation(), {}

    def show(self):
        plt.figure(figsize=(10, 1))
        plt.plot(self.df.index, self.df['prices'], label='Sine wave')
        plt.show()






        
