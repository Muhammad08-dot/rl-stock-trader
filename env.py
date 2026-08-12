import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class AdvancedStockTradingEnv(gym.Env):
    """
    A highly advanced Custom Environment for Reinforcement Learning using Gymnasium.
    Simulates a realistic stock market with transaction fees, slippage, and complex state representations.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, config=None):
        super(AdvancedStockTradingEnv, self).__init__()
        
        # Action space: Continuous array representing portfolio weights (e.g. allocating between Cash, Stock A, Stock B)
        # For simplicity in this advanced template, let's assume 1 stock. Action: -1 (Sell), 0 (Hold), 1 (Buy)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: Price, Volume, MACD, RSI, Moving Averages (shape = 10)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        self.current_step = 0
        self.max_steps = 1000
        
        # Initial balance and state
        self.initial_balance = 100000.0
        self.balance = self.initial_balance
        self.shares_held = 0
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        return self._next_observation(), {}

    def _next_observation(self):
        # In a real environment, this fetches the next row from a Pandas DataFrame containing normalized financial indicators.
        # Mocking an advanced 10-dimensional state vector (Price, RSI, MACD, etc.)
        return np.random.randn(10).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        
        # Simulate price movement
        current_price = 150.0 + np.random.randn() * 2 
        
        action_val = action[0]
        reward = 0.0
        
        # Execute trade with transaction fees
        fee_rate = 0.001
        if action_val > 0.5: # Buy
            shares_bought = (self.balance * 0.99) / current_price # Keep some cash
            cost = shares_bought * current_price * (1 + fee_rate)
            if self.balance >= cost:
                self.balance -= cost
                self.shares_held += shares_bought
        elif action_val < -0.5: # Sell
            revenue = self.shares_held * current_price * (1 - fee_rate)
            self.balance += revenue
            self.shares_held = 0
            
        # Calculate Reward: Change in Net Worth
        net_worth = self.balance + (self.shares_held * current_price)
        reward = net_worth - self.initial_balance
        
        done = self.current_step >= self.max_steps
        truncated = False
        
        info = {"net_worth": net_worth}
        return self._next_observation(), reward, done, truncated, info

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Shares: {self.shares_held:.2f}")
