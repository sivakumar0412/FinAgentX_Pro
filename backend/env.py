import numpy as np

class TradingEnv:
    def __init__(self, prices):
        self.prices = prices
        self.reset()

    def reset(self):
        self.step_index = 0
        self.balance = 10000
        self.shares = 0
        self.net_worth = self.balance
        return self._get_state()

    def _get_state(self):
        price = self.prices[self.step_index]
        return (int(price // 10), self.shares)

    def step(self, action):
        price = self.prices[self.step_index]

        # 0 = Hold, 1 = Buy, 2 = Sell
        if action == 1 and self.balance >= price:
            self.shares += 1
            self.balance -= price

        elif action == 2 and self.shares > 0:
            self.shares -= 1
            self.balance += price

        self.step_index += 1
        done = self.step_index == len(self.prices) - 1

        self.net_worth = self.balance + self.shares * price
        reward = self.net_worth - 10000

        return self._get_state(), reward, done
