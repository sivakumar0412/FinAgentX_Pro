import pandas as pd
import pickle
from backend.env import TradingEnv

data = pd.read_csv("backend/data/market_data.csv")
prices = data["close"].values

with open("backend/q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

env = TradingEnv(prices)
state = env.reset()
done = False

while not done:
    actions = [0, 1, 2]
    qs = [q_table.get((state, a), 0) for a in actions]
    action = actions[qs.index(max(qs))]
    state, reward, done = env.step(action)

print("Final Net Worth:", env.net_worth)
