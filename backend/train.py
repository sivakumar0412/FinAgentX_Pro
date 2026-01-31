import pandas as pd
import pickle

from env import TradingEnv
from q_agent import QLearningAgent

data = pd.read_csv("data/market_data.csv")
prices = data["close"].values

env = TradingEnv(prices)
agent = QLearningAgent(actions=[0, 1, 2])

episodes = 50

for ep in range(episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state

    print(f"Episode {ep+1}/{episodes} | Net Worth: {env.net_worth}")

with open("q_table.pkl", "wb") as f:
    pickle.dump(agent.q_table, f)

print("✅ Training completed")
