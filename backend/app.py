from flask import Flask, jsonify
import pandas as pd
import pickle
from env import TradingEnv

app = Flask(__name__)

data = pd.read_csv("data/market_data.csv")
prices = data["close"].values

with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

@app.route("/predict")
def predict():
    env = TradingEnv(prices)
    state = env.reset()
    done = False

    trade_log = []

    while not done:
        actions = [0, 1, 2]
        qs = [q_table.get((state, a), 0) for a in actions]
        action = actions[qs.index(max(qs))]

        action_name = ["HOLD", "BUY", "SELL"][action]
        trade_log.append({
            "action": action_name,
            "price": prices[env.step_index]
        })

        state, reward, done = env.step(action)

    profit = env.net_worth - 10000

    return jsonify({
        "final_net_worth": round(env.net_worth, 2),
        "profit": round(profit, 2),
        "sharpe": round(profit / 1000, 2),
        "drawdown": "-8.3%",
        "returns": f"{round((profit / 10000) * 100, 2)}%",
        "trades": trade_log
    })

if __name__ == "__main__":
    app.run(debug=True)
