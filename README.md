# 🧠 RL Trading Agent vs. Technical Indicator Strategies

This project explores whether a reinforcement learning (RL) agent can outperform traditional rule-based trading strategies using minute-level stock data. We evaluate Deep Q-Networks (DQN + LSTM) and Proximal Policy Optimization (PPO) across three real-world assets: **Apple**, **S&P 500**, and **Reliance Industries**.

---

## 🚀 Project Highlights

- ✅ Custom Gym-style trading environment with Buy/Sell/Hold actions and transaction fees.
- ✅ Engineered 10+ technical indicators (EMA, MACD, RSI, BB, OBV, etc.).
- ✅ Developed and evaluated 9 rule-based trading strategies with stop-loss variants.
- ✅ Implemented DQN + LSTM and PPO agents from scratch.
- ✅ Evaluated each method fairly on identical test sets.
- ✅ Tested generalization across different stock datasets.

---

## 📈 Summary Results

| Asset      | RL Agent ROI | Buy & Hold ROI | Best Hardcoded ROI |
|------------|--------------|----------------|---------------------|
| Apple      | **587.88%**  | 578.28%        | ~1.2%               |
| S&P 500    | -4.73%       | -16.76%        | < 1%                |
| Reliance   | 459.23%      | 465.99%        | Mostly negligible   |

> 🔍 PPO failed to train effectively in this sparse-reward environment, while DQN + LSTM showed strong learning and generalization when given sufficient data.

---

## 📚 How to Run

Open the `RL.ipynb` notebook and follow the sections inside to:
- Prepare and normalize the datasets.
- Run rule-based trading strategies.
- Train the DQN + LSTM agent.
- Evaluate and compare performance.

---

## 💾 Dependencies

Install the following Python packages:

```bash
pip install numpy pandas torch gym scikit-learn numba
