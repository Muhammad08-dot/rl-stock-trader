<div align="center">
  <h1>📈 RL Stock Trader</h1>
  <p><strong>Reinforcement Learning agent trained to autonomously trade stocks.</strong></p>
</div>

## 🚀 Overview
The RL Stock Trader utilizes deep reinforcement learning to navigate the stock market. Built on top of a custom Gymnasium environment, the agent learns to maximize the Sharpe Ratio and portfolio value by taking Buy/Sell/Hold actions based on historical price action and technical indicators.

## ✨ Features
- **Custom Trading Environment:** Implements a realistic OpenAI `gym` (Gymnasium) environment factoring in transaction fees and slippage.
- **Advanced RL Algorithms:** Supports Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC).
- **Technical Indicators:** State space includes RSI, MACD, Bollinger Bands, and Moving Averages.
- **Backtesting Dashboard:** Visualize learning curves, portfolio net worth, and a detailed trade log.

## 🛠️ Tech Stack
- **RL Framework:** [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- **Environment:** [Gymnasium](https://gymnasium.farama.org/)
- **Data Analysis:** Pandas, NumPy, TA-Lib
- **Frontend UI:** [Streamlit](https://streamlit.io/)

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Muhammad08-dot/rl-stock-trader.git
   cd rl-stock-trader
   ```

2. **Install dependencies:**
   ```bash
   pip install stable-baselines3 gymnasium pandas numpy streamlit
   ```

3. **Run the application:**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📄 License
This project is licensed under the MIT License.
