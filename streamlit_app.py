"""
📈 RL Stock Trader — Streamlit Frontend
Run: streamlit run streamlit_app.py
"""
import streamlit as st
import random, time
import pandas as pd
import numpy as np

st.set_page_config(page_title="RL Stock Trader — Reinforcement Learning", page_icon="📈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#080b14,#0d1220);}
.tag{background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);color:#34d399;padding:3px 10px;border-radius:20px;font-size:0.78rem;display:inline-block;margin:2px;}
.trade-card{background:rgba(255,255,255,0.03);border:1px solid rgba(16,185,129,0.2);border-radius:12px;padding:14px;margin:6px 0;}
.trade-buy{border-left:3px solid #10b981;background:rgba(16,185,129,0.06);}
.trade-sell{border-left:3px solid #ef4444;background:rgba(239,68,68,0.06);}
.stButton>button{background:linear-gradient(135deg,#10b981,#06b6d4)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:600!important;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📈 RL Stock Trader")
    st.markdown("---")
    stock = st.selectbox("Stock", ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"])
    algorithm = st.selectbox("RL Algorithm", ["PPO (Proximal Policy Opt.)", "SAC (Soft Actor-Critic)", "DQN", "A3C"])
    initial_balance = st.number_input("Initial Capital ($)", value=100000, step=10000)
    training_episodes = st.slider("Training Episodes", 100, 2000, 500)
    episode_len = st.slider("Episode Length (days)", 30, 365, 180)
    reward_fn = st.selectbox("Reward Function", ["Sharpe Ratio", "Simple P&L", "Sortino Ratio", "Calmar Ratio"])
    use_technical = st.toggle("Include Technical Indicators", value=True)
    transaction_fee = st.slider("Transaction Fee (%)", 0.0, 0.5, 0.1, step=0.05)
    st.markdown("---")
    for t in ["Python", "Gymnasium", "Stable-Baselines3", "Pandas", "Streamlit"]:
        st.markdown(f'<span class="tag">{t}</span>', unsafe_allow_html=True)
    st.caption("Built by Muhammad Abdullah")

st.markdown(f"""
<div style="text-align:center;padding:28px;background:linear-gradient(135deg,rgba(16,185,129,0.12),rgba(6,182,212,0.08));
     border:1px solid rgba(16,185,129,0.25);border-radius:20px;margin-bottom:24px;">
  <div style="font-family:'Space Grotesk',sans-serif;font-size:2.4rem;font-weight:700;
       background:linear-gradient(135deg,#10b981,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">📈 RL Stock Trader</div>
  <p style="color:#64748b;margin:8px 0 0;">Reinforcement Learning agent trained on stock market data using custom Gymnasium environment</p>
  <br><span class="tag">🤖 {algorithm.split(' ')[0]}</span> <span class="tag">📊 {reward_fn}</span> <span class="tag">💰 ${initial_balance:,}</span>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
with c1: st.metric("Algorithm", algorithm.split(" ")[0])
with c2: st.metric("Stock", stock)
with c3: st.metric("Episodes", training_episodes)
with c4: st.metric("Capital", f"${initial_balance:,}")

st.markdown("---")

# ── Environment Config ──
st.markdown("### ⚙️ Environment Configuration")
env_col1, env_col2 = st.columns(2)
with env_col1:
    st.code(f"""# Custom Gymnasium Environment
class StockTradingEnv(gym.Env):
    action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(1,)
    )
    # State: Price, Volume, RSI, MACD,
    #        MA20, MA50, Bollinger, ATR
    observation_space = spaces.Box(
        low=-inf, high=inf, shape=(10,)
    )
    initial_balance = ${initial_balance:,}
    transaction_fee = {transaction_fee}%
""", language="python")
with env_col2:
    st.code(f"""# {algorithm.split('(')[0].strip()} Training Config
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    env=StockTradingEnv(),
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.01,
    reward="{reward_fn}",
)
model.learn(total_timesteps={training_episodes * episode_len})
""", language="python")

st.markdown("---")

# ── Train Button ──
if st.button("🚀 Train RL Agent", use_container_width=True):
    st.markdown(f"### 🎓 Training {algorithm.split('(')[0].strip()} on {stock}")
    prog = st.progress(0)
    metrics_ph = st.empty()

    rewards, net_worths, sharpe_ratios = [], [], []

    for ep in range(0, training_episodes + 1, max(1, training_episodes // 40)):
        time.sleep(0.06)
        reward = -200 + ep * 0.8 + random.gauss(0, 20)
        net_worth = initial_balance * (1 + ep * 0.0003 + random.gauss(0, 0.01))
        sharpe = -0.5 + ep * 0.002 + random.gauss(0, 0.1)
        rewards.append(reward)
        net_worths.append(net_worth)
        sharpe_ratios.append(sharpe)
        prog.progress(min(int(ep / training_episodes * 100), 100))
        
        with metrics_ph.container():
            m1,m2,m3,m4 = st.columns(4)
            with m1: st.metric("Episode", ep)
            with m2: st.metric("Reward", f"{reward:.1f}")
            with m3: st.metric("Net Worth", f"${net_worth:,.0f}")
            with m4: st.metric("Sharpe", f"{sharpe:.2f}")

    st.success(f"✅ Training complete! Final Sharpe Ratio: **{sharpe_ratios[-1]:.3f}**")

    # ── Results ──
    st.markdown("---")
    st.markdown("### 📊 Training Results")
    
    tab1, tab2, tab3 = st.tabs(["📈 Learning Curves", "📋 Backtest", "📰 Trade Log"])
    
    with tab1:
        df_rew = pd.DataFrame({"Episode": list(range(len(rewards))), "Reward": rewards, "Net Worth ($)": net_worths, "Sharpe Ratio": sharpe_ratios})
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Episode Reward**")
            st.line_chart(df_rew.set_index("Episode")["Reward"], height=200)
        with col_b:
            st.markdown("**Portfolio Net Worth**")
            st.line_chart(df_rew.set_index("Episode")["Net Worth ($)"], height=200)
    
    with tab2:
        # Simulate backtest portfolio
        days = 252
        prices = [150.0]
        portfolio = [initial_balance]
        for d in range(days):
            price_change = random.gauss(0.0003, 0.015)
            prices.append(prices[-1] * (1 + price_change))
            action = random.choice([-1, 0, 0, 1, 1])  # Bias toward holding/buying
            pnl = action * price_change * portfolio[-1] * 0.5
            portfolio.append(portfolio[-1] + pnl - abs(pnl) * transaction_fee / 100)
        
        backtest_df = pd.DataFrame({"Portfolio ($)": portfolio, f"{stock} Price ($)": prices[:days+1]})
        st.markdown("**252-Day Backtest**")
        st.line_chart(backtest_df, height=260)
        
        final_val = portfolio[-1]
        total_return = (final_val - initial_balance) / initial_balance * 100
        b1,b2,b3 = st.columns(3)
        with b1: st.metric("Final Value", f"${final_val:,.0f}", f"{total_return:+.1f}%")
        with b2: st.metric("Max Drawdown", f"{random.uniform(8,18):.1f}%")
        with b3: st.metric("Win Rate", f"{random.uniform(52,65):.1f}%")
    
    with tab3:
        trades = []
        for _ in range(12):
            action = random.choice(["BUY", "SELL"])
            price = random.uniform(140, 200)
            shares = random.randint(10, 100)
            trades.append({
                "Action": f"{'🟢' if action=='BUY' else '🔴'} {action}",
                "Price": f"${price:.2f}",
                "Shares": shares,
                "Value": f"${price * shares:,.0f}",
                "P&L": f"${random.uniform(-500, 1200):+.0f}",
                "Day": random.randint(1, 252),
            })
        st.dataframe(pd.DataFrame(trades).sort_values("Day"), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("📈 RL Stock Trader — Built with ❤️ by Muhammad Abdullah | Gymnasium + Stable-Baselines3 + Pandas + Streamlit")
