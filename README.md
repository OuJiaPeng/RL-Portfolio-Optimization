# DRL Portfolio Optimization

> PPO-based portfolio allocator for 10 ETFs with a 274-dimensional state space, achieving **Sharpe 2.00** vs 1.75 rolling Markowitz on 2025 H1 out-of-sample data.

Allocation decisions from this module determine the positions that an execution layer
then trades efficiently — see [RL-Optimal-Liquidation](https://github.com/OuJiaPeng/RL-Optimal-Liquidation)
for the downstream execution agent.

---

## Results

| Portfolio | Sharpe | Ann. Excess Return | Volatility | Max Drawdown | CAGR |
|-----------|-------:|-------------------:|-----------:|-------------:|-----:|
| **RL (PPO)** | **2.00** | 24.63 % | 12.31 % | −6.35 % | 32.13 % |
| Markowitz | 1.75 | 31.68 % | 18.06 % | −5.80 % | 40.56 % |
| Naive (Equal Wt) | 0.82 | 12.67 % | 15.44 % | −12.05 % | 16.48 % |
| SPY (Buy & Hold) | 0.06 | 1.49 % | 26.48 % | −19.00 % | 0.78 % |

![Overlayed wealth curves on 2025 H1 OOS](analysis/visuals/overlay_wealth_curves_with_spy.png)

**Key take-aways**

1. **Risk efficiency** — RL has 22 % lower excess return but 32 % lower volatility than Markowitz, lifting the Sharpe ratio from 1.75 → 2.00.
2. **Adaptive concentration** — the agent concentrates when signal quality is high, otherwise diversifies; unlike equal-weight or Markowitz, which can overfit means / covariances.
3. **Monte Carlo tail positioning** (1 M sims) — Markowitz Sharpe 1.75 ≈ top 0.3 %; RL Sharpe 2.00 ≈ top 0.01 % of simulated random-allocation paths.

---

## Methodology

### Agent

| Component | Detail |
|-----------|--------|
| Algorithm | PPO (Stable-Baselines3) with state-dependent exploration (`gSDE`) |
| State | 274-dim: 10-day stacked normalised log returns, multi-horizon returns, RSI-14, realised vol, downside semi-vol, cross-sectional ranks, mean correlation, cyclical time encodings, previous portfolio weights |
| Action | Continuous logits $\in \mathbb{R}^{10}$, softmax-normalised to portfolio weights |
| Reward | Excess return − turnover cost + movement bonus − variance penalty ± HHI band shaping + advantage tilt, with rolling std normalisation |
| Refit | Monthly fine-tune on most recent 90-day rolling window during test period |

### Baselines

| Baseline | Description |
|----------|-------------|
| **Rolling Markowitz** | Mean-variance optimisation with 6-month rolling window, daily rebalanced |
| **Naive Equal Weight** | Static 10 % per ETF, no rebalancing |

---

## Data & Universe

- **Universe** (10 ETFs): `SPY QQQ IWM EFA EEM VNQ TLT IEF GLD USO`
- **Train**: 2019-01-01 → 2024-05-31
- **Validation**: 2024-06-01 → 2024-12-31
- **Test (OOS)**: 2025-01-02 → 2025-07-01
- **Source**: Polygon.io daily OHLCV + technical indicators (RSI, MACD, EMA, Bollinger Bands)

---

## Repository Structure

```
├── data/                   # Data loading & feature engineering
│   ├── data_utils.py       #   Feature pipeline (274-dim state construction)
│   ├── load_full_data.py   #   Polygon.io API fetcher
│   └── load_prices.py      #   Lightweight price loader
├── rl_ppo/                 # PPO agent
│   ├── config.py           #   All hyperparameters
│   ├── refit_config.py     #   Monthly-refit overrides
│   ├── train_rl.py         #   Training entry point
│   ├── eval_rl_refit.py    #   Evaluation with monthly refit
│   ├── env/
│   │   ├── env.py          #     Gymnasium environment
│   │   └── policy.py       #     Custom MLP policy
│   └── outputs/            #   Models, metrics & result CSVs
├── markowitz/              # Rolling Markowitz baseline (notebook)
│   └── outputs/            #   Metrics, weights & wealth CSVs
├── naive/                  # Equal-weight baseline (notebook)
│   └── outputs/            #   Metrics & plots
├── analysis/               # Monte Carlo simulation & visualisations
│   ├── monte_carlo/
│   ├── visuals/
│   └── feature_analysis/
├── Makefile
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/OuJiaPeng/RL-Portfolio-Optimization.git
cd RL-Portfolio-Optimization
pip install -r requirements.txt

# 2. Train the PPO agent
make train

# 3. Evaluate with monthly refit on the test period
make eval
```

### Configuration

All hyperparameters live in [`rl_ppo/config.py`](rl_ppo/config.py).  
Refit-specific overrides (learning rate, position limits, turnover cost) are in [`rl_ppo/refit_config.py`](rl_ppo/refit_config.py).

---

## License

MIT

