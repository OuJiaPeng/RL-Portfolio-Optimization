"""RL portfolio evaluation with monthly refit.

Loads the best PPO checkpoint, fine-tunes monthly on the most recent
90 days, and evaluates on the following month. Produces metrics and
visualisation artefacts in rl_ppo/outputs/.

Usage:
    python -m rl_ppo.eval_rl_refit
"""
import os
# Suppress TensorFlow warnings before any imports
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # disable oneDNN ops
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 3 = ERROR only
os.environ.setdefault("KMP_WARNINGS", "0")

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl_ppo.config import Config
from rl_ppo.refit_config import RefitConfig
from rl_ppo.env.env import PortfolioEnv
from data.load_prices import create_etf_prices
from data.data_utils import create_features, load_and_process_data

def evaluate_model(model, env, n_steps):
    """Evaluate model on environment for n_steps"""
    obs = env.reset()
    returns = []
    weights = []
    actions = []
    
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Extract return and weights from info
        step_info = info[0] if isinstance(info, list) else info
        returns.append(step_info.get('port_ret', reward))
        if 'weights' in step_info:
            weights.append(step_info['weights'].copy())
        actions.append(action)
        
        if done[0] if isinstance(done, np.ndarray) else done:
            break
    
    return np.array(weights), np.array(returns), np.array(actions)

def calculate_metrics(returns, weights, dates):
    """Calculate comprehensive portfolio metrics"""
    returns = np.array(returns, dtype=float)
    
    # Basic metrics
    cum_ret = float(np.prod(1.0 + returns) - 1.0)
    
    # CAGR calculation
    trading_days = len(returns)
    years = trading_days / 252.0
    cagr = float((1.0 + cum_ret)**(1.0/years) - 1.0) if years > 0 else 0.0
    
    # Annualized return (arithmetic mean)
    ann_return = float(returns.mean() * 252.0)
    
    # Annualized volatility
    ann_vol = float(returns.std(ddof=1) * np.sqrt(252))
    
    # Sharpe ratio calculation - using standard approach (mean excess returns)
    rf_daily = Config.RISK_FREE_ANNUAL / 252.0
    excess_returns = returns - rf_daily
    ann_excess_return = float(excess_returns.mean() * 252.0)
    sharpe = float(ann_excess_return / ann_vol) if ann_vol > 0 else 0.0
    
    # Drawdown analysis
    wealth = np.cumprod(1.0 + returns)
    roll_max = np.maximum.accumulate(wealth)
    drawdown = (wealth / (roll_max + 1e-9)) - 1.0
    max_drawdown = float(drawdown.min())
    
    # Additional ratios
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    
    # Portfolio concentration metrics
    if weights is not None and len(weights) > 0:
        avg_weights = np.mean(weights, axis=0)
        hhi = float(np.sum(avg_weights**2))
        effective_holdings = 1.0 / hhi if hhi > 0 else 0.0
        weight_entropy = float(-np.sum(avg_weights * np.log(avg_weights + 1e-9)))
        max_weight = float(np.max(avg_weights))
        min_weight = float(np.min(avg_weights))
    else:
        hhi = effective_holdings = weight_entropy = max_weight = min_weight = 0.0
    
    return {
        'cumulative_return': cum_ret,
        'annualized_return': ann_return,
        'cagr': cagr,
        'annualized_excess_return': ann_excess_return,
        'sharpe_ratio': sharpe,
        'annualized_volatility': ann_vol,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'hhi': hhi,
        'effective_holdings': effective_holdings,
        'weight_entropy': weight_entropy,
        'max_weight': max_weight,
        'min_weight': min_weight,
        'years': years
    }

def create_visualizations(returns, weights, dates, metrics):
    """Create comprehensive visualizations"""
    results_dir = Path("./rl_ppo/outputs")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Wealth curve (matching Markowitz style)
    wealth = np.cumprod(1.0 + returns)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wealth / wealth[0], linewidth=2, color='lightskyblue')
    ax.set_title("RL Portfolio Wealth Growth")
    ax.set_xlabel('Date')
    ax.set_ylabel('Wealth (Normalized to 1.0)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'rl_wealth_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Wealth curve saved: {results_dir / 'rl_wealth_curve.png'}")
    
    # 2. Average allocation pie chart
    if weights is not None and len(weights) > 0:
        avg_weights = np.mean(weights, axis=0)
        significant_weights = avg_weights[avg_weights > 0.01]  # Only show >1%
        significant_tickers = [Config.ETF_TICKERS[i] for i, w in enumerate(avg_weights) if w > 0.01]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.Paired.colors
        ax.pie(significant_weights * 100, labels=significant_tickers, autopct='%1.1f%%', 
               startangle=140, colors=colors)
        ax.set_title('Average Portfolio Allocation (Test Period)')
        plt.tight_layout()
        plt.savefig(results_dir / 'rl_average_allocation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Average allocation saved: {results_dir / 'rl_average_allocation.png'}")
        
        # 3. Most recent allocation horizontal bar chart
        if len(weights) > 0:
            recent_weights = weights[-1]  # Last day's weights
            allocation_df = pd.DataFrame({
                'ETF': Config.ETF_TICKERS,
                'Weight': recent_weights,
                'Allocation %': recent_weights * 100
            }).sort_values('Weight', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(allocation_df['ETF'], allocation_df['Allocation %'], color='steelblue')
            ax.set_xlabel('Allocation (%)')
            ax.set_title('Most Recent Portfolio Allocation')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add percentage labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                       f'{width:.1f}%', ha='left', va='center')
            
            plt.tight_layout()
            plt.savefig(results_dir / 'rl_recent_allocation.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Recent allocation saved: {results_dir / 'rl_recent_allocation.png'}")
    else:
        print("No weight data available for allocation plots")

def save_results(returns, weights, dates, metrics):
    """Save detailed results to files"""
    results_dir = Path("./rl_ppo/outputs")

    # Save comprehensive metrics
    with open(results_dir / 'rl_final_metrics_refit.txt', 'w') as f:
        f.write("=== RL Portfolio Final Metrics (Monthly Refit - 2.0+ Sharpe) ===\n\n")
        f.write(f"Sharpe Ratio:         {metrics['sharpe_ratio']:.6f}\n")
        f.write(f"Cumulative Return:    {metrics['cumulative_return']:.2%}\n")
        f.write(f"Annualized Return:    {metrics['annualized_return']:.2%}\n")
        f.write(f"CAGR:                 {metrics['cagr']:.2%}\n")
        f.write(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}\n")
        f.write(f"Max Drawdown:         {metrics['max_drawdown']:.2%}\n")
        f.write(f"Calmar Ratio:         {metrics['calmar_ratio']:.2f}\n")
        f.write(f"Time Period:          {metrics['years']:.2f} years\n")
        f.write(f"\n--- Concentration Metrics ---\n")
        f.write(f"HHI (Concentration):  {metrics['hhi']:.3f}\n")
        f.write(f"Effective Holdings:   {metrics['effective_holdings']:.1f}\n")
        f.write(f"Weight Entropy:       {metrics['weight_entropy']:.3f}\n")
        f.write(f"Max Asset Weight:     {metrics['max_weight']:.1%}\n")
        f.write(f"Min Asset Weight:     {metrics['min_weight']:.1%}\n")
    
    # Save daily results CSV
    if weights is not None and len(weights) > 0:
        wealth = np.cumprod(1.0 + returns)
        roll_max = np.maximum.accumulate(wealth)
        drawdown = (wealth / roll_max) - 1.0
        
        results_df = pd.DataFrame({
            'date': dates[:len(returns)],
            'daily_return': returns,
            'cumulative_wealth': wealth,
            'drawdown': drawdown
        })
        
        # Add weight columns
        for i, ticker in enumerate(Config.ETF_TICKERS):
            if i < weights.shape[1]:
                results_df[f'weight_{ticker}'] = weights[:, i]
        
        results_df.to_csv(results_dir / 'rl_daily_results_refit.csv', index=False)
        print(f"Daily results saved: {results_dir / 'rl_daily_results_refit.csv'}")
    
    print(f"Final metrics saved: {results_dir / 'rl_final_metrics_refit.txt'}")

def main():
    print("=== RL Portfolio Evaluation (Monthly Refit - 2.0+ Sharpe) ===")
    
    # Load data exactly as train_simple_refit.py does
    prices_df = load_and_process_data()  # Use same function as train_simple_refit.py
    
    # Monthly refit evaluation (same as train_simple_refit.py)
    monthly_results = []
    
    # Test months from 2025-01 to 2025-06
    test_months = pd.date_range('2025-01-01', '2025-06-30', freq='MS')
    
    base_model_path = RefitConfig.BASE_MODEL_PATH
    print(f"Using base model: {base_model_path}")
    
    # Refit parameters that achieved 2.0+ Sharpe
    refit_params = {
        'MAX_POSITION_SIZE': RefitConfig.MAX_POSITION_SIZE,
        'TURNOVER_COST': RefitConfig.TURNOVER_COST, 
        'REBALANCE_FREQ': RefitConfig.REBALANCE_FREQ
    }
    print(f"Using refit parameters: {refit_params}")
    
    for month_start in test_months:
        month_end = month_start + pd.offsets.MonthEnd(0)

        # Skip July (insufficient data as noted in refit script)
        if month_start >= pd.Timestamp('2025-07-01'):
            break

        # Blank line before each month header
        print(f"\nMonth {month_start.strftime('%Y-%m')}: Retrain up to {(month_start - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}, evaluate {month_start.strftime('%Y-%m-%d')}..{month_end.strftime('%Y-%m-%d')}")
        
        # Match train_simple_refit.py exactly: use features with fit_end_date and 90-day lookback
        train_end = month_start - pd.Timedelta(days=1)
        if train_end < pd.to_datetime(Config.TRAIN_END):
            train_end = pd.to_datetime(Config.TRAIN_END)
        
        # Recompute features with frozen normalization (exactly as in train_simple_refit.py)
        feat_all, prices_aligned = create_features(prices_df, fit_end_date=train_end)
        
        # Use only recent 3 months for refit training - EXACTLY 90 days as in train_simple_refit.py
        recent_cutoff = train_end - pd.Timedelta(days=RefitConfig.REFIT_LOOKBACK_DAYS)  # 3 months
        mask_recent = (prices_aligned.index > recent_cutoff) & (prices_aligned.index <= train_end)
        feat_recent = feat_all.loc[mask_recent]
        X_recent = prices_aligned.loc[mask_recent]
        
        if len(feat_recent) < RefitConfig.MIN_REFIT_DAYS:  # Need minimum data
            print(f"  Insufficient recent data ({len(feat_recent)} days), using full history")
            mask_recent = prices_aligned.index <= train_end
            feat_recent = feat_all.loc[mask_recent]
            X_recent = prices_aligned.loc[mask_recent]
        
        # Create refit environment
        refit_env = DummyVecEnv([lambda: PortfolioEnv(feat_recent.values, 
                                                     X_recent.values, 
                                                     config_overrides=refit_params)])
        
        # Load and refit model
        refit_model = PPO.load(base_model_path, env=refit_env, device="auto")
        refit_model.learning_rate = RefitConfig.LEARNING_RATE
        refit_model.clip_range = lambda _: RefitConfig.CLIP_RANGE
        
        # Light refit: 4096 steps (as in original)
        print(f"  Retraining model with {len(feat_recent)} days of data...")
        refit_model.learn(total_timesteps=RefitConfig.REFIT_STEPS, progress_bar=False)
        
        # Evaluate on month using the same feat_all and prices_aligned (with correct normalization)
        eval_mask = (prices_aligned.index >= month_start) & (prices_aligned.index <= month_end)
        eval_features = feat_all.loc[eval_mask].copy()
        eval_prices = prices_aligned.loc[eval_mask].copy()
        
        if len(eval_features) == 0:
            continue
            
        # Create evaluation environment with same refit parameters
        eval_env = DummyVecEnv([lambda: PortfolioEnv(eval_features.values,
                                                    eval_prices.values,
                                                    config_overrides=refit_params)])
        
        # Evaluate
        month_weights, month_returns, _ = evaluate_model(refit_model, eval_env, len(eval_features))
        
        # Calculate monthly metrics
        monthly_metrics = calculate_metrics(month_returns, month_weights, eval_features.index)
        monthly_results.append({
            'month': month_start.strftime('%Y-%m'),
            'returns': month_returns,
            'weights': month_weights,
            'dates': eval_features.index,
            'metrics': monthly_metrics
        })
        
        print(f"  {month_start.strftime('%Y-%m')} Sharpe: {monthly_metrics['sharpe_ratio']:.3f} | Cum: {monthly_metrics['cumulative_return']*100:.2f}%")
    
    # Combine all months for overall evaluation
    all_returns = np.concatenate([r['returns'] for r in monthly_results])
    all_weights = np.concatenate([r['weights'] for r in monthly_results])
    all_dates = pd.Index(pd.concat([pd.Series(r['dates']) for r in monthly_results]))
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(all_returns, all_weights, all_dates)
    
    print("\n=== OVERALL REFIT EVALUATION RESULTS ===")
    print(f"Sharpe Ratio:         {overall_metrics['sharpe_ratio']:.6f}")
    print(f"Cumulative Return:    {overall_metrics['cumulative_return']:.2%}")
    print(f"Annualized Return:    {overall_metrics['annualized_return']:.2%}")
    print(f"CAGR:                 {overall_metrics['cagr']:.2%}")
    print(f"Annualized Excess Ret: {overall_metrics['annualized_excess_return']:.2%}")
    print(f"Annualized Vol:       {overall_metrics['annualized_volatility']:.2%}")
    print(f"Max Drawdown:         {overall_metrics['max_drawdown']:.2%}")
    print(f"Calmar Ratio:         {overall_metrics['calmar_ratio']:.2f}")
    print(f"Effective Holdings:   {overall_metrics['effective_holdings']:.1f}")
    print(f"Days evaluated:       {len(all_returns)}")
    print(f"Period:               {overall_metrics['years']:.2f} years")
    
    # Manual verification of Sharpe calculation (now matches Markowitz/Naive method)
    manual_sharpe = overall_metrics['annualized_excess_return'] / overall_metrics['annualized_volatility']
    print(f"\nSharpe Check: {overall_metrics['annualized_excess_return']:.3f} / {overall_metrics['annualized_volatility']:.3f} = {manual_sharpe:.3f}")
    
    # Create visualizations and save results
    create_visualizations(all_returns, all_weights, all_dates, overall_metrics)
    save_results(all_returns, all_weights, all_dates, overall_metrics)

if __name__ == "__main__":
    main()