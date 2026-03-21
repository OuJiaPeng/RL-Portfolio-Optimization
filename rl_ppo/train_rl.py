"""PPO training script for portfolio optimization.

Usage:
    python -m rl_ppo.train_rl          (from project root)
    python rl_ppo/train_rl.py          (from project root)
"""
import os
import sys
import warnings
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("KMP_WARNINGS", "0")

warnings.filterwarnings(
    "ignore",
    message="You provided an OpenAI Gym environment",
    category=UserWarning,
    module="stable_baselines3.common.vec_env.patch_gym",
)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_configure_logger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl_ppo.config import Config
from rl_ppo.env.env import PortfolioEnv
from data.data_utils import load_and_process_data, create_features, train_val_test_split


def _sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    excess = returns - getattr(Config, 'RISK_FREE_DAILY', 0.0)
    std = excess.std(ddof=1)
    if std <= 1e-9:
        return 0.0
    return float(excess.mean() / std * np.sqrt(252))


class MultiWindowEvalCallback(BaseCallback):
    def __init__(self, windows: list[tuple[np.ndarray, np.ndarray]], eval_every: int, patience: int, verbose: int = 1):
        super().__init__(verbose)
        self.windows = windows
        self.eval_every = eval_every
        self.patience = patience
        self.best_mean = -1e9
        self.best_step = 0
        self.best_path = Path('./rl_ppo/outputs/best_model.zip')

    def _eval_window(self, feat: np.ndarray, prices: np.ndarray) -> dict:
        env = PortfolioEnv(feat, prices)
        obs, _ = env.reset()
        terminated = truncated = False
        rets = []
        while not (terminated or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            rets.append(info.get('port_ret', r))
        rets = np.array(rets, dtype=float)
        return {
            'sharpe': _sharpe(rets),
            'cum': float(np.prod(1.0 + rets) - 1.0) if rets.size else 0.0,
            'len': int(rets.size)
        }

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step % self.eval_every != 0:
            return True
        metrics = [self._eval_window(f, p) for (f, p) in self.windows]
        sharpes = [m['sharpe'] for m in metrics]
        mean_s = float(np.mean(sharpes)) if sharpes else 0.0
        median_s = float(np.median(sharpes)) if sharpes else 0.0
        disp = float(np.std(sharpes, ddof=0)) if len(sharpes) > 1 else 0.0
        # Soft worst-window penalty (guard)
        worst = min(sharpes) if sharpes else 0.0
        if getattr(Config, 'ENABLE_SOFT_GUARD', False):
            thresh = float(getattr(Config, 'WORST_GUARD_THRESHOLD', -1.0))
            lam = float(getattr(Config, 'SOFT_GUARD_LAMBDA', 0.5))
            shortfall = max(0.0, thresh - worst)
            adj_mean = mean_s - lam * shortfall
        else:
            adj_mean = mean_s
        self.logger.record('val/mean_sharpe', mean_s)
        self.logger.record('val/median_sharpe', median_s)
        self.logger.record('val/dispersion', disp)
        self.logger.record('val/worst_sharpe', worst)
        self.logger.record('val/adjusted_mean', adj_mean)
        for i, m in enumerate(metrics):
            self.logger.record(f'val/window_{i}_sharpe', m['sharpe'])
            self.logger.record(f'val/window_{i}_len', m['len'])
        improve = adj_mean > self.best_mean + 1e-6
        if improve:
            self.best_mean = adj_mean
            self.best_step = step
            os.makedirs('./rl_ppo/outputs', exist_ok=True)
            self.model.save(self.best_path.as_posix())
        elif step - self.best_step > self.patience * self.eval_every:
            if self.verbose:
                print(f"Early stop triggered at step {step}. Best adjusted mean {self.best_mean:.3f}")
            return False
        return True


class EntropyAnnealCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.initial = float(getattr(Config, 'ENT_COEF_INITIAL', getattr(Config, 'ENT_COEF', 0.0)))
        self.final = float(getattr(Config, 'ENT_COEF_FINAL', self.initial))
        self.steps = int(getattr(Config, 'ENT_COEF_ANNEAL_STEPS', 0) or 0)

    def _on_step(self) -> bool:
        if self.steps <= 0:
            return True
        frac = min(1.0, self.num_timesteps / self.steps)
        cur = self.initial + (self.final - self.initial) * frac
        self.model.ent_coef = cur
        self.logger.record('anneal/ent_coef', cur)
        return True


class KLAnealCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.init_kl = float(getattr(Config, 'TARGET_KL_INITIAL', 0.1))
        self.final_kl = float(getattr(Config, 'TARGET_KL_FINAL', self.init_kl))
        self.steps = int(getattr(Config, 'TARGET_KL_ANNEAL_STEPS', 0) or 0)
        self.delay = int(getattr(Config, 'TARGET_KL_DELAY_STEPS', 0) or 0)

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t < self.delay:
            self.model.target_kl = None
            return True
        if self.steps <= 0:
            self.model.target_kl = self.final_kl
            return True
        frac = min(1.0, (t - self.delay) / self.steps)
        cur = self.init_kl + (self.final_kl - self.init_kl) * frac
        self.model.target_kl = cur
        self.logger.record('anneal/target_kl', cur)
        return True


class LogitClipAnnealCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.init = float(getattr(Config, 'INITIAL_LOGIT_CLIP', getattr(Config, 'LOGIT_CLIP', 0.0)))
        self.final = float(getattr(Config, 'FINAL_LOGIT_CLIP', self.init))
        self.steps = int(getattr(Config, 'LOGIT_CLIP_ANNEAL_STEPS', 0) or 0)

    def _on_step(self) -> bool:
        if self.steps <= 0:
            Config.LOGIT_CLIP = self.final  # type: ignore[attr-defined]
            return True
        frac = min(1.0, self.num_timesteps / self.steps)
        cur = self.init + (self.final - self.init) * frac
        Config.LOGIT_CLIP = cur  # dynamic attribute mutation
        self.logger.record('anneal/logit_clip', cur)
        return True


def main():
    # Seeds
    random.seed(Config.SEED); np.random.seed(Config.SEED); torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    prices = load_and_process_data()
    feat_all, prices_aligned = create_features(prices, fit_end_date=Config.TRAIN_END)
    train_df, val_df, test_df = train_val_test_split(prices_aligned)
    feat_train = feat_all.loc[train_df.index]
    feat_val = feat_all.loc[val_df.index]

    train_env = VecMonitor(DummyVecEnv([lambda: PortfolioEnv(feat_train.values, train_df.values)]))

    # Validation windows split
    n_win = max(1, int(getattr(Config, 'VAL_N_WINDOWS', 3) or 1))
    val_prices_np = val_df.values; val_feat_np = feat_val.values
    total_len = val_prices_np.shape[0]
    while n_win > 1 and total_len / n_win < 5:  # ensure at least ~5 days per window
        n_win -= 1
    window_slices: list[tuple[np.ndarray, np.ndarray]] = []
    if n_win == 1:
        window_slices.append((val_feat_np, val_prices_np))
    else:
        idxs = np.linspace(0, total_len, n_win + 1, dtype=int)
        for i in range(n_win):
            s, e = idxs[i], idxs[i+1]
            if e - s < 3:  # skip too small windows
                continue
            window_slices.append((val_feat_np[s:e], val_prices_np[s:e]))
    if not window_slices:
        window_slices.append((val_feat_np, val_prices_np))

    act_map = {'tanh': nn.Tanh, 'relu': nn.ReLU}
    activation_fn = act_map.get(getattr(Config, 'ACTIVATION_FN', 'tanh').lower(), nn.Tanh)
    policy_kwargs = {
        'net_arch': getattr(Config, 'POLICY_NET_ARCH', {'pi': [256,256], 'vf': [256,256]}),
        'activation_fn': activation_fn,
        'ortho_init': getattr(Config, 'ORTHO_INIT', True),
    }

    lr = Config.LEARNING_RATE
    clip = Config.CLIP_RANGE

    model = PPO(
        'MlpPolicy',
        train_env,
        n_steps=Config.N_STEPS,
        batch_size=Config.BATCH_SIZE,
        n_epochs=Config.N_EPOCHS,
        gamma=Config.GAMMA,
        gae_lambda=Config.GAE_LAMBDA,
        ent_coef=float(getattr(Config, 'ENT_COEF_INITIAL', getattr(Config, 'ENT_COEF', 0.0))),
        vf_coef=Config.VF_COEF,
        max_grad_norm=Config.MAX_GRAD_NORM,
        learning_rate=lr,
        clip_range=clip,
        target_kl=None,  # start without KL until annealed
        use_sde=Config.USE_SDE,
        sde_sample_freq=Config.SDE_SAMPLE_FREQ,
        seed=Config.SEED,
        device=getattr(Config, 'DEVICE', 'auto'),
        verbose=1,
        policy_kwargs=policy_kwargs,
    )

    os.makedirs('./rl_ppo/outputs', exist_ok=True)
    model.set_logger(sb3_configure_logger('./rl_ppo/outputs', ['csv']))

    callbacks = [
        MultiWindowEvalCallback(window_slices, eval_every=int(getattr(Config, 'VAL_EVAL_FREQ', 10_000) or 10_000), patience=int(getattr(Config, 'VAL_EARLY_STOP_PATIENCE', 15) or 15)),
        EntropyAnnealCallback(),
        KLAnealCallback(),
        LogitClipAnnealCallback(),
    ]

    model.learn(total_timesteps=Config.TOTAL_TIMESTEPS, callback=callbacks)
    model.save('./rl_ppo/outputs/final_model')

    # Evaluate on test split
    feat_test = feat_all.loc[test_df.index].values
    test_env = PortfolioEnv(feat_test, test_df.values)
    obs, _ = test_env.reset(); terminated = truncated = False; rets = []
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = test_env.step(action)
        rets.append(info.get('port_ret', r))
    rets = np.array(rets, dtype=float)
    test_sharpe = _sharpe(rets)
    test_cum = float(np.prod(1.0 + rets) - 1.0)
    print(f"Test Sharpe (pre-refit): {test_sharpe:.3f} | Cumulative Return: {test_cum:.2%}")
    with open('./rl_ppo/outputs/test_metrics.txt', 'w', encoding='utf-8') as f:
        f.write(f"test_sharpe,{test_sharpe}\n")
        f.write(f"test_cum_return,{test_cum}\n")


if __name__ == '__main__':
    main()