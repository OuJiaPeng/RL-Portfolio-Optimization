"""Gymnasium environment for PPO-based portfolio allocation."""
from typing import Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rl_ppo.config import Config


class PortfolioEnv(gym.Env):
    """Portfolio allocation environment.

    State:  [features_t, previous_weights]  (274-dim for 10 ETFs).
    Action: continuous logits in R^n_assets, softmax-normalized to weights.

    Reward (applied in order):
      base  = portfolio excess return - turnover cost
      + movement bonus        (encourages rebalancing)
      - variance penalty       (penalises portfolio variance)
      +/- HHI band shaping    (two-sided anti-uniform & anti-concentration)
      + advantage tilt         (relative-return bonus)
      * reward scale
      => rolling std normalization
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, features: np.ndarray, prices: np.ndarray, config_overrides: dict = None):
        super().__init__()
        assert features.ndim == 2 and prices.ndim == 2
        assert features.shape[0] == prices.shape[0]
        self.features = features.astype(np.float32)
        self.config_overrides = config_overrides or {}
        self.prices = prices.astype(np.float32)
        self.n_assets = self.prices.shape[1]
        assert self.n_assets == len(Config.ETF_TICKERS)

        self.action_space = spaces.Box(low=-10, high=10, shape=(self.n_assets,), dtype=np.float32)
        self.obs_dim = self.features.shape[1] + self.n_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(Config.SEED)
        self.t = 0
        self.w_prev = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._rew_hist: list[float] = []
        self._asset_bias: np.ndarray | None = None

    def _get_config_value(self, attr_name: str, default=None):
        """Get config value with override support for refit parameters."""
        if attr_name in self.config_overrides:
            return self.config_overrides[attr_name]
        return getattr(Config, attr_name, default)

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)

        # Logit clipping
        logit_clip = float(getattr(Config, 'LOGIT_CLIP', 0.0) or 0.0)
        if logit_clip > 0.0:
            a = np.clip(a, -logit_clip, logit_clip)

        # Per-episode asset bias (symmetry breaker)
        if self._asset_bias is not None:
            a = a + self._asset_bias

        # Softmax to portfolio weights
        temp = Config.ACTION_TEMPERATURE
        z = (a / temp) - (a / temp).max()
        w = np.exp(z)
        w = w / (w.sum() + 1e-9)

        # Position size constraints
        max_w = float(self._get_config_value('MAX_POSITION_SIZE', 1.0) or 1.0)
        min_w = float(self._get_config_value('MIN_POSITION_SIZE', 0.0) or 0.0)
        if max_w < 1.0 or min_w > 0.0:
            w = np.clip(w, min_w, max_w)
            s = w.sum()
            w = (w / s) if s > 1e-9 else np.ones_like(w) / len(w)

        # Rebalance frequency
        k_reb = int(self._get_config_value('REBALANCE_FREQ', 1) or 1)
        w_eff = self.w_prev if (k_reb > 1 and (self.t % k_reb != 0)) else w

        # Portfolio return
        p0 = self.prices[self.t]
        p1 = self.prices[self.t + 1]
        ret_vec = (p1 - p0) / (p0 + 1e-9)
        port_ret = float(np.dot(w_eff, ret_vec))

        # ── Reward ───────────────────────────────────────────────────────────
        rf = Config.RISK_FREE_DAILY if Config.REWARD_USE_EXCESS_RET else 0.0
        cost_bps = float(self._get_config_value('TURNOVER_COST', 0.00025))
        turnover_cost = float(np.sum(np.abs(w_eff - self.w_prev)) * cost_bps)
        reward = (port_ret - rf) - turnover_cost

        # Movement bonus
        if Config.MOVE_BONUS_COEF > 0.0:
            reward += Config.MOVE_BONUS_COEF * float(np.sum(np.abs(w_eff - self.w_prev)))

        # Variance penalty
        if Config.RISK_PENALTY_ALPHA > 0.0:
            win = Config.RISK_PENALTY_WINDOW
            if self.t >= win:
                ret_hist = (self.prices[self.t - win + 1:self.t + 1]
                            / self.prices[self.t - win:self.t] - 1.0)
                if ret_hist.shape[0] == win:
                    cov = np.cov(ret_hist.T, ddof=0)
                    reward -= Config.RISK_PENALTY_ALPHA * float(w_eff @ cov @ w_eff)

        # Two-sided HHI band (diversification shaping)
        hhi = float(np.sum(w_eff ** 2))
        if Config.DIVERSITY_SCHEME == 'two_sided_band':
            if hhi < Config.HHI_LOWER_BAND:
                reward -= Config.UNI_TOO_LOW_COEF * (Config.HHI_LOWER_BAND - hhi)
            elif hhi > Config.HHI_UPPER_BAND:
                reward -= Config.CONC_PEN_COEF * (hhi - Config.HHI_UPPER_BAND)
            else:
                reward += Config.BAND_CENTER_BONUS

        # Advantage tilt (relative-return bonus)
        if Config.ADV_COEF != 0.0:
            reward += Config.ADV_COEF * float(np.dot(w_eff, ret_vec - ret_vec.mean()))

        reward *= Config.REWARD_SCALE

        # Rolling reward normalization
        if Config.REWARD_NORMALIZE:
            self._rew_hist.append(reward)
            if len(self._rew_hist) > Config.REWARD_STD_WINDOW:
                self._rew_hist = self._rew_hist[-Config.REWARD_STD_WINDOW:]
            std = float(np.std(self._rew_hist, ddof=0))
            if std > 1e-8:
                reward = reward / std

        # ── State update ─────────────────────────────────────────────────────
        turnover_realized = float(np.sum(np.abs(w_eff - self.w_prev)))
        self.w_prev = w_eff
        self.t += 1
        terminated = self.t >= len(self.prices) - 1
        obs = self._get_obs()

        info: Dict[str, Any] = {
            'port_ret': port_ret,
            'turnover_cost': turnover_cost,
            'weights': w_eff.copy(),
            'hhi': hhi,
            'enh': 1.0 / (hhi + 1e-9),
            'turnover': turnover_realized,
        }
        return obs, float(reward), bool(terminated), False, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self.t = 0
        self._rew_hist = []

        if Config.RANDOM_INIT_WEIGHTS:
            raw = self._rng.normal(size=self.n_assets)
            raw = raw - raw.mean()
            w = np.exp(raw)
            self.w_prev = (w / w.sum()).astype(np.float32)
        else:
            self.w_prev = np.ones(self.n_assets, dtype=np.float32) / self.n_assets

        # Per-episode asset bias for symmetry breaking
        if Config.USE_ASSET_BIAS and Config.ASSET_BIAS_STD > 0:
            self._asset_bias = self._rng.normal(
                0.0, Config.ASSET_BIAS_STD, size=self.n_assets
            ).astype(np.float32)
        else:
            self._asset_bias = None

        return self._get_obs(), {}

    def _get_obs(self):
        feat_t = self.features[self.t]
        return np.concatenate([feat_t, self.w_prev], axis=0).astype(np.float32)

    def seed(self, seed=None):
        """Legacy compatibility."""
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        return [int(seed) if seed is not None else Config.SEED]
