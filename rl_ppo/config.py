"""PPO training configuration.

All hyperparameters for the DRL portfolio optimization pipeline.
Refit-specific overrides live in refit_config.py.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # ── Paths ────────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    ETF_DATA_PATH: Path = BASE_DIR / "data" / "etf_data_with_indicators.csv"

    # ── Universe ─────────────────────────────────────────────────────────
    ETF_TICKERS = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VNQ', 'TLT', 'IEF', 'GLD', 'USO']

    # ── Date splits ──────────────────────────────────────────────────────
    TRAIN_START = '2019-01-01'
    TRAIN_END   = '2024-05-31'
    VAL_START   = '2024-06-01'
    VAL_END     = '2024-12-31'
    TEST_START  = '2025-01-01'
    TEST_END    = '2025-07-01'

    # ── Risk-free rate ───────────────────────────────────────────────────
    RISK_FREE_ANNUAL: float = 0.04
    RISK_FREE_DAILY: float = RISK_FREE_ANNUAL / 252

    # ── PPO hyperparameters ──────────────────────────────────────────────
    SEED = 42
    N_ENVS = 1
    N_STEPS = 4096
    BATCH_SIZE = 512
    N_EPOCHS = 5
    GAMMA = 0.995
    CLIP_RANGE = 0.2
    LEARNING_RATE = 2e-4
    DEVICE: str = "auto"

    # ── Policy architecture ──────────────────────────────────────────────
    POLICY_NET_ARCH = {"pi": [256, 256], "vf": [256, 256]}
    ACTIVATION_FN = "relu"
    TOTAL_TIMESTEPS: int = 300_000

    # ── Trading frictions ────────────────────────────────────────────────
    TURNOVER_COST: float = 0.00025
    REBALANCE_FREQ: int = 1

    # ── Reward shaping ───────────────────────────────────────────────────
    REWARD_SCALE: float = 1.0
    REWARD_USE_EXCESS_RET: bool = True
    RISK_PENALTY_ALPHA: float = 0.001
    RISK_PENALTY_WINDOW: int = 20

    STACK_LEN: int = 10
    NORM_WINDOW: int = 63

    # ── Entropy annealing ────────────────────────────────────────────────
    ENT_COEF_INITIAL: float = 0.02
    ENT_COEF_FINAL: float = 0.002
    ENT_COEF_ANNEAL_STEPS: int = 80_000
    GAE_LAMBDA: float = 0.95
    MAX_GRAD_NORM: float = 0.5
    VF_COEF: float = 0.5
    ORTHO_INIT: bool = True

    # ── KL annealing ─────────────────────────────────────────────────────
    TARGET_KL_INITIAL: float = 0.20
    TARGET_KL_FINAL: float = 0.05
    TARGET_KL_ANNEAL_STEPS: int = 100_000
    TARGET_KL_DELAY_STEPS: int = 40_000
    USE_SDE: bool = True
    SDE_SAMPLE_FREQ: int = 4

    # ── Logit clipping ──────────────────────────────────────────────────
    ACTION_TEMPERATURE: float = 1.35
    LOGIT_CLIP: float = 4.0
    INITIAL_LOGIT_CLIP: float = 4.0
    FINAL_LOGIT_CLIP: float = 3.0
    LOGIT_CLIP_ANNEAL_STEPS: int = 60_000

    # ── Reward normalization ─────────────────────────────────────────────
    REWARD_NORMALIZE: bool = True
    REWARD_STD_WINDOW: int = 60

    # ── Diversification (two-sided HHI band) ─────────────────────────────
    DIVERSITY_SCHEME: str = "two_sided_band"
    HHI_LOWER_BAND: float = 0.13
    HHI_UPPER_BAND: float = 0.28
    UNI_TOO_LOW_COEF: float = 0.05
    CONC_PEN_COEF: float = 0.02
    BAND_CENTER_BONUS: float = 0.0
    ADV_COEF: float = 0.01

    # ── Position constraints & exploration ───────────────────────────────
    MAX_POSITION_SIZE: float = 0.35
    MIN_POSITION_SIZE: float = 0.0
    INCLUDE_PREV_WEIGHTS: bool = True
    RANDOM_INIT_WEIGHTS: bool = True
    MOVE_BONUS_COEF: float = 0.01
    USE_ASSET_BIAS: bool = True
    ASSET_BIAS_STD: float = 0.05

    # ── Validation & early stopping ──────────────────────────────────────
    VAL_N_WINDOWS: int = 3
    VAL_EVAL_FREQ: int = 5_000
    VAL_EARLY_STOP_PATIENCE: int = 15
    ENABLE_SOFT_GUARD: bool = True
    WORST_GUARD_THRESHOLD: float = -1.0
    SOFT_GUARD_LAMBDA: float = 0.25

    # ── Feature engineering ──────────────────────────────────────────────
    FEAT_ADD_RET_HORIZONS: bool = True
    FEAT_RET_HORIZONS = (1, 5, 21, 63)
    FEAT_ADD_REALIZED_VOL: bool = True
    FEAT_VOL_WINDOWS = (5, 21, 63)
    FEAT_ADD_XS_RANKS: bool = True
    FEAT_ADD_MEAN_CORR: bool = True
    FEAT_CORR_WINDOW: int = 21
    FEAT_ADD_DOWNSIDE_SEMIVOL: bool = True
    FEAT_SEMIVOL_WINDOWS = (21, 63)
    FEAT_ADD_TIME_CYCLICAL: bool = True
    FEAT_ADD_ABS_RET: bool = True