"""
Refit-specific configuration for evaluation scripts.
Separated from main Config to keep training config clean.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RefitConfig:
    """Configuration for monthly refit evaluation in eval_rl_refit.py"""
    
    # Learning parameters for refit
    LEARNING_RATE: float = 5e-5  # Conservative LR for fine-tuning
    CLIP_RANGE: float = 0.4      # Tighter clip range for stability
    
    # Training steps for refit (kept minimal to prevent overfitting)
    REFIT_STEPS: int = 4096      # Hardcoded in eval script, documented here
    
    # Position constraints during refit evaluation
    MAX_POSITION_SIZE: float = 0.8    # Relaxed from training's 0.35
    MIN_POSITION_SIZE: float = 0.0
    
    # Trading costs during refit evaluation
    TURNOVER_COST: float = 0.0001     # Reduced from training's 0.00025
    REBALANCE_FREQ: int = 1           # Daily rebalancing
    
    # Data window for refit training
    REFIT_LOOKBACK_DAYS: int = 90     # 3 months of recent data
    
    # Minimum data requirement
    MIN_REFIT_DAYS: int = 40          # Fallback to full history if insufficient
    
    # Model paths
    BASE_MODEL_PATH: str = "./rl_ppo/outputs/best_model.zip"