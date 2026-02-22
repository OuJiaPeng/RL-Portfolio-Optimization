.PHONY: train eval clean

# Train the PPO agent from scratch (writes best_model.zip to models/)
train:
	python -m rl_ppo.train_rl

# Evaluate with monthly refit on the test period (Jan–Jun 2025)
eval:
	python -m rl_ppo.eval_rl_refit

# Remove caches and temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
