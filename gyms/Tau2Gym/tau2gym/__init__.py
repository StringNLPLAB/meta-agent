"""
Tau2Gym - Tau2-Bench Integration for UserRL

This package integrates tau2-bench into the UserRL framework, providing
a Gymnasium-compatible environment that preserves all original tau2-bench
functionality, prompts, and evaluation logic.

Features:
- Direct wrapper around tau2-bench's AgentGymEnv
- Support for train/test splits for proper RL training
- Multiple domains: retail, airline, telecom
- Preserves original prompts and agent context
- Identical evaluation metrics to tau2-bench
"""

from .config import (
    Tau2GymConfig,
    get_default_config,
    get_demo_config,
    get_train_config,
    get_test_config,
)
from .env.tau2_env import Tau2Env

__all__ = [
    "Tau2Env",
    "Tau2GymConfig",
    "get_default_config",
    "get_demo_config",
    "get_train_config",
    "get_test_config",
]

__version__ = "1.0.0"

# Register with gymnasium
try:
    import gymnasium as gym
    gym.register(
        id='Tau2Gym-v0',
        entry_point='tau2gym.env:Tau2Env',
        max_episode_steps=30
    )
except ImportError:
    pass
