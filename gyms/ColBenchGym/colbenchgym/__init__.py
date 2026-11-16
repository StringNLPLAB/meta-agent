"""
ColBenchGym - Gymnasium environments for Collaborative Agent Bench
"""

from .config import (
    ColBenchGymConfig,
    get_default_config,
    get_code_config,
)
from .env.code_env import ColBenchCodeEnv

__version__ = "0.1.0"

__all__ = [
    "ColBenchGymConfig",
    "get_default_config",
    "get_code_config",
    "ColBenchCodeEnv",
]
