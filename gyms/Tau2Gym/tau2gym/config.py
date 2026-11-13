"""
Tau2Gym Configuration Module

This module provides configuration management for the Tau2Gym environment,
which integrates tau2-bench into UserRL framework.
"""

import os
from dataclasses import dataclass
from typing import Optional, Union, List


@dataclass
class Tau2GymConfig:
    """Configuration class for Tau2Gym environment.

    This configuration is designed to integrate tau2-bench with UserRL's training pipeline
    while preserving all original tau2-bench functionality, prompts, and evaluation logic.
    """

    # ===== Domain Configuration =====
    domain: str = "retail"  # Domain name: retail, airline, telecom
    task_split: str = "train"  # Task split: train, test, or base (for full evaluation)

    # ===== Data Configuration =====
    data_mode: str = "sequential"  # "random", "sequential", "single"
    data_source: Optional[Union[str, List[str]]] = None  # Specific task ID(s) or None for all
    tau2_data_dir: Optional[str] = None  # Path to tau2-bench data directory

    # ===== Model Configuration (User Simulator) =====
    user_llm: str = "gpt-4o-mini-2024-07-18"
    user_llm_args: dict = None  # LLM arguments for user simulator
    user_temperature: float = 0.7
    user_max_tokens: int = 1000

    # ===== Environment Configuration =====
    max_steps: int = 30  # Maximum steps per episode
    verbose: bool = False
    seed: Optional[int] = None
    solo_mode: bool = False  # If True, agent works independently without user interaction

    # ===== Reward Configuration =====
    reward_scale: float = 1.0
    step_penalty: float = 0.0
    normalize_rewards: bool = False

    # ===== Evaluation Configuration =====
    # These settings control which tau2-bench evaluation criteria to use
    use_action_evaluation: bool = True  # Evaluate based on required actions
    use_nl_assertions: bool = True  # Evaluate based on natural language assertions
    use_env_assertions: bool = True  # Evaluate based on environment state assertions

    def __post_init__(self):
        """Post-initialization processing."""
        # Set tau2 data directory
        if not self.tau2_data_dir:
            # Try environment variable first
            self.tau2_data_dir = os.getenv("TAU2_DATA_DIR")

            # Try to find tau2-bench in project
            if not self.tau2_data_dir:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                tau2_data_path = os.path.join(project_root, "tau2-bench", "data")
                if os.path.exists(tau2_data_path):
                    self.tau2_data_dir = tau2_data_path

        # Initialize user_llm_args if not provided
        if self.user_llm_args is None:
            self.user_llm_args = {
                "temperature": self.user_temperature,
                "max_tokens": self.user_max_tokens,
            }

        # Validate domain
        valid_domains = ["retail", "airline", "telecom", "mock"]
        if self.domain not in valid_domains:
            raise ValueError(f"Invalid domain: {self.domain}. Must be one of {valid_domains}")

        # Validate task_split
        valid_splits = ["train", "test", "base"]
        if self.task_split not in valid_splits:
            raise ValueError(f"Invalid task_split: {self.task_split}. Must be one of {valid_splits}")

    def validate(self):
        """Validate configuration parameters."""
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        if self.data_mode not in ["random", "sequential", "single"]:
            raise ValueError("data_mode must be 'random', 'sequential', or 'single'")

        # Check tau2 data directory exists
        if not self.tau2_data_dir or not os.path.exists(self.tau2_data_dir):
            raise ValueError(
                f"tau2_data_dir not found: {self.tau2_data_dir}. "
                "Please set TAU2_DATA_DIR environment variable or provide tau2_data_dir in config."
            )

        return True


def get_default_config() -> Tau2GymConfig:
    """Get default configuration for Tau2Gym.

    Returns:
        Default Tau2GymConfig instance configured for retail domain training.
    """
    return Tau2GymConfig()


def get_demo_config(domain: str = "mock") -> Tau2GymConfig:
    """Get demo configuration for Tau2Gym.

    Args:
        domain: Domain to use for demo (default: mock)

    Returns:
        Tau2GymConfig instance configured for interactive demo.
    """
    return Tau2GymConfig(
        domain=domain,
        task_split="base",
        verbose=True,
        max_steps=20,
        data_mode="single",
        data_source=None,  # Will use first task
    )


def get_train_config(domain: str = "retail") -> Tau2GymConfig:
    """Get training configuration for Tau2Gym.

    Args:
        domain: Domain to train on (default: retail)

    Returns:
        Tau2GymConfig instance configured for training.
    """
    return Tau2GymConfig(
        domain=domain,
        task_split="train",
        data_mode="random",
        verbose=False,
        max_steps=30,
    )


def get_test_config(domain: str = "retail") -> Tau2GymConfig:
    """Get test configuration for Tau2Gym.

    Args:
        domain: Domain to test on (default: retail)

    Returns:
        Tau2GymConfig instance configured for testing.
    """
    return Tau2GymConfig(
        domain=domain,
        task_split="test",
        data_mode="sequential",
        verbose=False,
        max_steps=30,
    )
