import os
import yaml
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ColBenchGymConfig:
    """Complete configuration for ColBenchGym environment."""

    # Task configuration
    task_type: str = "code"  # Only "code" for Backend Programming

    # Environment simulator configuration (VLLM server for human collaborator simulation)
    env_hostname: str = "localhost"
    env_port: int = 8000
    env_model_name: str = ""  # Model name on the VLLM server
    env_base_url: str = ""  # Will be constructed from hostname and port if not provided
    env_api_key: str = ""  # API key for the environment simulator (empty for VLLM, or custom API key)

    # Environment configuration
    max_steps: int = 10
    min_questions: int = 0  # answers submitted before this many questions are refused
    verbose: bool = False
    seed: Optional[int] = 42
    human_response_char_limit: int = 400  # Character limit for human responses (code task)

    # Scoring configuration
    reward_scale: float = 1.0
    step_penalty: float = 0.0
    normalize_rewards: bool = False

    # Data configuration
    data_mode: str = "single"  # "random", "single", "list"
    data_source: Optional[Union[str, Dict[str, Any]]] = None  # Task dict or None

    def __post_init__(self):
        # Construct base URL if not provided
        if not self.env_base_url:
            self.env_base_url = f"http://{self.env_hostname}:{self.env_port}/v1"

        # Validate task type
        if self.task_type != "code":
            raise ValueError("task_type must be 'code'")

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'ColBenchGymConfig':
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ColBenchGymConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def validate(self) -> bool:
        """Validate configuration."""
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

        if self.task_type != "code":
            raise ValueError("task_type must be 'code'")

        if self.data_mode not in ["random", "single", "list"]:
            raise ValueError("data_mode must be 'random', 'single', or 'list'")

        if not self.env_hostname:
            raise ValueError("env_hostname must be provided")

        return True


# Pre-built configurations
def get_default_config(task_type: str = "code") -> ColBenchGymConfig:
    """Get default configuration."""
    return ColBenchGymConfig(task_type=task_type)


def get_code_config() -> ColBenchGymConfig:
    """Get configuration for Backend Programming tasks."""
    return ColBenchGymConfig(
        task_type="code",
        max_steps=10,
        verbose=True,
        human_response_char_limit=400
    )
