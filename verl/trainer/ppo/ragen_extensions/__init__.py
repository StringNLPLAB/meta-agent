# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RAGEN Extensions for UserRL

This module provides optional extensions from the RAGEN paper (https://arxiv.org/abs/2501.xxxxx)
including:
- Uncertainty-based Rollout Filtering (StarPO-S stabilization mechanism)
- Asymmetric Clipping (Gradient Shaping)
- KL Term Removal (Gradient Shaping)

These extensions can be enabled through configuration without modifying core training code.
"""

from .rollout_filter import (
    RolloutFilter,
    RolloutFilterConfig,
    RewardRolloutFilter,
    EntropyRolloutFilter,
    create_rollout_filter,
)

__all__ = [
    "RolloutFilter",
    "RolloutFilterConfig",
    "RewardRolloutFilter",
    "EntropyRolloutFilter",
    "create_rollout_filter",
]
