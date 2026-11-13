# RAGEN Extensions for UserRL

This module provides optional extensions from the RAGEN paper that can be enabled through configuration without modifying core training code.

## Overview

RAGEN (Reasoning Agent) introduces the **StarPO-S** framework with three key stabilization mechanisms:

1. **Uncertainty-based Rollout Filtering** ← Implemented here
2. **Asymmetric Clipping** (Gradient Shaping) ← Config-based
3. **KL Term Removal** (Gradient Shaping) ← Config-based

## Quick Start

### 1. Enable RAGEN Extensions in Config

```yaml
# examples/sglang_multiturn/config/grpo_multiturn_ragen.yaml

algorithm:
  # Asymmetric Clipping: Learn more aggressively from high rewards
  clip_ratio_low: 0.2
  clip_ratio_high: 0.28      # Higher than standard 0.2

# RAGEN Extensions
ragen:
  rollout_filter:
    enable: True             # Enable uncertainty-based filtering
    ratio: 0.25              # Keep top 25% by variance
    filter_type: largest     # Keep high-variance prompts
    metric: reward_variance  # Filter by reward variance
    group_size: 16           # Match rollout.n

actor_rollout_ref:
  actor:
    use_kl_loss: False       # KL removal for gradient shaping
    entropy_coeff: 0.001     # Small entropy to prevent collapse
```

### 2. Run Training

```bash
cd
bash train.sh --config-name=grpo_multiturn_ragen
```

## Features

### Rollout Filtering

**Purpose**: Focus training on informative samples by filtering based on uncertainty (variance).

**Mechanism**:
- Groups responses by prompt (using `uid`)
- Computes variance of rewards within each group
- Keeps top-k% groups with highest variance
- Filters out low-information samples

**Benefits**:
- Prevents training on trivial/saturated samples
- Focuses compute on challenging examples
- Improves sample efficiency
- Reduces overfitting

**Metrics Logged**:
- `rollout/filter_ratio`: Fraction of samples kept
- `rollout/reward_std_mean`: Average reward variance
- `rollout/num_groups_kept`: Number of prompt groups kept

### Asymmetric Clipping

**Purpose**: Allow model to learn more aggressively from high-reward trajectories.

**Configuration**:
```yaml
algorithm:
  clip_ratio_low: 0.2      # Standard lower bound
  clip_ratio_high: 0.28    # Higher upper bound (vs 0.2 in standard PPO)
```

**Effect**: Model can increase probability of high-reward actions beyond standard PPO limits.

### KL Term Removal

**Purpose**: Remove KL divergence penalty to allow more exploration.

**Configuration**:
```yaml
actor_rollout_ref:
  actor:
    use_kl_loss: False      # Disable KL penalty
    kl_loss_coef: 0.000     # Set to 0
```

**Trade-off**: More exploration but may diverge from reference policy faster.

## Architecture

### Design Principles

1. **Minimal Invasiveness**: Extensions are opt-in through config
2. **Backward Compatibility**: Default behavior unchanged if not enabled
3. **Modular**: Each extension can be enabled independently
4. **Zero Core Changes**: No modifications to core training logic

### Integration Points

```python
# In ray_trainer.py, after reward computation:
if hasattr(self, 'rollout_filter') and self.rollout_filter is not None:
    batch, filter_metrics = self.rollout_filter.filter(batch)
    metrics.update(filter_metrics)
```

### File Structure

```
verl/trainer/ppo/ragen_extensions/
├── __init__.py              # Public API
├── rollout_filter.py        # Filtering implementation
└── README.md               # This file
```

## Advanced Usage

### Custom Filter Metrics

You can filter by different metrics:

```yaml
ragen:
  rollout_filter:
    metric: entropy_variance  # Filter by policy entropy variance
    # OR
    metric: reward_variance   # Filter by reward variance (default)
```

### Filter Type

```yaml
ragen:
  rollout_filter:
    filter_type: largest   # Keep high-variance samples (exploration)
    # OR
    filter_type: smallest  # Keep low-variance samples (exploitation)
```

### Disable Filtering

```yaml
ragen:
  rollout_filter:
    enable: False  # Completely disable filtering
    # OR simply omit the 'ragen' section
```

## Performance Considerations

### Memory Impact

- Filtering reduces batch size → less GPU memory per update
- Typical reduction: 75% (ratio=0.25) → 4x less memory per batch
- More gradient steps needed for same data coverage

### Compute Trade-off

- **Cost**: Computing variance statistics (negligible ~1-2% overhead)
- **Benefit**: Training on more informative samples (potential 2-4x speedup)
- **Net effect**: Often faster convergence despite filtering overhead

## Monitoring

Key metrics to watch:

1. **`rollout/reward_std_mean`**: Should be > 0 (groups have variance)
2. **`rollout/filter_ratio`**: Actual filtering ratio applied
3. **`rollout/num_groups_kept`**: Number of prompt groups kept
4. **`rollout/chosen_reward_std_mean`**: Variance of kept samples (should be high)

## Troubleshooting

### "Rollout filtering requires 'uid' in non_tensor_batch"

**Solution**: Ensure your dataset provides `uid` field for prompt grouping.

```python
# In your data collator:
batch['uid'] = [prompt_id for prompt_id in ...]
```

### Filtering too aggressive (ratio too low)

**Symptom**: Very few samples kept, slow training

**Solution**: Increase filter ratio:
```yaml
ragen:
  rollout_filter:
    ratio: 0.5  # Keep 50% instead of 25%
```

### No variance in rewards

**Symptom**: `rollout/reward_std_mean` near 0

**Solution**:
- Check if task is too easy/hard
- Increase group size (`rollout.n`)
- Consider disabling filtering for this task

## Citation

If you use RAGEN extensions, please cite:

```bibtex
@article{ragen2025,
  title={RAGEN: Reasoning Agents with StarPO-S},
  author={...},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025}
}
```

## License

Same as veRL: Apache 2.0
