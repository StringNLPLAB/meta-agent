"""Finalize rewards already returned by the synchronous Tau2 adapter."""

import math


def tau2_step_reward_scores(environment, conversation_histories):
    """Return native-adapter scores, or None for the legacy simulation API.

    Tau2Env.step already adds the terminal score to its last returned reward.
    Preserve that history for PPO and keep the cached native task score as a
    separate metric. Calling the legacy inner-environment reward API cannot
    recover this adapter's score and must not overwrite its last turn.
    """
    if environment is None or not hasattr(environment, '_final_reward'):
        return None
    rewards = [float(turn.get('reward', 0.0) or 0.0) for turn in conversation_histories]
    terminal = (float(environment._final_reward)
                if getattr(environment, 'episode_complete', False) else 0.0)
    if not all(math.isfinite(value) for value in [terminal, *rewards]):
        raise ValueError('Nonfinite Tau2 step or terminal reward')
    return {'interact_with_env': sum(rewards),
            'interact_with_env_original': terminal}
