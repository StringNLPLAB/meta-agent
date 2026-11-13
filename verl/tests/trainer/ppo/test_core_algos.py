import torch

from verl.trainer.ppo.core_algos import compute_policy_loss


def test_policy_loss_zeroes_out_of_range_ratios():
    old_log_prob = torch.zeros(1, 4)
    ratios = torch.tensor([[0.69, 0.70, 6.00, 6.01]])
    log_prob = torch.log(ratios)
    advantages = torch.ones(1, 4)
    response_mask = torch.ones(1, 4)

    pg_loss, pg_clipfrac, _, pg_clipfrac_lower = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        cliprange_low=0.3,
        cliprange_high=5.0,
        clip_zero_out_of_range=True,
    )

    expected_loss = torch.tensor(-(0.0 + 0.70 + 6.00 + 0.0) / 4)
    assert torch.isclose(pg_loss, expected_loss)
    assert torch.isclose(pg_clipfrac, torch.tensor(0.5))
    assert torch.isclose(pg_clipfrac_lower, torch.tensor(0.25))


def test_policy_loss_zeroes_out_of_range_ratios_with_negative_advantages():
    old_log_prob = torch.zeros(1, 4)
    ratios = torch.tensor([[0.69, 0.70, 6.00, 6.01]])
    log_prob = torch.log(ratios)
    advantages = -torch.ones(1, 4)
    response_mask = torch.ones(1, 4)

    pg_loss, pg_clipfrac, _, pg_clipfrac_lower = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        cliprange_low=0.3,
        cliprange_high=5.0,
        clip_zero_out_of_range=True,
    )

    expected_loss = torch.tensor((0.0 + 0.70 + 6.00 + 0.0) / 4)
    assert torch.isclose(pg_loss, expected_loss)
    assert torch.isclose(pg_clipfrac, torch.tensor(0.5))
    assert torch.isclose(pg_clipfrac_lower, torch.tensor(0.25))
