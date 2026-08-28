"""Dependency-free versions of the central Search-R1 RL calculations.

These functions intentionally use Python lists instead of torch tensors. They
make it possible to inspect every intermediate value before reading the batched
implementations in ``verl/trainer/ppo/core_algos.py``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence


@dataclass(frozen=True)
class PPOLoss:
    policy_loss: float
    clip_fraction: float
    approximate_kl: float


def _sample_std(values: Sequence[float]) -> float:
    """Match torch.std's default sample standard deviation for a 1-D group."""
    if len(values) < 2:
        return 1.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def group_normalized_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[Hashable],
    epsilon: float = 1e-6,
) -> list[float]:
    """Compute Search-R1's outcome-only GRPO advantage for each trajectory."""
    if len(rewards) != len(group_ids):
        raise ValueError("rewards and group_ids must have the same length")

    grouped_rewards: dict[Hashable, list[float]] = defaultdict(list)
    for reward, group_id in zip(rewards, group_ids):
        grouped_rewards[group_id].append(float(reward))

    stats = {}
    for group_id, values in grouped_rewards.items():
        if len(values) == 1:
            stats[group_id] = (0.0, 1.0)
        else:
            stats[group_id] = (sum(values) / len(values), _sample_std(values))

    return [
        (float(reward) - stats[group_id][0]) / (stats[group_id][1] + epsilon)
        for reward, group_id in zip(rewards, group_ids)
    ]


def ppo_clipped_policy_loss(
    old_log_probs: Sequence[float],
    new_log_probs: Sequence[float],
    advantages: Sequence[float],
    loss_mask: Sequence[int],
    clip_range: float = 0.2,
) -> PPOLoss:
    """Compute the masked PPO objective used by ``compute_policy_loss``."""
    lengths = {len(old_log_probs), len(new_log_probs), len(advantages), len(loss_mask)}
    if len(lengths) != 1:
        raise ValueError("all token-level inputs must have the same length")
    if not any(loss_mask):
        raise ValueError("loss_mask must select at least one token")

    losses = []
    clipped = []
    kls = []
    for old_log_prob, new_log_prob, advantage, keep in zip(
        old_log_probs, new_log_probs, advantages, loss_mask
    ):
        if not keep:
            continue
        log_ratio = new_log_prob - old_log_prob
        ratio = math.exp(log_ratio)
        clipped_ratio = min(max(ratio, 1.0 - clip_range), 1.0 + clip_range)
        unclipped_loss = -advantage * ratio
        clipped_loss = -advantage * clipped_ratio
        losses.append(max(unclipped_loss, clipped_loss))
        clipped.append(float(clipped_loss > unclipped_loss))
        kls.append(-log_ratio)

    count = len(losses)
    return PPOLoss(
        policy_loss=sum(losses) / count,
        clip_fraction=sum(clipped) / count,
        approximate_kl=sum(kls) / count,
    )


def reward_with_search_cost(answer_score: float, searches: int, cost: float = 0.02) -> float:
    """Example extension: retain outcome reward while charging for tool calls."""
    if searches < 0:
        raise ValueError("searches cannot be negative")
    if cost < 0:
        raise ValueError("cost cannot be negative")
    return float(answer_score) - searches * cost


def broadcast_trajectory_advantages(
    trajectory_advantages: Sequence[float], response_masks: Sequence[Sequence[int]]
) -> list[list[float]]:
    """Broadcast one outcome advantage over each trajectory's valid tokens."""
    if len(trajectory_advantages) != len(response_masks):
        raise ValueError("one response mask is required per trajectory")
    return [
        [advantage * int(keep) for keep in mask]
        for advantage, mask in zip(trajectory_advantages, response_masks)
    ]


def _format(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value:+.4f}" for value in values) + "]"


def main() -> None:
    rewards = [0, 0, 1, 1, 0]
    group_ids = ["question-1"] * len(rewards)
    advantages = group_normalized_advantages(rewards, group_ids)
    print("rewards:   ", rewards)
    print("advantages:", _format(advantages))
    print("group mean:", f"{sum(advantages) / len(advantages):+.6f}")

    token_advantages = [advantages[2]] * 4
    loss = ppo_clipped_policy_loss(
        old_log_probs=[-1.0, -1.2, -0.8, -1.1],
        new_log_probs=[-0.9, -1.3, -0.5, -1.1],
        advantages=token_advantages,
        loss_mask=[1, 0, 1, 1],
    )
    print("masked PPO:", loss)
    print("reward after two searches:", reward_with_search_cost(1.0, searches=2))


if __name__ == "__main__":
    main()
