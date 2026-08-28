import math
import unittest

from example.learning.agentic_rl_basics import (
    broadcast_trajectory_advantages,
    group_normalized_advantages,
    ppo_clipped_policy_loss,
    reward_with_search_cost,
)
from example.learning.mock_retriever import QueryRequest, load_corpus, retrieve, search_corpus


class GRPOTest(unittest.TestCase):
    def test_group_advantages_are_centered(self):
        advantages = group_normalized_advantages(
            rewards=[0, 0, 1, 1, 0],
            group_ids=["q1"] * 5,
        )
        self.assertAlmostEqual(sum(advantages), 0.0, places=6)
        self.assertGreater(advantages[2], 0)
        self.assertLess(advantages[0], 0)

    def test_groups_are_normalized_independently(self):
        advantages = group_normalized_advantages(
            rewards=[0, 1, 10, 12],
            group_ids=["q1", "q1", "q2", "q2"],
        )
        self.assertAlmostEqual(advantages[0], -advantages[1], places=6)
        self.assertAlmostEqual(advantages[2], -advantages[3], places=6)

    def test_advantage_is_broadcast_only_to_valid_tokens(self):
        actual = broadcast_trajectory_advantages([2.0], [[1, 1, 0]])
        self.assertEqual(actual, [[2.0, 2.0, 0.0]])


class PPOTest(unittest.TestCase):
    def test_masked_token_does_not_change_loss(self):
        baseline = ppo_clipped_policy_loss(
            old_log_probs=[-1.0, -1.0],
            new_log_probs=[-0.9, -1.0],
            advantages=[1.0, 1.0],
            loss_mask=[1, 0],
        )
        changed_masked_token = ppo_clipped_policy_loss(
            old_log_probs=[-1.0, -100.0],
            new_log_probs=[-0.9, 100.0],
            advantages=[1.0, -100.0],
            loss_mask=[1, 0],
        )
        self.assertEqual(baseline, changed_masked_token)

    def test_positive_advantage_uses_clipped_ratio(self):
        result = ppo_clipped_policy_loss(
            old_log_probs=[-1.0],
            new_log_probs=[-1.0 + math.log(2.0)],
            advantages=[1.0],
            loss_mask=[1],
            clip_range=0.2,
        )
        self.assertAlmostEqual(result.policy_loss, -1.2)
        self.assertEqual(result.clip_fraction, 1.0)

    def test_search_cost(self):
        self.assertAlmostEqual(reward_with_search_cost(1.0, searches=2), 0.96)


class MockRetrieverTest(unittest.TestCase):
    def test_query_returns_expected_document(self):
        result = search_corpus("Leonardo da Vinci Pavia Cathedral", load_corpus(), topk=1)
        self.assertEqual(result[0][0]["id"], "4")

    def test_http_contract_shape(self):
        response = retrieve(QueryRequest(
            queries=["Leonardo da Vinci Pavia Cathedral"],
            topk=1,
            return_scores=True,
        ))
        self.assertEqual(set(response), {"result"})
        self.assertEqual(len(response["result"]), 1)
        self.assertEqual(set(response["result"][0][0]), {"document", "score"})


if __name__ == "__main__":
    unittest.main()
