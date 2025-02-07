# This class was mostly generated using AI
import unittest
import numpy as np
import optuna
from optuna.trial import TrialState
import optuna.distributions as dists
from typing import Any

# Import the RLSampler class from your module.
# Adjust the import below to match your project structure.
from rl_sampler import RLSampler, ESP

# -----------------------------------------------------------------------------
# Dummy classes for simulating trials and non-categorical distributions
# -----------------------------------------------------------------------------

class DummyTrial:
    """
    A minimal dummy trial to simulate the attributes used by RLSampler.
    """
    def __init__(self, number: int, params: dict[str, Any], distributions: dict[str, Any]):
        self.number = number          # The trial number
        self.params = params          # A dictionary of parameter selections
        self.distributions = distributions  # A dict mapping parameter names to distributions


class DummyNonCategoricalDistribution:
    """
    A dummy non-categorical distribution with a simple single() method.
    """
    def single(self):
        return 42

# -----------------------------------------------------------------------------
# DummySampler subclass for testing param_cost_estimator behavior.
# This subclass overrides param_cost_estimator so that it stores the
# converted external proposal for inspection.
# -----------------------------------------------------------------------------

class DummySampler(RLSampler):
    def param_cost_estimator(self, trial_proposal: dict[str, Any]) -> float:
        total_cost = 0.0
        ext_trial_proposal = {}
        for param_name, dist in self.param_categorical_dist.items():
            if dist.single():
                continue
            if isinstance(trial_proposal[param_name], str):
                internal_repr = dist.to_internal_repr(trial_proposal[param_name])
            else:
                internal_repr = trial_proposal[param_name]
            ext_trial_proposal[param_name] = dist.to_external_repr(internal_repr)
        self.last_ext_trial_proposal = ext_trial_proposal
        return total_cost

# -----------------------------------------------------------------------------
# Combined Unit Tests for the RLSampler
# -----------------------------------------------------------------------------

class TestRLSampler(unittest.TestCase):

    # --- Tests for sample_independent and after_trial methods ---

    def setUp(self):
        # A dummy cost estimator (not used in most tests but required for instantiation)
        self.cost_estimator = lambda params: 1.0
        self.sampler = RLSampler(
            cost_estimator=self.cost_estimator,
            epsilon=0.0,  # set epsilon=0 to force greedy selection post startup
            alpha=0.1,
            n_startup_trials=5
        )

    def test_sample_independent_startup_categorical(self):
        """When in the startup phase, sampling a categorical parameter returns a random choice."""
        trial = DummyTrial(number=0, params={}, distributions={})
        choices = ["a", "b", "c"]
        cat_dist = dists.CategoricalDistribution(choices=choices)
        result = self.sampler.sample_independent(None, trial, "param1", cat_dist)
        self.assertIn(result, choices)

    def test_sample_independent_startup_non_categorical(self):
        """When in the startup phase, a non-categorical parameter is sampled via its 'single()' method."""
        trial = DummyTrial(number=0, params={}, distributions={})
        dummy_dist = DummyNonCategoricalDistribution()
        result = self.sampler.sample_independent(None, trial, "param_noncat", dummy_dist)
        self.assertEqual(result, 42)

    def test_sample_independent_post_startup_categorical(self):
        """After the startup phase, the Q‑table is used to select the action (with epsilon=0)."""
        trial = DummyTrial(number=10, params={}, distributions={})
        choices = ["a", "b", "c"]
        cat_dist = dists.CategoricalDistribution(choices=choices)
        # First call initializes the Q‑table; the result is one of the choices.
        result = self.sampler.sample_independent(None, trial, "param1", cat_dist)
        self.assertIn(result, choices)

        # Simulate a Q‑table update by manually setting Q‑values.
        self.sampler.q_table["param1"] = {"a": 1.0, "b": 0.5, "c": 0.0}
        # With epsilon=0, the sampler should select the option with the highest Q‑value ("a").
        result = self.sampler.sample_independent(None, trial, "param1", cat_dist)
        self.assertEqual(result, "a")

    def test_after_trial_complete_without_constraint(self):
        """After a COMPLETE trial, the Q‑table is updated for categorical parameters."""
        choices = ["a", "b"]
        cat_dist = dists.CategoricalDistribution(choices=choices)
        trial = DummyTrial(
            number=5,
            params={"param1": "a"},
            distributions={"param1": cat_dist}
        )
        # Call after_trial with a result value of 10.
        self.sampler.after_trial(None, trial, TrialState.COMPLETE, 10)
        # With cost_estimator returning 1.0, cost ≈ 1.0 + ESP and:
        # adjusted_reward = 10 - log(1.0 + ESP) ≈ 10.
        # With alpha=0.1 and initial Q=0, new Q-value ≈ 1.0.
        expected = 0.1 * (10 - np.log(1.0 + ESP))
        self.assertAlmostEqual(self.sampler.q_table["param1"]["a"], expected, places=5)

    def test_after_trial_non_complete(self):
        """If trial state is not COMPLETE, Q‑table should not be updated."""
        choices = ["a", "b"]
        cat_dist = dists.CategoricalDistribution(choices=choices)
        trial = DummyTrial(
            number=6,
            params={"param1": "a"},
            distributions={"param1": cat_dist}
        )
        self.sampler.q_table["param1"] = {"a": 0.5}
        self.sampler.after_trial(None, trial, TrialState.FAIL, 10)
        self.assertEqual(self.sampler.q_table["param1"]["a"], 0.5)

    def test_after_trial_with_constraint_violation(self):
        """If a constraints function is provided and signals a violation, extra penalty is applied."""
        choices = ["a", "b"]
        cat_dist = dists.CategoricalDistribution(choices=choices)
        trial = DummyTrial(
            number=7,
            params={"param1": "a"},
            distributions={"param1": cat_dist}
        )
        self.sampler.constraints_func = lambda trial: (-1,)  # Violation signaled
        self.sampler.after_trial(None, trial, TrialState.COMPLETE, 10)
        expected = 0.1 * (10 - np.log(1.0 + ESP) - 100)
        self.assertAlmostEqual(self.sampler.q_table["param1"]["a"], expected, places=5)

    def test_infer_relative_search_space(self):
        """The minimal implementation returns an empty dict."""
        result = self.sampler.infer_relative_search_space(None, None)
        self.assertEqual(result, {})

    def test_sample_relative(self):
        """The minimal implementation returns an empty dict."""
        result = self.sampler.sample_relative(None, None, {})
        self.assertEqual(result, {})

    # --- Tests for param_cost_estimator using Optuna distributions ---

    def test_param_cost_estimator_with_string_value(self):
        """
        Test that for a string input the categorical distribution converts it to its internal
        representation and back to external.
        """
        dummy_sampler = DummySampler(cost_estimator=self.cost_estimator)
        cat_dist = dists.CategoricalDistribution(choices=["a", "b", "c"])
        dummy_sampler.param_categorical_dist = {"param1": cat_dist}
        trial_proposal = {"param1": "b"}
        dummy_sampler.param_cost_estimator(trial_proposal)
        expected = {"param1": "b"}  # "b" converts to internal index 1 then back to "b"
        self.assertEqual(dummy_sampler.last_ext_trial_proposal, expected)
        self.assertIsInstance(dummy_sampler.last_ext_trial_proposal["param1"], str)

    def test_param_cost_estimator_with_non_string_value(self):
        """
        Test that for a non-string input the value is used directly.
        For a distribution with choices ["x", "y", "z"], providing 2 should yield "z".
        """
        dummy_sampler = DummySampler(cost_estimator=self.cost_estimator)
        cat_dist = dists.CategoricalDistribution(choices=["x", "y", "z"])
        dummy_sampler.param_categorical_dist = {"param1": cat_dist}
        trial_proposal = {"param1": 2}  # Provided as internal index
        dummy_sampler.param_cost_estimator(trial_proposal)
        expected = {"param1": "z"}
        self.assertEqual(dummy_sampler.last_ext_trial_proposal, expected)
        self.assertIsInstance(dummy_sampler.last_ext_trial_proposal["param1"], str)

    def test_param_cost_estimator_skips_if_single(self):
        """
        If the distribution's single() returns True (only one option),
        the parameter is skipped.
        """
        dummy_sampler = DummySampler(cost_estimator=self.cost_estimator)
        cat_dist = dists.CategoricalDistribution(choices=["only"])
        dummy_sampler.param_categorical_dist = {"param1": cat_dist}
        trial_proposal = {"param1": "only"}
        dummy_sampler.param_cost_estimator(trial_proposal)
        expected = {}
        self.assertEqual(dummy_sampler.last_ext_trial_proposal, expected)

    def test_param_cost_estimator_type_conversions(self):
        """
        Verify that if a non‑string input is provided for a distribution with numeric choices,
        the conversion produces an output of the correct type.
        """
        dummy_sampler = DummySampler(cost_estimator=self.cost_estimator)
        choices = [10, 20, 30]
        cat_dist = dists.CategoricalDistribution(choices=choices)
        dummy_sampler.param_categorical_dist = {"param1": cat_dist}
        trial_proposal = {"param1": 1}  # Taken as an internal index
        dummy_sampler.param_cost_estimator(trial_proposal)
        expected = {"param1": 20}  # to_external_repr(1) yields 20
        self.assertEqual(dummy_sampler.last_ext_trial_proposal, expected)
        self.assertIsInstance(dummy_sampler.last_ext_trial_proposal["param1"], type(choices[0]))

    # --- Simulated RL Exploration Test ---
    def test_rl_exploration_finds_best_option(self):
        """
        Simulate a series of trials where the RL sampler updates its Q‑table.
        In our discrete space the hyperparameter "hp" has three options.
        We define a dummy cost estimator so that option "b" has a lower cost (and thus a higher adjusted reward)
        than the others. After many trials the sampler should favor "b".
        """
        # Define a dummy cost estimator: cost is 1.0 if the option is "b", else 2.0.
        def dummy_cost_estimator(params: dict[str, Any]) -> float:
            return 1 if params["hp"] == "b" else 2

        # Create an RL sampler with epsilon=0 to force greedy (post-startup) behavior.
        sampler = RLSampler(
            cost_estimator=dummy_cost_estimator,
            epsilon=0.25,
            alpha=0.1,
            n_startup_trials=5
        )
        # Define a categorical distribution for hyperparameter "hp".
        choices = ["a", "b", "c"]
        cat_dist = dists.CategoricalDistribution(choices=choices)
        sampler.param_categorical_dist = {"hp": cat_dist}

        # Simulate many trials.
        # The reward is constant (10) for all trials.
        # Option "b" gets an adjusted_reward ≈ 10 - log(1) = 10,
        # while options "a" and "c" get ≈ 10 - log(2) ≈ 9.307.
        num_trials = 100
        for trial_number in range(5, 5 + num_trials):
            trial = DummyTrial(trial_number, params={}, distributions={"hp": cat_dist})
            # Sample an action using sample_independent.
            action = sampler.sample_independent(None, trial, "hp", cat_dist)
            trial.params["hp"] = action
            # Simulate evaluation: constant reward of 10.
            sampler.after_trial(None, trial, TrialState.COMPLETE, 1)

        # After many trials, the Q‑table for "hp" should have the highest value for option "b".
        q_values = sampler.q_table.get("hp", {})
        self.assertTrue(q_values, "Q‑table for 'hp' should not be empty.")
        best_option = max(q_values, key=q_values.get)
        # Debug output (optional):
        print("Final Q‑values for 'hp':", q_values, best_option)
        self.assertEqual(best_option, "b", "The RL sampler should favor option 'b' as the best option.")

        # Finally, a new trial (post startup) should choose "b" deterministically.
        new_trial = DummyTrial(100, params={}, distributions={"hp": cat_dist})
        chosen = sampler.sample_independent(None, new_trial, "hp", cat_dist, disable_exploration=True)
        self.assertEqual(chosen, "b", "The sampler should choose 'b' as the best option.")


if __name__ == '__main__':
    unittest.main()
