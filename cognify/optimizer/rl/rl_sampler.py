from typing import Callable, Any, Optional
import numpy as np
import optuna
from optuna.samplers import BaseSampler
import random

ESP = 1e-6

class RLSampler(BaseSampler):
    def __init__(
        self,
        cost_estimator: Callable[[dict[str, Any]], float],
        epsilon: float = 0.1,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.98,
        alpha: float = 0.1,
        n_startup_trials: int = 5,
        penalization_missed_constraints: int = 100,
        constraints_func: Optional[Callable[[optuna.trial.FrozenTrial], tuple]] = None,
    ):
        self.cost_estimator = cost_estimator
        # Exploitation vs exploration
        self.epsilon = epsilon  
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Learning Rate
        self.alpha = alpha

        # Cost Constraint
        self.penalization_missed_constraints = penalization_missed_constraints

        self.n_startup_trials = n_startup_trials
        self.constraints_func = constraints_func
        self.q_table = {
            # param -> option -> qvalue
        } 
    def init_q_value(self, param_name, param_distribution):
        self.q_table[param_name] = {choice: 0.0 for choice in param_distribution.choices}

    def sample_independent(self, study, trial, param_name, param_distribution, disable_exploration=False):
        is_categorical = hasattr(param_distribution, "choices")

        # BO has some startup trials with random choice. Mimicing behavior
        if trial.number < self.n_startup_trials:
            return (
                np.random.choice(param_distribution.choices)
                if is_categorical
                else param_distribution.single()
            )

        # Issue with not categorical data and current smapler
        if not is_categorical:
            return param_distribution.single()

        if param_name not in self.q_table:
            self.init_q_value(param_name, param_distribution)

        if np.random.rand() < self.epsilon and not disable_exploration: # Exploration
            return np.random.choice(param_distribution.choices)
        # Exploitation
        items = self.q_table[param_name].items()
        max_value = max(self.q_table[param_name].values())
        return random.choice([k for k, v in items if v == max_value])

    def calculate_adjusted_reward(self, value, trial):
        reward = value if value is not None else 0.0
        params = trial.params
        cost = self.cost_estimator(params) + ESP
        adjusted_reward = reward - np.log(cost)

        if self.constraints_func is not None:
            constraint_tuple = self.constraints_func(trial)
            if any(val < 0 for val in constraint_tuple):
                adjusted_reward -= self.penalization_missed_constraints
        return adjusted_reward

    def after_trial(self, study, trial, state, value):
        if state != optuna.trial.TrialState.COMPLETE:
            return

        adjusted_reward = self.calculate_adjusted_reward(value, trial)
        # Update the Q-values for each categorical parameter.
        for param_name, param_distribution in trial.distributions.items():
            if not hasattr(param_distribution, 'choices'):
                continue
            action = trial.params.get(param_name)
            self.q_table.setdefault(param_name, {})
            current_q = self.q_table[param_name].get(action, 0.0)

            # Q(a)←Q(a)+α[r−Q(a)] Multi Armed bandit based update function
            self.q_table[param_name][action] = current_q + self.alpha * (adjusted_reward - current_q)

        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def infer_relative_search_space(self, study, trial):
        """
        required function for abstract class. Not used by RL
        """
        return {}

    def sample_relative(self, study, trial, search_space):
        """
        required function for abstract class. Not used by RL
        """
        return {}
