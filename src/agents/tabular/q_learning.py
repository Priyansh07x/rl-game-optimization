"""Off-Policy Q-Learning Agent implementation."""

from typing import Any, Dict, Optional
import numpy as np

from src.agents.tabular.base_tabular import BaseTabularAgent


class QLearningAgent(BaseTabularAgent):
    """Off-Policy Q-Learning Agent for tabular reinforcement learning.

    Updates Q-values using the maximum expected reward over all actions in the next state:
      Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

    For terminal transitions (done=True):
      target = r
    """

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """Execute off-policy Q-Learning temporal difference update.

        Args:
            state: Current state index in [0, num_states).
            action: Action taken in current state in [0, num_actions).
            reward: Scalar reward received.
            next_state: Next state index in [0, num_states).
            done: Terminal flag boolean. If True, bootstrapping is omitted.

        Returns:
            Calculated temporal difference error (float).
        """
        s = self.validate_state(state)
        a = self.validate_action(action)
        r = float(reward)

        if done:
            target = r
        else:
            s_next = self.validate_state(next_state)
            max_next_q = float(np.max(self.q_table[s_next]))
            target = r + self.gamma * max_next_q

        td_error = target - self.q_table[s, a]
        self.q_table[s, a] += self.alpha * td_error

        return float(td_error)
