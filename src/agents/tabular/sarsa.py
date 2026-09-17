"""On-Policy SARSA Agent implementation."""

from typing import Any, Dict, Optional
import numpy as np

from src.agents.tabular.base_tabular import BaseTabularAgent


class SARSAAgent(BaseTabularAgent):
    """On-Policy SARSA Agent for tabular reinforcement learning.

    Updates Q-values using the action selected by the behavior policy in the next state:
      Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]

    For terminal transitions (done=True):
      target = r  (next_action is ignored and not bootstrapped)
    """

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: Optional[int],
        done: bool,
    ) -> float:
        """Execute on-policy SARSA temporal difference update.

        Args:
            state: Current state index in [0, num_states).
            action: Action taken in current state in [0, num_actions).
            reward: Scalar reward received.
            next_state: Next state index in [0, num_states).
            next_action: Action selected for next state. Required if done is False;
                         ignored (can be None or int) if done is True.
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
            if next_action is None:
                raise ValueError("next_action must be specified when done is False")
            s_next = self.validate_state(next_state)
            a_next = self.validate_action(next_action)
            target = r + self.gamma * self.q_table[s_next, a_next]

        td_error = target - self.q_table[s, a]
        self.q_table[s, a] += self.alpha * td_error

        return float(td_error)
