"""
Template for implementing QLearner  (c) 2015 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tucker Balch (replace with your name)
GT User ID: crailton3 (replace with your User ID)
GT ID: 900897987 (replace with your GT ID)
"""

import random as rand
import numpy as np


class QLearner(object):
    """
    This is a Q learner object.

    :param num_states: The number of states to consider.
    :type num_states: int
    :param num_actions: The number of actions available..
    :type num_actions: int
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
    :type alpha: float
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
    :type gamma: float
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
    :type rar: float
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
    :type radr: float
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
    :type dyna: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
    :type verbose: bool
    """

    def author(self):
        return "crailton3"

    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=200,
        verbose=False,
    ):
        """
        Constructor method
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.state = 0
        self.action = 0
        self.qtable = np.zeros((self.num_states, self.num_actions))

        # dyna model: (s,a) -> list of (s', r)
        self.model = {}
        self.seen_sa = []
        self.buf_cap = 10  # max buffer length per (s,a)

    def querysetstate(self, state):
        """
        Update the state without updating the Q-table
        :param state: The new state
        :type state: int
        :return: The selected action
        :rtype: int
        """
        self.state = state

        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = self.epsilon_greedy(state, self.rar)

        self.action = action

        if self.verbose:
            print(f"s = {self.state}, a = {self.action}")
        return self.action

    def query(self, next_state, reward):
        """
        Update the Q table and return an action
        :param next_state: The new state
        :type next_state: int
        :param reward: The immediate reward
        :type reward: float
        :return: The selected action
        :rtype: int
        """
        previous_state = self.state
        previous_action = self.action

        # real experience Q-update
        self.q_update(previous_state, previous_action, next_state, reward)

        # update model for Dyna-Q
        key = (previous_state, previous_action)
        buffer = self.model.get(key)
        if buffer is None:
            buffer = []
            self.model[key] = buffer
            self.seen_sa.append(key)
        buffer.append((next_state, reward))
        if len(buffer) > self.buf_cap:
            buffer.pop(0)

        # Dyna-Q planning updates
        if self.dyna > 0 and self.seen_sa:
            total_samples = sum(len(self.model[k]) for k in self.seen_sa)
            num_planning_steps = min(self.dyna, total_samples) if total_samples > 0 else 0
            for _ in range(num_planning_steps):
                # sample a previously seen (s,a)
                idx = rand.randint(0, len(self.seen_sa) - 1)
                sampled_state, sampled_action = self.seen_sa[idx]
                samples = self.model[(sampled_state, sampled_action)]
                j = rand.randint(0, len(samples) - 1)
                imagined_next_state, imagined_reward = samples[j]
                self.q_update(sampled_state, sampled_action, imagined_next_state, imagined_reward)

        # choose next action with epsilon-greedy
        next_action = self.epsilon_greedy(next_state, self.rar)
        # decay exploration rate
        self.rar *= self.radr

        # update internal state/action
        self.state = next_state
        self.action = next_action

        if self.verbose:
            print(f"s = {next_state}, a = {next_action}, r = {reward}")
        return next_action

    def q_update(self, current_state, current_action, next_state, reward):
        """
        Core Q-learning update rule.
        """
        curr = self.qtable[current_state, current_action]
        if reward >= 1.0:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.qtable[next_state, :])
        self.qtable[current_state, current_action] = (1.0 - self.alpha) * curr + self.alpha * target

    def epsilon_greedy(self, state, eps):
        """
        Epsilon-greedy policy with deterministic tie-breaking.
        """
        if rand.random() < eps:
            return rand.randint(0, self.num_actions - 1)
        row = self.qtable[state, :]
        max_val = np.max(row)
        ties = np.flatnonzero(row == max_val)
        # deterministic: always pick the smallest index among ties
        return int(ties[0])


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
