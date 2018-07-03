from random import random, randrange


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()

    def update_memory(self, observation, action, reward, next_observation, done):
        pass

class QLearningAgent(object):
    """Agent with Q-table"""
    def __init__(self, action_space):
        self.action_space = action_space.n
        self.memory = dict()  # Q

        self.LR = 0.5
        self.DF = 0.99
        self.RF = 0.5

    def act(self, observation):
        if random() < self.RF:
            return randrange(self.action_space)
        self._fill_memory(observation)
        action = 0
        for i in range(self.action_space):
            if self.memory[observation][i] > self.memory[observation][action]:
                action = i
        return action

    def _fill_memory(self, value):
        if value not in self.memory:
            self.memory[value] = [random() * 2 - 1 for i in range(self.action_space)]

    def update_memory(self, observation, action, reward, next_observation, done):
        self._fill_memory(observation)
        self._fill_memory(next_observation)

        self.memory[observation][action] = (1 - self.LR) * self.memory[observation][action] + self.LR * \
            (reward + self.DF * max([self.memory[next_observation][i] for i in range(self.action_space)]))
