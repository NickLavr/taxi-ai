from random import random, randrange, choice
import math
import pickle

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


class MAXQAgent(object):
    def __init__(self, action_space, log = False):
        self.action_space = action_space.n
        self.log = log

        self.LR = 0.5
        self.DF = 0.99
        self.RF = 0.3

        # V[(Node ID, observation)] -> the expected cumulative reward of executing Node
        # starting in state observatioin until Node terminates.
        self.V = dict()

        # C[(Node ID, observation, Action (node ID))] -> the expected discounted cumulative
        # reward of completing subtask Node after invoking the subroutine for subtask Action in state observation
        self.C = dict()

        self.init_nodes()

    def save_memory(self, filename = 'memory.pickle'):
        pickle.dump([self.V, self.C], open(filename, 'wb'))

    def load_memory(self, filename = 'memory.pickle'):
        try:
            self.V, self.C = pickle.load(open(filename, 'rb'))
        except Exception as e:
            print(str(e), 'in load memory')

    def getQ(self, node, observation, action):
        if (node.id, observation, action.id) not in self.C:
            self.C[(node.id, observation, action.id)] = -1
        return self.getV(action, observation) + self.C[(node.id, observation, action.id)]

    def getV(self, node, observation):
        if node.is_primitive():
            if (node.id, observation) not in self.V:
                self.V[(node.id, observation)] = -1
            return self.V[(node.id, observation)]
        else:
            return max([self.getQ(node, observation, node.childrens[child_id]) for child_id in node.childrens])

    def taxiDecode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return list(reversed(out))

    def check_navigate(self, destination):
        locx, locy = [(0, 0), (0, 4), (4, 0), (4, 3)][destination]
        def func(observation):
            decoded = self.taxiDecode(observation)
            return decoded[0] == locx and decoded[1] == locy

        return func

    def check_pickup(self):
        def func(observation):
            decoded = self.taxiDecode(observation)
            return decoded[2] == 4
        return func

    def check_dropoff(self):
        def func(observation):
            decoded = self.taxiDecode(observation)
            return decoded[2] != 4
        return func

    def check_done(self):
        def func(observation):
            return False
        return func

    class Node:
        def __init__(self, agent, action, id, check_T, childrens = None):
            self.agent = agent
            self.id = id
            self.action = action
            self.T = check_T
            if childrens is None:
                self.childrens = dict()
            else:
                self.childrens = {i.id: i for i in childrens}

        def add_childer(self, child):
            self.childrens[child.id] = child

        def is_primitive(self):
            return len(self.childrens) == 0

        def choose_child(self, observation):
            if random() < self.agent.RF:
                return self.childrens[choice(list(self.childrens.keys()))]
            res = None
            res_Q_val = -math.inf
            for i in self.childrens:
                Q_val = self.agent.getQ(self, observation, self.childrens[i])
                if res is None or Q_val > res_Q_val:
                    res_Q_val = Q_val
                    res = self.childrens[i]
            return res

    def init_nodes(self):
        primitive_move = []
        for i in range(4):
            primitive_move.append(self.Node(self, i, i, None))
        primitive_pickup = self.Node(self, 4, 4, None)
        primitive_dropoff = self.Node(self, 5, 5, None)
        navigates = [self.Node(self, -1, 6 + i, self.check_navigate(i), primitive_move) for i in range(4)]
        get = self.Node(self, -1, 10, self.check_pickup(), [primitive_pickup] + navigates)
        put = self.Node(self, -1, 11, self.check_dropoff(), [primitive_dropoff] + navigates)
        self.root = self.Node(self, -1, 12, self.check_done(), [get, put])

    def MAXQ(self, current_node, observation, env):
        if self.log:
            print("in node {}, observation {}".format(current_node.id, self.taxiDecode(observation)))
        if current_node.is_primitive():
            next_observation, reward, done, _ = env.step(current_node.action)
            self.V[(current_node.id, observation)] = self.V[(current_node.id, observation)] * (1 - self.LR) + \
                self.LR * reward
            if self.log:
                env.render()
            if done:
                return (1, "DONE")
            return (1, next_observation)
        else:
            count = 0
            while not current_node.T(observation):
                next_node = current_node.choose_child(observation)
                N, next_observation = self.MAXQ(next_node, observation, env)
                if next_observation == "DONE":
                    return (count + N, next_observation)
                if (current_node.id, observation, next_node.id) not in self.C:
                    self.C[(current_node.id, observation, next_node.id)] = -1
                self.C[(current_node.id, observation, next_node.id)] = \
                    (1 - self.LR) * self.C[(current_node.id, observation, next_node.id)] + \
                    self.LR * (self.DF ** N) * self.getV(current_node, next_observation)
                observation = next_observation
                count += N
            return count, observation


    def act(self, observation):
        pass

    def update_memory(self, observation, action, reward, next_observation, done):
        pass

