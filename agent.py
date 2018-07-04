from random import random, choice
import math
import pickle
from collections import defaultdict


class MAXQAgent(object):
    def __init__(self, action_space, log=False):
        self.action_space = action_space.n
        self.log = log

        self.LR = 0.5
        self.DF = 0.99
        self.RF = 0.3

        self.total_reward = 0

        # V[(Node ID, observation)] -> the expected cumulative reward of executing Node
        # starting in state observatioin until Node terminates.
        self.V = defaultdict(lambda: 0)

        # C[(Node ID, observation, Action (node ID))] -> the expected discounted cumulative
        # reward of completing subtask Node after invoking the subroutine for subtask Action in state observation
        self.C = defaultdict(lambda: 0)

        self.init_nodes()

    def reset(self):
        self.total_reward = 0

    def save_memory(self, filename='memory.pickle'):
        pickle.dump([dict(self.V), dict(self.C)], open(filename, 'wb'))

    def load_memory(self, filename='memory.pickle'):
        try:
            self.V, self.C = [defaultdict(lambda: 0, i) for i in  pickle.load(open(filename, 'rb'))]
        except Exception as e:
            print(str(e), 'in load memory')

    def getQ(self, node, observation, action):
        return self.getV(action, observation) + self.C[node.id, observation, action.id]

    def getV(self, node, observation):
        if node.is_primitive():
            return self.V[node.id, observation]
        else:
            return max([self.getQ(node, observation, node.children[child_id]) for child_id in node.children])

    @staticmethod
    def taxiDecode(i):
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
        """
        returns function which check that current position is equal to destination
        """
        locx, locy = [(0, 0), (0, 4), (4, 0), (4, 3)][destination]

        def func(observation):
            decoded = MAXQAgent.taxiDecode(observation)
            return decoded[0] == locx and decoded[1] == locy

        return func

    @staticmethod
    def check_pickup(observation):
        """
        Checks whether there was a pickup
        """
        decoded = MAXQAgent.taxiDecode(observation)
        return decoded[2] == 4

    @staticmethod
    def check_dropoff(observation):
        """
        Checks whether there was a dropoff
        """
        decoded = MAXQAgent.taxiDecode(observation)
        return decoded[2] != 4

    def check_done(observation):
        return False

    class Node:
        """
        Node is vertex in graph of subtask.
        """

        def __init__(self, agent, action, id, check_T, children=None):
            self.agent = agent
            self.id = id
            self.action = action
            self.T = check_T
            if children is None:
                self.children = dict()
            else:
                self.children = {i.id: i for i in children}

        def add_childer(self, child):
            self.children[child.id] = child

        def is_primitive(self):
            return len(self.children) == 0

        def choose_child(self, observation):
            """
            Return a child according eps-greedy policy

            :param observation: current state
            :return: One of the children (type Node)
            """
            if random() < self.agent.RF:
                return self.children[choice(list(self.children.keys()))]
            res = None
            res_Q_val = -math.inf
            for i in self.children:
                Q_val = self.agent.getQ(self, observation, self.children[i])
                if res is None or Q_val > res_Q_val:
                    res_Q_val = Q_val
                    res = self.children[i]
            return res

    def init_nodes(self):
        """
        Initializes the task graph for the Taxi problem

        """
        primitive_move = []
        MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PICKUP, DROPOFF, \
            NAVIGATE_R, NAVIGATE_G, NAVIGATE_Y, NAVIGATE_B, GET_PASS, PUT_PASS, ROOT = range(13)
        MOVES = [MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT]
        NAVIGATES = [NAVIGATE_R, NAVIGATE_G, NAVIGATE_Y, NAVIGATE_B]
        for i in range(4):
            primitive_move.append(self.Node(self, MOVES[i], MOVES[i], None))
        primitive_pickup = self.Node(self, PICKUP, PICKUP, None)
        primitive_dropoff = self.Node(self, DROPOFF, DROPOFF, None)
        navigates = [self.Node(self, -1, NAVIGATES[i], self.check_navigate(i), primitive_move) for i in range(4)]
        get = self.Node(self, -1, GET_PASS, MAXQAgent.check_pickup, [primitive_pickup] + navigates)
        put = self.Node(self, -1, PUT_PASS, MAXQAgent.check_dropoff, [primitive_dropoff] + navigates)
        self.root = self.Node(self, -1, ROOT, MAXQAgent.check_done, [get, put])


    def MAXQ(self, current_node, observation, env):
        if self.log:
            print("in node {}, observation {}".format(current_node.id, self.taxiDecode(observation)))
        if current_node.is_primitive():
            next_observation, reward, done, _ = env.step(current_node.action)
            self.total_reward += reward
            self.V[current_node.id, observation] = self.V[current_node.id, observation] * (1 - self.LR) + \
                                                   self.LR * reward
            if self.log:
                env.render()
            if done:
                return (1, "DONE")
            return (1, next_observation)
        else:
            count, reward = 0, 0
            while not current_node.T(observation):
                next_node = current_node.choose_child(observation)
                N, next_observation = self.MAXQ(next_node, observation, env)
                if next_observation == "DONE":
                    return (count + N, next_observation)
                self.C[current_node.id, observation, next_node.id] = \
                    (1 - self.LR) * self.C[current_node.id, observation, next_node.id] + \
                    self.LR * (self.DF ** N) * self.getV(current_node, next_observation)
                observation = next_observation
                count += N
            return count, observation
