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



class HG(object):
    def __init__(self):
        pass

    observation_variables_amount = 6
    # taxi x, taxi y, passx, passy, pass in taxi, destination_index
    @staticmethod
    def taxiDecode(i):
        if (i == -1):
            return [-1] * HG.observation_variables_amount
        destination = i % 4 # destination
        i = i // 4
        pl = i % 5
        # out.append(i % 5) # passloc [0..3] location or 4 in taxi
        i = i // 5
        taxiy = i % 5 # taxi y
        i = i // 5
        taxix = i # taxi x
        assert 0 <= i < 5
        px, py = 0, 0
        in_taxi = 1 if pl == 4 else 0
        if pl == 4:
            px, py = taxix, taxiy
        else:
            px, py = [(0, 0), (0, 4), (4, 0), (4, 3)][pl]
        return [taxix, taxiy, px, py, in_taxi, destination]

    @staticmethod
    def decode_trajectory(trajectory):
        decode_item  = lambda x: [x[0], x[1], HG.taxiDecode(x[2])]
        return list(map(decode_item, trajectory))


    @staticmethod
    def build_CAT(env, trajectories_amount=10):
        trajectories = [HG.decode_trajectory(HG.relax_trajectory(HG.get_random_trajectory(env))) for _ in range(trajectories_amount)]

        return trajectories

    @staticmethod
    def get_random_trajectory(env, step_limit=10000):
        observation = env.reset()
        trajectory = [[None, None, observation]]

        for _ in range(step_limit):
            action = env.action_space.sample()
            next_observation, reward, done, _ = env.step(action)
            # trajectory.append([action, reward, next_observation if not done else -1])
            trajectory.append([action, reward, next_observation])
            if done:
                return trajectory
        return trajectory

    @staticmethod
    def relax_trajectory(trajectory):

        # observation -> index
        observations = dict()

        for index, (action, reward, observation) in enumerate(trajectory):
            if observation not in observations:
                observations[observation] = index
            else:
                for i in range(observations[observation] + 1, index):
                    if trajectory[i] is not None:
                        observations.__delitem__(trajectory[i][2])
                        trajectory[i] = None
                trajectory[index] = None
        return [i for i in trajectory if i is not None]


    @staticmethod
    def get_goal_condition(cats):
        raise NotImplementedError

    @staticmethod
    def HierGen(models, cats):
        g = HG.get_goal_condition(cats)
        many_actions = False
        for i in cats:
            if len(cats) > 1:
                many_actions = True
        if many_actions:
            tasks = HG.HierBuilder(models, cats)
            if len(tasks) > 0:
                raise NotImplementedError
                return
            actions = ExtractUltimateActions(cats)
            taskQ = HG.HierBuilder(models, cats.extract_addition(actions))
            if len(taskQ) > 0:
                task = HG.HierGen(models, cats.extract(actions))
                task.add_child(taskQ)
                return task
        return task(g.variables, g, cats.actions)

    
    @staticmethod
    def ExtractUltimateActions(cats):
        raise NotImplementedError


    class CAT_trajectory:
        start_action = -1
        end_action = -2
        #observation:  taxi x, taxi y, passx, passy, pass in taxi, destination_index
        #actions: up, down, left, right, pickup, dropoff, ..., end action, start action
        checked_variables = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1, 2, 3, 4], [0, 1, 4, 5], [0, 1, 2, 3, 4, 5], []]


        def __init__(self, list_trajectory):
            trajectory = []

            trajectory.append(self.tr_node([None] * HG.observation_variables_amount, self.start_action, 0))
            for i in range(len(list_trajectory) - 1):
                trajectory.append(self.tr_node(list_trajectory[i][2], list_trajectory[i + 1][0], list_trajectory[i + 1][1]))
            trajectory.append(self.tr_node(list_trajectory[-1][2], self.end_action, 0))

            for variable in range(HG.observation_variables_amount):
                tail = 0
                for head in range(1, len(trajectory) - 1):
                    if variable in self.checked_variables[trajectory[head].action] :
                        trajectory[head].incoming[variable] = tail
                        trajectory[tail].outgoing[variable].add(head)

                    if trajectory[head].observation[variable] != trajectory[head + 1].observation[variable]:
                        tail = head
                if True:
                    trajectory[-1].incoming[variable] = tail;
                    trajectory[tail].outgoing[variable].add(len(trajectory) - 1);


            self.trajectory = trajectory


        class tr_node:
            def __init__(self, observation, action, reward):
                self.observation = observation
                self.action = action
                self.reward = reward
                # arc arrows
                self.incoming = [None] * HG.observation_variables_amount
                self.outgoing = [set() for _ in range(HG.observation_variables_amount)]

            def __repr__(self):
                return "tr_node(obs: {}, action: {}, reward: {}. Arcs: incoming {}; outgoin {})".format(self.observation, self.action, self.reward, self.incoming, self.outgoing)

            def get_all_outgoing(self):
                res = set()
                for i in self.outgoing:
                    res = res.union(i)
                return res

        def __repr__(self):
            return "trajectory: " + '\n'.join([i.__repr__() for i in self.trajectory])

    
    @staticmethod
    def HierBuilder(models, cats):
        g = HG.get_goal_condition(cats)
        if g.all_false():
            return []]
        subscats = []
        for var in g.variables:
            subscats.append(Cat_scan(cats, [var]))
        unfied_subcats = HG.unify(subscats)
        result = []
        if len(unfied_subcats) > 0:
            subscats = unfied_subcats
            for subcat in subscats:
                taskQ = HG.HierBuilder(models, cats.extract_addition(subcat))
                if len(taskQ) > 0:
                    task = HG.HierGen(models, cats.extract(subcat))
                    task.add_child(taskQ)
                    result.append(task)
                    subscats.remove(subcat)
        if len(subcats) > 0:
            merged = HG.merge_subcats(subcats)
            if len(merged) == 0:
                return []
            taskQ = HG.HierBuilder(model, cats.extract_addition(merged))
            if len(task) == 0:
                return []
            task = HG.HierGen(model, cats.extract(merged))
            task.add_child(taskq)
            result.append(task)
        return result



    @staticmethod
    def merge_subcats(subcats):
        """
        Merge unsuccessful sets of sub-CATs into one set
        """
        raise NotImplementedError
        
    @staticmethod
    def unify(subscats):
        """Unify the partition of goal variables across trajectories"""
        raise NotImplementedError
    
    @staticmethod
    def CAT_scan(cats, variables):
        ans = []
        for cat in cats:
            phi = set()
            for var in variables:
                for i, node in enumerate(cat.trajectory):
                    if node.outgoing[var] == set(len(cat.trajectory) - 1):
                        phi.add(i)

            for i in reversed(range(len(cat.trajectory))):
                if cat.trajectory[i].get_all_outgoing().issubset(phi):
                    phi.add(i)

            phi = sorted(list(phi))
            for var in variables:
                last_incoming = None
                for i in range(len(phi)):
                    incoming_index = cat.trajectory[phi[i]].incoming[var]
                    if incoming_index is not None and incoming_index not in phi:
                        last_incoming = i
                phi = phi[last_incoming:]

            ans.append(phi)
        return ans

    

if __name__ == "__main__":
    from taxi import TaxiEnv
    agent = HG()
    env = TaxiEnv()
    tr = agent.build_CAT(env)
    print(*tr, sep = '\n')
    ct = HG.CAT_trajectory(tr[0])