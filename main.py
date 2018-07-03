from taxi import TaxiEnv
from agent import QLearningAgent as Agent
# from agent import RandomAgent as Agent

class RunAgent():
    def __init__(self):
        self.env = TaxiEnv()
        self.agent = Agent(self.env.action_space)
        self.max_step = 100000
        self.render = True

    def do_episode(self):
        observation = self.env.reset()
        reward, done = 0, False
        amount = 0
        for step_i in range(self.max_step):
            if self.render:
                self.env.render()
            action = self.agent.act(observation)
            next_observation, reward, done, _ = self.env.step(action)
            self.agent.update_memory(observation, action, reward, next_observation, done)
            amount += 1
            if reward > 0:
                print(reward)
            if done:
                break
        return amount




ra = RunAgent()
ra.render = False
i = 0
while True:
    print('Epoh {}; total steps: {}; LR {}; RF {}'.format(i, ra.do_episode(), ra.agent.LR, ra.agent.RF))
    ra.agent.RF *= 0.999
    ra.agent.LR *= 0.999
    i += 1