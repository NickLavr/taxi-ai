from taxi import TaxiEnv
from agent import RandomAgent as Agent


class RunAgent():
    def __init__(self):
        self.env = TaxiEnv()
        self.agent = Agent(self.env.action_space)
        self.max_step = 1000
        self.render = True

    def do_episode(self):
        observation = self.env.reset()
        reward, done = 0, False
        amount = 0
        for step_i in range(self.max_step):
            if self.render:
                self.env.render()
            action = self.agent.act(observation, reward, done)
            next_observation, reward, done, _ = self.env.step(action)
            amount += 1
            if done:
                break
        return amount




ra = RunAgent()
print('total steps', ra.do_episode())