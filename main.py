from taxi import TaxiEnv
from agent import MAXQAgent as Agent
# from agent import QLearningAgent as Agent
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

    def run_maxQ(self):
        return self.agent.MAXQ(self.agent.root, self.env.reset(), self.env)



ra = RunAgent()
# ra.agent.log = True
ra.agent.load_memory()
# ra.agent.RF = 0.1
cnt = 0
try:
    while True:
        ra.agent.LR *= 0.9999
        # ra.agent.RF *= 0.999
        print(cnt, ra.run_maxQ(), ra.agent.RF)
        ra.agent.save_memory()
        cnt += 1
        break

except Exception as e:
    print('Terminate')
    ra.agent.save_memory()