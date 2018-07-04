from gym.envs.toy_text.taxi import TaxiEnv
from agent import MAXQAgent as Agent


class RunAgent():
    def __init__(self):
        self.env = TaxiEnv()
        self.agent = Agent(self.env.action_space)
        self.max_step = 100000
        self.render = True

    def run_maxQ(self):
        return self.agent.MAXQ(self.agent.root, self.env.reset(), self.env)


if __name__ == "__main__":
    ra = RunAgent()
    # ra.agent.log = True
    ra.agent.load_memory()
    ra.agent.RF = 0.1
    cnt = 0
    while True:
        ra.agent.LR *= 0.9999
        ra.agent.RF *= 0.999
        print(cnt, ra.run_maxQ(), ra.agent.RF)
        ra.agent.save_memory()
        cnt += 1
        # break
