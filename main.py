from gym.envs.toy_text.taxi import TaxiEnv
from agent import MAXQAgent as Agent
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class RunAgent():
    def __init__(self):
        self.env = TaxiEnv()
        self.agent = Agent(self.env.action_space)
        self.max_step = 100000
        self.render = True

    def run_maxQ(self):
        self.agent.reset()
        self.agent.MAXQ(self.agent.root, self.env.reset(), self.env)
        return self.agent.total_reward

def generate_plot(data, smoothing=10, x_label="Episodes amount", y_label="Cumulative reward", filename = "diagram"):

    fig2 = plt.figure(figsize=(10, 5))

    rewards_smoothed = pd.Series(data).rolling(smoothing, min_periods=smoothing).mean()
    plt.plot(rewards_smoothed)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(filename + ".png")
    plt.show(fig2)



if __name__ == "__main__":
    ra = RunAgent()
    # ra.agent.log = True
    ra.agent.load_memory()
    ra.agent.RF = 0.1
    episode_amount = 5000
    rewards = []
    for episode_index in range(episode_amount):
        ra.agent.LR *= 0.9999
        ra.agent.RF *= 0.999
        reward = ra.run_maxQ()
        print(episode_index, reward, ra.agent.RF)
        rewards.append(reward)
    ra.agent.save_memory()
    generate_plot(rewards)
        # break
