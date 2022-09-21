from agent import PlanningAgent
import pickle

class SampleRunner:
    def __init__(self, config) -> None:
        self.env = config['envs']
        self.args = config['all_args']
        self.num_agents = config['num_agents']
        self.run_dir = config['run_dir']
        self.logs = []

        self.agent = PlanningAgent(2)

    
    def warmup(self):
        obs, done, reward, info = self.env.reset()
        if self.args.log:
            self.logs.append(info['log'])
        return obs, done, reward, info

    def run(self):
        self.env_step = 0
        obs, done, reward, info = self.warmup()

        for epi in range(self.args.episodes):
            for _ in range(self.args.max_steps):
                actions = self.agent.act(obs)
                obs, done, reward, info = self.env.step(actions)
                if self.args.log: self.logs.append(info['log'])
                self.env_step += 1
                if done or self.env_step == self.args.max_steps - 1:
                    self.env_step = 0
                    obs, done, reward, info = self.env.reset()
                    if self.args.log: self.logs.append(info['log'])
                    break
    
    def save_trajectory(self):
        path = self.run_dir + '/traj.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.env.log, f, pickle.HIGHEST_PROTOCOL)