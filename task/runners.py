from agent import PlanningAgent
import pickle

class SampleRunner:
    def __init__(self, config) -> None:
        self.env = config['envs']
        self.args = config['all_args']
        self.num_agents = config['num_agents']
        self.run_dir = config['run_dir']

        self.agent = PlanningAgent(2)

    
    def warmup(self):
        obs, done, _, _ = self.env.reset()
        return obs, done, _, _

    def run(self):
        obs, done, _, _ = self.warmup()

        for epi in range(self.args.episodes):
            for step in range(self.args.max_steps):
                actions = self.agent.act(obs)
                obs, done, _, _ = self.env.step(actions)
                if done:
                    obs, done,_,_ = self.env.reset()
                    self.save_trajectory()
                    break
    
    def save_trajectory(self):
        path = self.run_dir + '/traj.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.env.log, f, pickle.HIGHEST_PROTOCOL)