from task.collect import Collect
from icecream import ic
import pickle
import gym

class TaskEnv(gym.Env):
    def __init__(self, args):
        self.args = args
        self.steps = 0
        
        
        """     self.log['task_type'] = tasktype
            self.log['num_agents'] = 2
            self.log['scene_type'] = 'kitchen'
            self.log['scene'] = scene
            self.log['layout'] = layout
            self.log['random_seed'] = random_seed """

        if args.task_type == 'collect':
            self.task = Collect(args)
        else:
            raise NotImplementedError

    def log_info(self):
        pass

    def reset(self):
        obs, info = self.task.reset(True)
        self.steps = 0
        return obs, False, None, info
    
    def step(self, actions):
        self.steps += 1
        obs, info = self.task.step(actions)
        done = self.task.is_done()
        return obs, done, None, info
    
    def render(self, mode='human'):
        pass

    def close(self):
        self.task.close()

    def seed(self, seed=None):
        if seed is None:
            self.task.seed(1)
        else:
            self.task.seed(seed)

    def _make_action_space(self):
        self.action_space = gym.spaces.Discrete(

        )

    def _make_observation_space(self):
        pass
    
if __name__ == '__main__':
    env =TaskEnv('collect', scene_type='house',log=True)
    agent = Agent(2)
    obs, done, _, _ = env.reset()
    while not done:
        actions = agent.act(obs)
        obs, done, _, _ = env.step(actions)
    path = 'C:/Users/YangYuxiang/Desktop/proj/proj/task/'+'log_with_seed_'+str(SEED)+'.pkl'
    with open(path, 'wb') as f:
        pickle.dump(env.log, f, pickle.HIGHEST_PROTOCOL)
    ic(env.steps)