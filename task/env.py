from collect import Collect
from agent import Agent
from icecream import ic
import pickle
SEED = 7

class TaskEnv():
    
    def __init__(self, tasktype, port=None, launch_build=True, num_agents=2, scene_type='kitchen', scene='1b', layout=0, random_seed=SEED, log=False) -> None:
        self.steps = 0
        self.log = dict() if log else None
        if log:
            self.log['task_type'] = tasktype
            self.log['num_agents'] = 2
            self.log['scene_type'] = 'kitchen'
            self.log['scene'] = scene
            self.log['layout'] = layout
            self.log['random_seed'] = random_seed
        if tasktype == 'collect':
            self.task = Collect(port=port, launch_build=launch_build, num_agents=num_agents, scene=scene,\
            layout=layout, random_seed=random_seed, scene_type=scene_type, log = self.log)
        else:
            raise NotImplementedError

    def reset(self):
        obs = self.task.reset(True)
        self.steps = 0
        return obs, False, None, None
    
    def step(self, actions):
        self.steps += 1
        obs = self.task.step(actions)
        done = self.task.is_done()
        return obs, done, None, None
    
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