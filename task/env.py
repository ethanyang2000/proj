from collect import Collect
from agent import Agent
from icecream import ic

class TaskEnv():
    
    def __init__(self, tasktype, port=None, launch_build=True, num_agents=2, scene='2b', layout=0, random_seed=4) -> None:
        self.steps = 0
        if tasktype == 'collect':
            self.task = Collect(port=port, launch_build=launch_build, num_agents=num_agents, scene=scene,\
            layout=layout, random_seed=random_seed)
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
    env =TaskEnv('collect')
    agent = Agent(2)
    obs, done, _, _ = env.reset()
    while not done:
        actions = agent.act(obs)
        obs, done, _, _ = env.step(actions)
    ic(env.steps)