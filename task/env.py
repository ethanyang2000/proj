from collect import Collect


class TaskEnv():
    
    def __init__(self, tasktype, port=None, launch_build=True, num_agents=2, scene='1a', layout=0, random_seed=0) -> None:
        
        if tasktype == 'collect':
            self.task = Collect(port=port, launch_build=launch_build, num_agents=num_agents, scene=scene,\
            layout=layout, random_seed=random_seed)
        else:
            raise NotImplementedError

    def reset(self):
        obs = self.task.reset_task()
        return obs
    
    def step(self, actions):
        pass
