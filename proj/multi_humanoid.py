from collections import Counter
from json import loads
from typing import List, Optional, Dict, Union, Tuple
from click import command
import numpy as np
from overrides import final
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.scene_data.scene_bounds import SceneBounds
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.step_physics import StepPhysics
from tdw.add_ons.floorplan import Floorplan
from humanoid import myBot
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH
from tdw.add_ons.object_manager import ObjectManager
from icecream import ic
from utils import all_moving

class HumanoidController(Controller):

    def __init__(self, port: int = 1071, launch_build: bool = True, screen_width: int = 1280, screen_height: int = 720,
                 random_seed: int = None,check_pypi_version: bool = False, num_agents = 2):

        super().__init__(port=port, launch_build=launch_build, check_version=False)
        self.occupancy_map: np.array = np.array([], dtype=int)
        # Get a random seed.
        if random_seed is None:
            random_seed = self.get_unique_id()
        """:field
        A random number generator.
        """
        self.rng: np.random.RandomState = np.random.RandomState(random_seed)
        # Set the screen and render quality.
        self.communicate([{"$type": "set_render_quality",
                           "render_quality": 5},
                          {"$type": "set_screen_size",
                           "width": screen_width,
                           "height": screen_height}])
        # Skip a set number of frames per communicate() call.
        self._check_pypi_version: bool = check_pypi_version
        # The scene bounds. This is used along with the occupancy map to get (x, z) worldspace positions.
        self._scene_bounds: Optional[SceneBounds] = None
        self.bots: Optional[myBot] = None
        self.objects: Optional[ObjectManager] = None
        self.num_agents = 2

    def _get_random(self,type):
        if type=='pos':
            return {'x':np.random.randint(-7,7),'y':0,'z':np.random.randint(-7,7)}
        else:
            return {'x':0,'y':np.random.randint(-170,170),'z':0}

    def init_scene(self) -> None:
        self.action_pool = [[] for _ in range(self.num_agents)]
        self.om = ObjectManager()
        self.bots = [myBot(id=self.get_unique_id(), position=self._get_random('pos'), rotation=self._get_random('rot')) for _ in range(self.num_agents)]
        self.objects = ['iron_box','basket_18inx18inx12iin_bamboo','afl_lamp']
        obj = {}
        for object in self.objects:
            obj[object] = self.get_unique_id()
        self.objects = obj
        obj_command = [self.get_add_object(model_name=k,object_id=v,position=self._get_random('pos'), rotation=self._get_random('rot')) for k,v in self.objects.items()]
        self.add_ons.extend(self.bots)
        self.cam = ThirdPersonCamera(avatar_id="observer",
                           position={"x": 20, "y": 15, "z": 0},
                           rotation={'x':30, 'y':-90, 'z':0})
        path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("44")
        print(f"Images will be saved to: {path}")
        avatars = ['observer']
        for j in self.bots:
            avatars.append(str(j.id)+'_cam')
        capture = ImageCapture(avatar_ids=avatars, path=path)
        self.add_ons.extend([self.cam, capture, self.om])
        commands = [TDWUtils.create_empty_room(16, 16)]
        commands.extend(obj_command)
        self.communicate(commands)
    
    def act(self, actions):
        for i in range(len(actions)):
            self.action_pool[i].append(actions[i])
        
        while True:
            flag = False
            for i in range(len(self.bots)):
                if not self.bots[i].action_status.ongoing:
                    if len(self.action_pool[i]) == 0:
                        flag = True
                        break
                    else:
                        if self.action_pool[i][0] is not None:
                            self._act(self.bots[i], self.action_pool[i][0])
                            self.action_pool[i].pop(0)
            if flag:break
            self.communicate([])
            
    def _act(self, bot, action):
        if 'navigate' in action.keys():
            bot.navigate_to(action['navigate'])
        elif 'move_by' in action.keys():
            bot.move_by(action['move_by'])
        elif 'rotate' in action.keys():
            bot.rotate_by(action['rotate'])
        elif 'pick_up' in action.keys():
            bot.pick_up(action['pick_up'])
        elif 'put' in action.keys():
            bot.put(action['put'])


    def end(self) -> None:
        """
        End the simulation. Terminate the build process.
        """

        self.communicate({"$type": "terminate"})


c = HumanoidController()
c.init_scene()
c.act([{'navigate':c.om.transforms[c.objects['iron_box']].position},{'navigate':c.om.transforms[c.objects['basket_18inx18inx12iin_bamboo']].position}])
c.act([{'pick_up':c.objects['iron_box']},{'pick_up':c.objects['basket_18inx18inx12iin_bamboo']}])
c.act([{'navigate':[0,0,0]},{'navigate':[0,0,0]}])
c.act([{'put':{'x':0,'y':0,'z':0}},{'put':{'x':0,'y':0,'z':0}}])
c.act([{'navigate':c.om.transforms[c.objects['afl_lamp']].position},None])
c.act([{'pick_up':c.objects['afl_lamp']},None])
c.act([{'navigate':[0,0,0]},None])
c.act([{'put':{'x':0,'y':0,'z':0}},None])
c.end()