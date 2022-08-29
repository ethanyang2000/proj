from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from magnebot import Magnebot, ActionStatus
from tdw.add_ons.image_capture import ImageCapture
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH
from tdw.add_ons.floorplan import Floorplan
from tdw.add_ons.proc_gen_kitchen import ProcGenKitchen
from tdw.add_ons.object_manager import ObjectManager
from tdw.scene_data.scene_bounds import SceneBounds
from tdw.add_ons.step_physics import StepPhysics
import numpy as np
from pathlib import Path
import os
from icecream import ic
from bot import Bot

MAGNEBOT_RADIUS: float = 0.22
OCCUPANCY_CELL_SIZE: float = (MAGNEBOT_RADIUS * 2) + 0.05

class BasicTasks():
    def __init__(self, port, launch_build, num_agents, scene, layout, local_dir, random_seed, debug) -> None:
        if local_dir is not None:
            TDWUtils.set_default_libraries(scene_library=local_dir+"scenes.json",
                                                    model_library=local_dir+'models.json')
        
        # scene: init_scene
        # object manager & robots: reset()
        # camera & capture: initialized False

        self.num_agents = num_agents
        self._debug = debug
        self.scene = scene
        self.layout = layout

        self.agents = list()

        if random_seed is not None:
            self._rng = np.random.RandomState(random_seed)

        self.controller = Controller()
        self.scene_instance = Floorplan()

        self.scene_instance.init_scene(scene, layout)
        self.om = ObjectManager()
        self._step_physics: StepPhysics = StepPhysics(10)

        self.camera = ThirdPersonCamera(position={"x": 0, "y": 40, "z": 0},
                                look_at={"x": 0, "y": 0, "z": 0},
                                avatar_id="bird_view")
        self.capture_path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath('demo_epi_0')
        self.capture = ImageCapture(avatar_ids=['bird_view'], path=self.capture_path)
        self.controller.add_ons.extend([self.scene_instance, self.camera, self.capture, self.om, self._step_physics])
        self.controller.communicate([{"$type": "set_screen_size",
                "width": 1280,
                "height": 720}])
        
    def _init_robot_pos(self):
        pass


    def terminate(self):
        self.controller.communicate([])
        self.controller.communicate({"$type": "terminate"})
    
    def _init_robots(self, first):
        init_pos = self._init_robot_pos()
        for i in range(self.num_agents):
            if first:
                bot = Bot(position=init_pos[i],
                            robot_id=self.controller.get_unique_id(), bound=self._scene_bounds, map=self.occupancy_map)
                self.agents.append(bot)
                self.controller.add_ons.append(bot)
            else:
                self.agents[i].reset(init_pos[i])
            self.controller.communicate([])
            self.agents[i].collision_detection.exclude_objects.extend(list(self.target_obj_id.keys()))
        '''self.controller.communicate([{"$type": "set_field_of_view",
                         "field_of_view": 90}])'''
        print('robots loaded')

    def init_scene(self, first):
          self._init_floorplan(self.scene, self.layout, first)
    
    def _init_target_objects(self):
        pass

    def _init_floorplan(self, scene, layout, first):
        if not first:
            self.scene_instance.init_scene(scene=scene, layout=layout)
            self.controller.communicate(self.scene_instance.commands)
        else:
            resp = self.controller.communicate([{"$type": "set_floorplan_roof",
                    "show": False},{"$type": "send_scene_regions"},{"$type": "bake_nav_mesh"}])
        
            self._scene_bounds = SceneBounds(resp)
            commands = []
            for obj in self.om.transforms.keys():
                commands.append({"$type": "make_nav_mesh_obstacle",
                "id": obj,
                "carve_type": "all",
                "scale": 0.5,
                "shape": "box"})
                commands.append({"$type": "bake_nav_mesh"})
            self.controller.communicate(commands)
        print('scene loaded')

    def _init_kitchen(self):
        kitchen = ProcGenKitchen()
        kitchen.create()
        self.controller.add_ons.append(kitchen)
        self.controller.add_ons.extend([self.camera, self.capture])
    
    def get_occupancy_position(self, i: int, j: int):
        x = self._scene_bounds.x_min + (i * OCCUPANCY_CELL_SIZE)
        z = self._scene_bounds.z_min + (j * OCCUPANCY_CELL_SIZE)
        return x, z

    def reset(self, first):
        pass

    def _reset_addons(self):
        self.camera.initialized = False
        self.capture.initialized = False
        path = EXAMPLE_CONTROLLER_OUTPUT_PATH
        file_num = len(os.listdir(path))
        path = path.joinpath('demo_epi_'+str(file_num))
        self._reset_capture_path(path)
    
    def _reset_capture_path(self, path):
        if isinstance(path, str):
            self.capture.path: Path = Path(path)
        else:
            self.capture.path: Path = path
        if not self.capture.path.exists():
            self.capture.path.mkdir(parents=True)
