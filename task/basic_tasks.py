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
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.add_ons.proc_gen_kitchen import ProcGenKitchen
from magnebot.paths import ROOM_MAPS_DIRECTORY, OCCUPANCY_MAPS_DIRECTORY, SPAWN_POSITIONS_PATH
import numpy as np
from pathlib import Path
import os
from icecream import ic
from bot import Bot
from constant import available_actions
from constant import scene_const
from utils import grid_to_pos

MAGNEBOT_RADIUS: float = 0.22
OCCUPANCY_CELL_SIZE: float = (MAGNEBOT_RADIUS * 2) + 0.05

class BasicTasks():
    def __init__(self, port, launch_build, num_agents, scene_type, scene, layout, local_dir, random_seed, debug) -> None:
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
        self.scene_type = scene_type

        self.agents = list()
        self.actions = available_actions()

        self._rng = np.random.RandomState(random_seed)

        self.controller = Controller()


        if scene_type == 'kitchen':
            self.scene_instance = ProcGenKitchen()
            self.scene_instance.create(scene='mm_kitchen_2b')
        elif scene_type == 'house':
            self.scene_instance = Floorplan()
            self.scene_instance.init_scene(scene, layout)
        
        self.om = ObjectManager()
        self.map_manager = OccupancyMap(cell_size=0.49)
        self._step_physics: StepPhysics = StepPhysics(10)
        
        camera_height = 20 if scene_type == 'kitchen' else 40

        self.camera = ThirdPersonCamera(position={"x": 0, "y": camera_height, "z": 0},
                                look_at={"x": 0, "y": 0, "z": 0},
                                avatar_id="bird_view")
        
        self.capture_path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath(str(random_seed)+'/demo_epi_0')
        self.capture = ImageCapture(avatar_ids=['bird_view'], path=self.capture_path)
        self.controller.add_ons.extend([self.scene_instance, self.camera, self.capture, self.om, self._step_physics, self.map_manager])
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
                            robot_id=self.controller.get_unique_id(), bound=self._scene_bounds, map=self.occupancy_map, rng=self._rng)
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
        if self.scene_type == 'kitchen':
            self.map_manager.generate()
            self.controller.communicate([])
            self.occupancy_map = self.map_manager.occupancy_map
            self.room_map = np.zeros_like(self.occupancy_map)

        elif self.scene_type == 'house':
            self.occupancy_map = np.load(str(OCCUPANCY_MAPS_DIRECTORY.joinpath(f"{self.scene[0]}_{self.layout}.npy").resolve()))
            
            self.occupancy_map[15:20,18:20] = 0
            self.occupancy_map[11,-2] = -1
            self.occupancy_map[23,-8] = -1
            self.occupancy_map[31,-10] = -1
            self.occupancy_map[35,12] = -1
            self.occupancy_map[34,13] = -1
            
            self._init_floorplan(self.scene, self.layout, first)
            self.room_map = np.load(str(ROOM_MAPS_DIRECTORY.joinpath(f"{self.scene[0]}.npy").resolve()))
            self.room_map[self.occupancy_map==-1] = -1
        
        del_ids = []
        for o in self.om.objects_static:
            if self.om.objects_static[o].category == 'coffee table, cocktail table':
                del_ids.append(o)
        command = [{"$type": "destroy_object",
                        "id": oid} for oid in del_ids]
        command.extend([{"$type": "set_floorplan_roof",
                    "show": False},{"$type": "send_scene_regions"}])
        resp = self.controller.communicate(command)
        self._scene_bounds = SceneBounds(resp)
        
        if self.scene_type == 'house':
            self.room_center = scene_const().room_center[self.scene]
            for key, value in self.room_center.items():
                self.room_center[key] = grid_to_pos(value[0], value[1], self._scene_bounds)
            """ room_num = np.max(self.room_map)
            self.room_center = {}
            for ix, iy in np.ndindex(self.room_map.shape):
                if self.occupancy_map[ix, iy] == 0:
                    room_id = self.room_map[ix, iy]
                    if not(room_id) in self.room_center.keys():
                        pos = grid_to_pos(ix, iy, self._scene_bounds)
                        self.room_center[room_id] = pos """
            """ for i in range(0,room_num+1):
                bounds = np.where(self.room_map==i)[0]
                xx = bounds[0]
                yy = bounds[1]
                xx_min = np.min(xx)
                xx_max = np.max(xx)
                yy_min = np.min(yy)
                yy_max = np.max(yy)
                x = round((xx_max+xx_min)/2)
                y = round((yy_max+yy_min)/2)
                self.room_center[i] = grid_to_pos(x,y,self._scene_bounds) """
            
    def _init_target_objects(self):
        pass

    def _init_floorplan(self, scene, layout, first):
        if not first:
            self.scene_instance.init_scene(scene=scene, layout=layout)
            self.controller.communicate(self.scene_instance.commands)
        print('scene loaded')
    
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
