from copy import deepcopy
from basic_tasks import BasicTasks
from tdw.add_ons.floorplan import Floorplan
from tdw.add_ons.proc_gen_kitchen import ProcGenKitchen
import numpy as np
from magnebot.paths import ROOM_MAPS_DIRECTORY, OCCUPANCY_MAPS_DIRECTORY, SPAWN_POSITIONS_PATH
from json import loads
from scipy.signal import convolve2d
from icecream import ic
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, SegmentationColors, ObiParticles
from collections import Counter
from magnebot import ActionStatus
from utils import any_ongoing

class Collect(BasicTasks):
    NUM_TARGET_OBJECTS = 3

    def __init__(self, port, launch_build, num_agents, scene, layout, local_dir='E:/tdw_lib/', random_seed=2, debug=True) -> None:
        super().__init__(port, launch_build, num_agents, scene, layout, local_dir, random_seed = random_seed, debug=debug)
        
        self._target_object_names = ['iron_box']
        self.agent_pos = list()
        self.obj_color = dict()
        self.target_obj_id = dict()
        # Set the room of the goal.
        self.goal_position = list()
        
        self.reset(True)

    
    def _init_goal(self):
        self.goal_position.clear()
        target_list = dict()
        for obj_id, trans in self.om.transforms.items():
            target_list[obj_id] = trans.position

        target_idx = (self._rng.choice(range(len(target_list))))
        self.goal_position.append({
            'id': list(target_list.keys())[target_idx],
            'pos':target_list[list(target_list.keys())[target_idx]]
        })
        print('goal loaded')

    def _init_robot_pos(self):
        bot_pos = list()
        for p in self.agent_pos:
            bot_pos.append({'x':p[0], 'y':0, 'z':p[1]})
        return bot_pos

    def objects_in_goal_position(self):
        pass

    def _init_target_objects(self):
        # Clear the list of target objects and containers.
        self.agent_pos.clear()
        self.target_obj_id.clear()

        commands = list()
        
        # Load the map of the rooms in the scene, the occupancy map, and the scene bounds.
        room_map = np.load(str(ROOM_MAPS_DIRECTORY.joinpath(f"{self.scene[0]}.npy").resolve()))
        self.occupancy_map = np.load(str(OCCUPANCY_MAPS_DIRECTORY.joinpath(f"{self.scene[0]}_{self.layout}.npy").resolve()))

        # Prevent objects from spawning at edges of the occupancy map.
        convolve_map = np.zeros_like(self.occupancy_map)
        convolve_map.fill(1)
        convolve_map[self.occupancy_map == 0] = 0
        conv = np.ones((3, 3))
        convolve_map = convolve2d(convolve_map, conv, mode="same", boundary="fill")

        # Sort all free positions on the occupancy map by room.
        rooms = dict()

        for ix, iy in np.ndindex(room_map.shape):
            room_index = room_map[ix][iy]
            if convolve_map[ix][iy] == 0:
                if room_index not in rooms:
                    rooms[room_index] = list()
                rooms[room_index].append((ix, iy))
        used_target_object_positions = list()

        # Add target objects to the room.
        for i in range(Collect.NUM_TARGET_OBJECTS + 2):
            got_position = False
            ix, iy = -1, -1
            # Get a position where there isn't a target object.
            while not got_position:
                target_room_index = self._rng.choice(np.array(list(rooms.keys())))
                target_room_positions: np.array = np.array(rooms[target_room_index])
                ix, iy = target_room_positions[self._rng.randint(0, len(target_room_positions))]
                got_position = True
                for utop in used_target_object_positions:
                    if utop[0] == ix and utop[1] == iy:
                        got_position = False
            used_target_object_positions.append((ix, iy))
            # Get the (x, z) coordinates for this position.
            x, z = self.get_occupancy_position(ix, iy)
            if i >= Collect.NUM_TARGET_OBJECTS:
                self.agent_pos.append([x,z])
            else:
                obj_id = self.controller.get_unique_id()
                name = self._rng.choice(self._target_object_names)
                self.target_obj_id[obj_id] = name
                comm = self.controller.get_add_object(model_name=name,
                                        position={"x": x, "y": 0, "z": z}, object_id=obj_id)
                commands.append(comm)

        commands.append({"$type": "send_segmentation_colors",
                       "frequency": "once"})

        resp = self.controller.communicate(commands)
        self._parse_segment_color(resp)

        print('target objects loaded')

    def reset(self, first=False):
        super().init_scene(first)
        self._init_goal()
        self._init_target_objects()
        self._init_robots(first)
        if not first:
            self._reset_addons()
    
    def _parse_segment_color(self, resp):
        self.obj_color.clear()

        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            # Get segmentation color output data.
            if r_id == "segm":
                segm = SegmentationColors(resp[i])
                break
        for j in range(segm.get_num()):
            object_id = segm.get_object_id(j)
            segmentation_color = segm.get_object_color(j)
            self.obj_color[object_id] = segmentation_color

    def _parse_obs_per_agent(self, agent_id):
        obs = {
            'id':[],
            'pos':[],
            'relative_pos':[],
            'name':[]
        }
        pil_image = self.agent_cap[agent_id].get_pil_images()[str(agent_id)]["_id"]
        colors = Counter(pil_image.getdata())
        # Get the percentage of the image occupied by each object.
        for object_id in self.obj_color:
            segmentation_color = tuple(self.obj_color[object_id])
            if segmentation_color in colors:
                obs['id'].append(object_id)
                obs['name'].append(self.om.objects_static[object_id].name)
                obs['pos'].append(self.om.transforms[object_id].position)

    def step(self, actions):
        # actions.shape = (agents, action) or (batch, agents, action) if support multi-env
        # actions: go towards, pick_up, put
        def execute(agents, actions, agent_id):
            if actions[agent_id][0] == 0:
                self.agents[agent_id].move_towards(actions[agent_id][1])
            elif actions[agent_id][0] == 1:
                self.agents[agent_id].pick_up(actions[agent_id][1])
            elif actions[agent_id][0] == 2:
                self.agents[agent_id].drop(actions[agent_id][1])

        self.controller.communicate([])
        for agent_id in range(self.num_agents):
            execute(self.agents, actions, agent_id)
        
        while any_ongoing(self.agents):
            self.controller.communicate([])
        
        for agent_id in range(self.num_agents):
            if self.agents[agent_id].action.status == ActionStatus.collision:
                self.agents[agent_id].move_by(-0.2)
        while any_ongoing(self.agents):
            self.controller.communicate([])
        
        #execute(self.agents, actions, agent_id)

    def _get_obj_pos(self, obj_id):
        return self.om.transforms[obj_id].position



if __name__ == '__main__':
    b = Collect(None, True, 2, '2a', 0)
    ids = list(b.target_obj_id.keys())
    actions = [
        [0,b._get_obj_pos(ids[1])],
        [0,b._get_obj_pos(ids[2])]
    ]
    for i in range(2000):
        b.step(actions)
    b.step([1,b._get_obj_pos(ids[2])],[1,b._get_obj_pos(ids[2])])
    b.terminate()


