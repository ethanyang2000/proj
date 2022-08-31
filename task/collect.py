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
from magnebot import ActionStatus, Arm
from utils import eular_yaw, pos_to_grid, any_ongoing, l2_dis, grid_to_pos
from constant import constants
from PIL import Image


class Collect(BasicTasks):
    NUM_TARGET_OBJECTS = 4

    def __init__(self, port, launch_build, num_agents, scene, layout, local_dir='E:/tdw_lib/', random_seed=2, debug=True) -> None:
        super().__init__(port, launch_build, num_agents, scene, layout, local_dir, random_seed = random_seed, debug=debug)
        
        self.constants = constants()
        self.agent_pos = list()
        self.target_obj_id = dict()
        # Set the room of the goal.
        self.goal_position = list()
        self.collision_mark = False
        self.steps = 0
    
    def _init_goal(self):
        self.goal_position.clear()
        target_list = dict()
        for obj_id, trans in self.om.transforms.items():
            if not(obj_id in self.target_obj_id.keys()):
                target_list[obj_id] = trans.position
        
        self.occupancy_map = np.load(str(OCCUPANCY_MAPS_DIRECTORY.joinpath(f"{self.scene[0]}_{self.layout}.npy").resolve()))

        target_idx = (self._rng.choice(range(len(target_list))))
        flag = False
        goal_id = list(target_list.keys())[target_idx]
        goal_pos = target_list[list(target_list.keys())[target_idx]]
        goal_grid = pos_to_grid(goal_pos[0], goal_pos[2], self._scene_bounds)
        for xx in range(goal_grid[0]-2, goal_grid[0]+3):
            for yy in range(goal_grid[1]-2, goal_grid[1]+3):
                if self.occupancy_map[xx, yy] == 0 and not(xx == goal_grid[0] and yy == goal_grid[1]):
                    goal_grid = [xx,yy]
                    flag = True
                    break
            if flag:break

        goal_pos = grid_to_pos(goal_grid[0], goal_grid[1], self._scene_bounds)
        self.goal_position.append({
            'id': goal_id,
            'pos':goal_pos
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
                name = self._rng.choice(self.constants.target_objects)
                self.target_obj_id[obj_id] = name
                comm = self.controller.get_add_object(model_name=name,
                                        position={"x": x, "y": 0, "z": z}, object_id=obj_id)
                
                commands.append(comm)
        self.controller.communicate(commands)
        print('target objects loaded')

    def reset(self, first=False):
        super().init_scene(first)
        self._init_goal()
        self._init_target_objects()
        self._init_robots(first)
        if not first:
            self._reset_addons()
        
        return self._parse_obs()
    
    def is_done(self):
        goal_pos = self.goal_position[0]['pos']
        for obj in self.target_obj_id.keys():
            obj_pos = self.om.transforms[obj].position
            dis = l2_dis(goal_pos[0], obj_pos[0], goal_pos[2], obj_pos[2])
            if dis > 4:
                return False
        for agent in self.agents:
            for arm in [Arm.left, Arm.right]:
                if (agent.dynamic.held[arm]) > 0:
                    return False
        return True

    def _parse_obs(self):
        agent_pos = [a.dynamic.transform.position for a in self.agents]
        act_done = [a.action.status == ActionStatus.success for a in self.agents]

        obs = {
            'object_trans':self.om.transforms,
            'agent_pos': agent_pos,
            'goal_id': self.goal_position[0]['id'],
            'target_objects': list(self.target_obj_id.keys()),
            'action_done': act_done
        }
        return obs

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
        self.steps += 1
        # actions.shape = (agents, action) or (batch, agents, action) if support multi-env
        # actions: go towards, pick_up, put
        def execute(actions, agent_id):
            for a_id in range(self.num_agents):
                if self.steps > 1:
                    self.agents[a_id].dynamic.save_images('C:/Users/YangYuxiang/tdw_example_controller_output/demo_epi_0/agent_'+str(a_id))
            """ tem_pos = self.agents[agent_id].dynamic.transform.position
            fw = self.agents[agent_id].dynamic.transform.forward
            tem_pos = [tem_pos[0], tem_pos[1], tem_pos[2], fw[0], fw[1], fw[2]]
            if self.steps > 1:
                obs = {
                    'depth': TDWUtils.get_depth_values(self.agents[agent_id].dynamic.images['depth'], \
                            width = 960, height = 960),
                    'FOV':90,
                    'camera_matrix':self.agents[agent_id].dynamic.camera_matrix,
                    'agent':tem_pos
                }
                self.agents[agent_id].bridge.dep2map(obs) """
            if actions[agent_id] is None:
                return
            if actions[agent_id][0] == 0:
                if actions[agent_id][1] is None:
                    self.agents[agent_id].move_towards([0,0,-3])
                else:
                    pos = self.agents[1-agent_id].dynamic.transform.position
                    obj_pos = self._get_obj_pos(actions[agent_id][1])
                    if not(actions[agent_id][1] in list(self.target_obj_id.keys())):
                        obj_pos = self.goal_position[0]['pos']
                    grid = pos_to_grid(pos[0], pos[2], self._scene_bounds)
                    self.agents[agent_id].move_towards(obj_pos, grid)
            elif actions[agent_id][0] == 1:
                self.agents[agent_id].pick_up(actions[agent_id][1])
            elif actions[agent_id][0] == 2:
                self.agents[agent_id].drop(actions[agent_id][1])

        self.controller.communicate([])
        for agent_id in range(self.num_agents):
            execute(actions, agent_id)
        
        while any_ongoing(self.agents):
            self.controller.communicate([])
        
        for agent_id in range(self.num_agents):
            if self.agents[agent_id].action.status == ActionStatus.tipping:
                self.agents[agent_id].move_by(-0.5)
                ic('tipping')
            elif self.agents[agent_id].action.status == ActionStatus.collision:
                self.collision_process(agent_id)
            
        while any_ongoing(self.agents):
            self.controller.communicate([])

        for agent_id in range(self.num_agents):
            
            if actions[agent_id][0] == 1:
                for arm in [Arm.left, Arm.right]:
                    self.agents[agent_id].reset_arm(arm)
                    while any_ongoing(self.agents):
                        self.controller.communicate([])
        
        return self._parse_obs()

    def _to_close(self):
        pos = self.agents[0].dynamic.transform.position
        pos2 = self.agents[1].dynamic.transform.position
        dist = l2_dis(pos[0], pos2[0], pos[2], pos2[2])
        if dist < 1.5:
            return True
        else:return False

    def collision_process(self, agent_id):
        if self._to_close():
            if not self.collision_mark:
                self.agents[agent_id].move_by(-1)
                self.collision_mark = True
            else:
                pos = self.agents[agent_id].dynamic.transform.position
                pos[2] -= 3
                self.agents[agent_id].move_to(pos)
                self.collision_mark = False
            return
        if self._check_direction(agent_id):
            ic('collision')
            float_pos = self.agents[agent_id].dynamic.transform.position
            pos = pos_to_grid(float_pos[0], float_pos[2], self._scene_bounds)
            action = self.agents[agent_id].last_direction
            ic("-----------------------------------")
            ic(action)
            try:
                if action == 0:
                    grid1 = (self.occupancy_map[pos[0]-1, pos[1]-1] == 0)
                    grid2 = (self.occupancy_map[pos[0]-1, pos[1]+1] == 0)
                    grid3 = (self.occupancy_map[pos[0], pos[1]+1] == 0)
                    grid4 = (self.occupancy_map[pos[0], pos[1]-1] == 0)
                    if grid1:
                        bias = [0, 0, -0.5]
                    elif grid2:
                        bias = [0, 0, 0.5]
                    else:
                        self.agents[agent_id].move_by(-0.2)
                        return
                elif action == 1:
                    ic('trigger')
                    #self.agents[agent_id].move_by(-0.4)
                    #return
                    grid1 = (self.occupancy_map[pos[0]+1, pos[1]-1] == 0)
                    grid2 = (self.occupancy_map[pos[0]-1, pos[1]-1] == 0)
                    grid3 = (self.occupancy_map[pos[0]+1, pos[1]] == 0)
                    grid4 = (self.occupancy_map[pos[0]-1, pos[1]] == 0)
                    if grid1:
                        bias = [0.5, 0, 0]
                    elif grid2:
                        bias = [-0.5, 0, 0]
                    else:
                        self.agents[agent_id].move_by(-0.4)
                        return
                elif action == 2:
                    grid1 = (self.occupancy_map[pos[0]+1, pos[1]+1] == 0)
                    grid2 = (self.occupancy_map[pos[0]+1, pos[1]-1] == 0)
                    grid3 = (self.occupancy_map[pos[0], pos[1]+1] == 0)
                    grid4 = (self.occupancy_map[pos[0], pos[1]-1] == 0)
                    if grid1:
                        bias = [0, 0, 0.5]
                    elif grid2:
                        bias = [0, 0, -0.5]
                    else:
                        self.agents[agent_id].move_by(-0.2)
                        return
                elif action == 3:
                    grid1 = (self.occupancy_map[pos[0]-1, pos[1]+1] == 0)
                    grid2 = (self.occupancy_map[pos[0]+1, pos[1]+1] == 0)
                    grid3 = (self.occupancy_map[pos[0]-1, pos[1]] == 0)
                    grid4 = (self.occupancy_map[pos[0]+1, pos[1]] == 0)
                    if grid1:
                        bias = [-0.5, 0, 0]
                    elif grid2:
                        bias = [0.5, 0, 0]
                    else:
                        self.agents[agent_id].move_by(-0.2)
                        return
                
                self.agents[agent_id].move_to(float_pos+bias)
            except:
                self.agents[agent_id].move_by(-0.25)
        else:
            self.agents[agent_id].move_by(-0.25)
            '''pos = self.agents[agent_id].dynamic.transform.position
            grid_pos = pos_to_grid(pos[0], pos[2], self._scene_bounds)
            angle = eular_yaw(self.agents[agent_id].dynamic.transform.rotation)
            if angle <= 45 or angle > 315: bias = [0,1]
            elif angle > 45 and angle <= 135: bias = [1,0]
            elif angle > 135 and angle <= 225: bias = [0,-1]
            elif angle > 225 and angle <= 315: bias = [-1,0]
            grid_pos[0] += bias[0]
            grid_pos[1] += bias[1]
            if self.occupancy_map[grid_pos[0], grid_pos[1]] == 0:
                ic('forward')
                self.agents[agent_id].move_by(0.25)
            else:
                ic('backward')
                self.agents[agent_id].move_by(-0.25)'''
        
    def _check_direction(self, agent_id):
        angle = eular_yaw(self.agents[agent_id].dynamic.transform.rotation)
        action = self.agents[agent_id].last_direction
        if action == 0:
            if angle <= 290 and angle >= 250:
                return True
            else:
                return False
        elif action == 1:
            if angle <= 200 and angle >= 160:
                return True
            else:
                return False
        elif action == 2:
            if angle <= 110 and angle >= 70:
                return True
            else:
                return False
        elif action == 3:
            if angle <= 20 or angle >= 340:
                return True
            else:
                return False

    def _get_obj_pos(self, obj_id):
        return self.om.transforms[obj_id].position

    def _get_visiable_objects(self):
        ans = [[] for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            colors = list(set(Counter(self.agents[agent_id].dynamic.get_pil_images()["id"].getdata())))
            for o in self.om.objects_static:
                segmentation_color = self.om.objects_static[o].segmentation_color
                color = (segmentation_color[0], segmentation_color[1], segmentation_color[2])
                if color in colors:
                    ans[agent_id].append(o)
        return ans


if __name__ == '__main__':
    b = Collect(None, True, 2, '5a', 0)
    ids = list(b.target_obj_id.keys())
    actions = [
        [0,b._get_obj_pos(ids[1])],
        [0,b._get_obj_pos(ids[2])]
    ]
    for i in range(2000):
        b.step(actions)
    b.step([1,b._get_obj_pos(ids[2])],[1,b._get_obj_pos(ids[2])])
    b.terminate()


