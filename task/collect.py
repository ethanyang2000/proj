from copy import deepcopy
from task.basic_tasks import BasicTasks
from tdw.add_ons.floorplan import Floorplan
from tdw.add_ons.proc_gen_kitchen import ProcGenKitchen
import numpy as np
from json import loads
from scipy.signal import convolve2d
from icecream import ic
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, SegmentationColors, ObiParticles
from collections import Counter
from magnebot import ActionStatus, Arm
from utils.utils import eular_yaw, any_ongoing, l2_dis
from utils.constant import constants
from PIL import Image
from tdw.librarian import ModelLibrarian


class Collect(BasicTasks):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.agent_init_pos = list()
        self.target_obj_id = dict()
        self.target_obj_cate = dict()
        
        self.goal_position = list()
        self.collision_count = [None for _ in range(self.num_agents)]
        self.steps = 0
        self.tip_flag = True
        self.lock_step = 0
        self.obs = None
    
    def _init_goal(self):
        target_list = dict()

        for o in self.om.objects_static:
            if self.om.objects_static[o].category == 'sofa':
                target_list[o] = self.om.transforms[o].position
        
        target_idx = (self._rng.choice(range(len(target_list))))
        
        """ flag = False
        
        goal_id = list(target_list.keys())[target_idx]

        goal_pos = target_list[list(target_list.keys())[target_idx]]
        goal_pos = self.om.transforms[goal_id].position
        goal_grid = pos_to_grid(goal_pos[0], goal_pos[2], self._scene_bounds)
        for xx in range(goal_grid[0]-2, goal_grid[0]+3):
            for yy in range(goal_grid[1]-2, goal_grid[1]+3):
                if self.occupancy_map[xx, yy] == 0 and not(xx == goal_grid[0] and yy == goal_grid[1]):
                    goal_grid = [xx,yy]
                    flag = True
                    break
            if flag:break

        goal_pos = grid_to_pos(goal_grid[0], goal_grid[1], self._scene_bounds) """
        
        goal_id = list(target_list.keys())[target_idx]
        goal_pos = self.om.transforms[goal_id].position

        self.goal_position.append({
            'id': goal_id,
            'pos':goal_pos
        })
        print('goal loaded')

    def _init_robot_pos(self):
        bot_pos = list()
        for p in self.agent_init_pos:
            bot_pos.append({'x':p[0], 'y':0, 'z':p[1]})
        return bot_pos

    def _init_target_objects(self):
        
        lib = ModelLibrarian(library='models_core.json')
        
        commands = list()

        # Prevent objects from spawning at edges of the occupancy map.
        convolve_map = np.zeros_like(self.occupancy_map)
        convolve_map.fill(1)
        convolve_map[self.occupancy_map == 0] = 0
        conv = np.ones((3, 3))
        convolve_map = convolve2d(convolve_map, conv, mode="same", boundary="fill")
        
        rooms = dict()

        for ix, iy in np.ndindex(self.room_map.shape):
            room_index = self.room_map[ix][iy]
            if convolve_map[ix][iy] == 0 and self.room_map[ix][iy] > -1:
                if room_index not in rooms:
                    rooms[room_index] = list()
                rooms[room_index].append((ix, iy))
        used_target_object_positions = list()

        # Add target objects to the room.
        for i in range(self.args.num_agents + self.num_agents):
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
            pos = self.get_occupancy_position(ix, iy)
            x = pos[0]
            z = pos[2]
            if i >= self.args.num_agents:
                self.agent_init_pos.append([x,z])
            else:
                obj_id = self.controller.get_unique_id()
                name = self._rng.choice(self.constants.target_objects)
                self.target_obj_id[obj_id] = name
                for l in lib.records:
                    if l.name == name:
                        self.target_obj_cate[obj_id] = l.wcategory
                        break
                comm = self.controller.get_add_object(model_name=name,
                                        position={"x": x, "y": 0, "z": z}, object_id=obj_id)

                commands.append(comm)
        
        self.controller.communicate(commands)

        print('target objects loaded')

    def reset(self, init=False):
        super().reset_scene(init)
        self.reset_data()
        self._init_goal()
        self._init_target_objects()
        self._init_robots()
        
        self.obs, info = self._parse_obs()
        return self.obs, info
    
    def reset_data(self):
        self.goal_position.clear()
        self.agent_init_pos.clear()
        self.target_obj_id.clear()
        self.target_obj_cate.clear()
        self.steps = 0

    def is_done(self):
        done_th = 5 if self.scene_type == 'house' else 1.5
        goal_pos = self.goal_position[0]['pos']
        for obj in self.target_obj_id.keys():
            obj_pos = self.om.transforms[obj].position
            dis = l2_dis(goal_pos[0], obj_pos[0], goal_pos[2], obj_pos[2])
            if dis > done_th:
                return False
        for agent in self.agents:
            for arm in [Arm.left, Arm.right]:
                if (agent.dynamic.held[arm]) > 0:
                    return False
        return True

    def _parse_obs(self, action_done = None): #TODO
        if action_done is None:
            action_done = [True] * self.num_agents
        agent_pos = [a.dynamic.transform.position for a in self.agents]
        obj_graph = [self._get_partial_objects(a) for a in range(self.num_agents)]
        action_space = self._get_action_space(obj_graph)

        obs = {
            'object_trans':self.om.transforms,
            'agent_pos': agent_pos,
            'goal_id': self.goal_position[0]['id'],
            'target_objects': list(self.target_obj_id.keys()),
            'action_done': action_done,
            'action_space': action_space,
            'object_graph': obj_graph,
            'room_map': self.room_map,
            'bound': self.nav_scene_bounds
        }
        ic(obj_graph)
        info = None
        return obs, info

    def _get_partial_objects(self, agent_id):
        obs = []
        pos = self.agents[agent_id].dynamic.transform.position
        grid = self.get_occupancy_grid(pos[0], pos[2])
        room_id = self.room_map[grid[0], grid[1]]
                
        for obj_id in self.om.transforms:
            trans = self.om.transforms[obj_id]
            pos = trans.position
            grid = self.get_occupancy_grid(pos[0], pos[2])
            obj_room = self.room_map[grid[0], grid[1]]
            
            if obj_room == room_id:
                try: name = self.om.objects_static[obj_id].name
                except: name = self.target_obj_id[obj_id]
                temp_ans = {
                    'id': obj_id,
                    'name': name,
                    'position': trans.position
                }
                obs.append(temp_ans)
        return obs

    def _get_action_space(self, obj_graph=None):

        action_space = [dict() for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            for obj in obj_graph[agent_id]:
                object_id = obj['id']
                try: cate = self.om.objects_static[object_id].category
                except: cate = self.target_obj_cate[object_id]
                action_space[agent_id][object_id] = self.actions.get_actions(cate)
            for room in self.room_center.keys():
                action_space[agent_id][room] = [0]
        return action_space

    def step(self, actions):

        ic(actions)

        self.steps += 1
        action_done = [None for _ in range(self.num_agents)]
        
        # actions.shape = (agents, action) or (batch, agents, action) if support multi-env
        # actions: go towards, pick_up, put
        def execute(actions, agent_id):
            #for a_id in range(self.num_agents):
                #if self.steps > 1:
                    #self.agents[a_id].dynamic.save_images('C:/Users/YangYuxiang/tdw_example_controller_output/demo_epi_0/'+str(self.random_seed)+'/agent_'+str(a_id))
            
            if self.lock_step > 0 and agent_id == 0:
                self.lock_step -= 1
                if self.lock_step < 5:
                    ic('lock')
                    return
            
            actions[agent_id] = self.process_actions(actions, agent_id, self.obs['action_space'])
            if actions[agent_id] is None:
                return
            if actions[agent_id][0] == 0:
                if actions[agent_id][1] is None:
                    self.agents[agent_id].move_towards([0,0,-3])
                else:
                    obj_pos = self._get_obj_pos(actions[agent_id][1])
                    agent_pos = self.agents[1-agent_id].dynamic.transform.position
                    grid = self.get_occupancy_grid(agent_pos[0], agent_pos[2])
                    self.agents[agent_id].move_towards(obj_pos, grid)
            elif actions[agent_id][0] == 1:
                self.agents[agent_id].pick_up(actions[agent_id][1])
                #self.agents[agent_id].move_to(actions[agent_id][1])
            elif actions[agent_id][0] == 2:
                self.agents[agent_id].drop(actions[agent_id][1])


        self.controller.communicate([])
        for agent_id in range(self.num_agents):
            execute(actions, agent_id)
        
        while any_ongoing(self.agents):
            self.controller.communicate([])
        
        for agent_id in range(self.num_agents):
            if actions[agent_id][0] == 1:
                if actions[agent_id][1] in self.agents[agent_id].dynamic.held[Arm.left]\
                    or actions[agent_id][1] in self.agents[agent_id].dynamic.held[Arm.right]:
                    action_done[agent_id] = True
                else:
                    action_done[agent_id] = False
            if actions[agent_id][0] == 2:
                if actions[agent_id][1] in self.agents[agent_id].dynamic.held[Arm.left]\
                    or actions[agent_id][1] in self.agents[agent_id].dynamic.held[Arm.right]:
                    action_done[agent_id] = False
                else:
                    action_done[agent_id] = True

        for agent_id in range(self.num_agents):
            if self.agents[agent_id].action.status == ActionStatus.tipping:
                if self.tip_flag:
                    self.agents[agent_id].move_by(-2)
                else:
                    self.agents[agent_id].move_by(2)
                self.tip_flag = not(self.tip_flag)
                ic('tipping')
            elif self.agents[agent_id].action.status == ActionStatus.collision:
                self.collision_process(agent_id)
            else:
                self.collision_count = [None, None]
            
        while any_ongoing(self.agents):
            self.controller.communicate([])

        for agent_id in range(self.num_agents):
            
            if actions[agent_id][0] == 1:
                for arm in [Arm.left, Arm.right]:
                    self.agents[agent_id].reset_arm(arm)
                    while any_ongoing(self.agents):
                        self.controller.communicate([])
        
        self.obs, info = self._parse_obs(action_done)
        
        return self.obs, info

    def _to_close(self):
        pos = self.agents[0].dynamic.transform.position
        pos2 = self.agents[1].dynamic.transform.position
        dist = l2_dis(pos[0], pos2[0], pos[2], pos2[2])
        if dist < 1.5:
            return True
        else:return False

    def collision_process(self, agent_id):
        if self._to_close():
            if agent_id == 0:
                ic('collision')
                self.agents[agent_id].move_by(-1.5)
                self.lock_step = 5
            return
        if self.collision_count[agent_id] is not None:
            self.agents[agent_id].move_by(0.5)
            self.collision_count[agent_id] = None
            return
        self.agents[agent_id].move_by(-0.5)
        self.collision_count[agent_id] = 0
        return

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
        if obj_id in list(self.target_obj_id.keys()):
            pos = self.om.transforms[obj_id].position
        elif obj_id == self.goal_position[0]['id']:
            pos = self.goal_position[0]['pos']
        else:
            pos = self.room_center[obj_id]

        return pos

    def _get_visiable_objects(self):
        ans = [[] for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            try:
                colors = list(set(Counter(self.agents[agent_id].dynamic.get_pil_images()["id"].getdata()))) 
                for o in self.om.objects_static:
                    segmentation_color = self.om.objects_static[o].segmentation_color
                    color = (segmentation_color[0], segmentation_color[1], segmentation_color[2])
                    if color in colors:
                        ans[agent_id].append(o)
            except:pass
        return ans
    
    def process_actions(self, actions, agent_id, action_space):
        if actions[agent_id][0] == 2:
            return actions[agent_id]
        if actions[agent_id][1] is None:
            return actions[agent_id]
        else:
            if not(actions[agent_id][1] in action_space[agent_id].keys()):
                return None
            object_id = actions[agent_id][1]
            available_actions = action_space[agent_id][object_id]
            if actions[agent_id][0] in available_actions:
                return actions[agent_id]
            else:
                return None



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


