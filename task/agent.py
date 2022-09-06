from re import A
from icecream import ic
from utils import pos_to_grid
from utils import l2_dis
import numpy as np
import sys

class Agent:
    def __init__(self, num_agents) -> None:
        self.mode = [None for _ in range(num_agents)]
        self.num_agents = num_agents
        self.objects_in_hand = [0 for _ in range(self.num_agents)]
        self.obj_id_in_hand = [[] for _ in range(self.num_agents)]
        self.done_object = []
        self.nav_id = [None, None]

    def _arrived(self, trans, obj_id, agent_pos, transport=False):
        th = 3 if transport else 1
        pos = trans[obj_id].position
        dist = l2_dis(pos[0], agent_pos[0], pos[2], agent_pos[2])
        if dist < th:
            return True

    def act(self, obs):
        trans = obs['object_trans']
        agent_pos = obs['agent_pos']
        action_done = obs['action_done']
        trans_id = obs['goal_id']
        action_space = obs['action_space']
        room_map = obs['room_map']
        bound = obs['bound']

        for agent_id in range(self.num_agents):
            if self.mode[agent_id] == 'navigation':
                if self._arrived(trans, self.nav_target[agent_id], agent_pos[agent_id]):
                    self.mode[agent_id] = 'pickup'
            
            elif self.mode[agent_id] == 'transport':
                if self._arrived(trans, self.nav_target[agent_id], agent_pos[agent_id], True):
                    self.mode[agent_id] = 'drop'
            
            elif self.mode[agent_id] == 'pickup':
                if action_done:
                    self.mode[agent_id] = None
                    self.objects_in_hand[agent_id] += 1
                    self.obj_id_in_hand[agent_id].append(self.nav_target[agent_id])
            
            elif self.mode[agent_id] == 'drop':
                if action_done:
                    self.objects_in_hand[agent_id] -= 1
                    self.obj_id_in_hand[agent_id].pop()
                    if self.objects_in_hand[agent_id] == 0:
                        self.mode[agent_id] = None

        if None in self.mode:
            mask = [(m==None) for m in self.mode]
            new_mode, new_nav = self.check_mode(obs, mask)
            for idx in range(len(new_mode)):
                if mask[idx] == 1:
                    self.mode[idx] = new_mode[idx]
                    self.nav_id[idx] = new_nav[idx]
        
        actions = [[] for _ in range(self.num_agents)]
        
        for agent_id in range(self.num_agents):
            if self.mode[agent_id] == 'navigation':
                actions[agent_id] = [0, self.nav_id[agent_id]]
                self.nav_target = self.nav_id
            elif self.mode[agent_id] == 'transport':
                actions[agent_id] = [0, trans_id]
                self.nav_target[agent_id] = trans_id
            elif self.mode[agent_id] == 'done':
                actions[agent_id] = [0, None]
            elif self.mode[agent_id] == 'pickup':
                actions[agent_id] = [1, self.nav_target[agent_id]]
            elif self.mode[agent_id] == 'drop':
                actions[agent_id] = [2, self.obj_id_in_hand[agent_id][-1]]

            if actions[agent_id][0] == 0 and actions[agent_id][1] is not None:
                actions[agent_id][1] = self._parse_navigation(action_space[agent_id], actions[agent_id][1],
                trans, room_map, bound)

        ic(actions)
        ic(self.mode)
        return actions

    def check_mode(self, obs, mask):
        obj = obs['target_objects']
        trans = obs['object_trans']
        agent_pos = obs['agent_pos']
        target_obj = []
        for ob in obj:
            if ob not in self.done_object:
                target_obj.append(ob)

        ans = [None, None]
        self.mask = mask
        navigate_ids = [None, None]

        for i in range(self.num_agents):
            if mask[i] == 0:
                ans[i] = None
            elif self.objects_in_hand[i] == 2:
                ans[i] = 'transport'
        
        counts = 0
        for i in range(self.num_agents):
            if mask[i] == 0 or ans[i] is not None:
                counts += 1
        if counts == 2: return ans, [None, None]

        if len(target_obj) == 0:
            if np.sum(self.objects_in_hand) == 0:
                return ['done', 'done'], [None, None]
            else:
                for i in range(self.num_agents):
                    if self.objects_in_hand[i] > 0:
                        ans[i] = 'transport'
                    else:
                        ans[i] = 'done'
                return ans, [None, None]

        if len(target_obj) == 1:
            pair = self._get_nearest_pair(target_obj, trans, agent_pos)
            idx = pair.index(None)
            ans[1-idx] = 'navigation'
            self.done_object.append(pair[1-idx])
            navigate_ids[1-idx] = pair[1-idx]
            if mask[idx] == 0:
                pass
            else:
                if self.objects_in_hand[idx] == 0:
                    ans[idx] = 'done'
                else:
                    ans[idx] = 'transport'
            return ans, navigate_ids
        

        pairs = self._get_nearest_pair(target_obj, trans, agent_pos)
        for i in range(self.num_agents):
            if pairs[i] is not None:
                ans[i] = 'navigation'
                navigate_ids[i] = pairs[i]
                self.done_object.append(pairs[i])
            elif self.objects_in_hand[i] == 2:
                ans[i] = 'transport'
            elif self.objects_in_hand[i] == 0:
                ans[i] = 'done'
        return ans, navigate_ids

    def _nearest_object(self, objects, trans, agent_pos):
        min_dis = np.inf
        min_id = [None, None]
        masks = np.ones(len(objects))
        cost = 0

        if len(objects) == 1:
            pos = trans[objects[0]].position
            for agent_id in range(len(agent_pos)):
                dis = l2_dis(pos[0], agent_pos[agent_id][0], pos[2], agent_pos[agent_id][2])
                if dis < min_dis:
                    min_dis = dis
                    min_id[agent_id] = objects[0]
                    min_id[1-agent_id] = None
            return min_id

        if len(agent_pos) == 1:
            for obj in objects:
                pos = trans[obj].position
                dis = l2_dis(pos[0], agent_pos[0][0], pos[2], agent_pos[0][2])
                if dis < min_dis:
                    min_dis = dis
                    min_id[0] = obj
            return min_id[0]


        for i, obj1 in enumerate(objects):
            pos = trans[obj1].position
            dis1 = l2_dis(pos[0], agent_pos[0][0], pos[2], agent_pos[0][2])
            cost += dis1
            masks[i] = 0
            for j, obj2 in enumerate(objects):
                if masks[j] == 0:
                    continue
                pos = trans[obj2].position
                dis2 = l2_dis(pos[0], agent_pos[1][0], pos[2], agent_pos[1][2])
                cost += dis2

                if cost < min_dis:
                    min_dis = cost
                    min_id = [obj1, obj2]
                cost -= dis2
            cost -= dis1
            masks[i] = 1
        return min_id

    def _available(self, agent_id):
        if self.mask[agent_id] == 1 and self.objects_in_hand[agent_id] < 2:
            return True
        else:
            return False

    def _get_nearest_pair(self, objects, trans, agent_pos):
        obj_num = len(objects)
        if obj_num == 1:
            if not(self._available(0)) and self._available(1):
                target_id = [None, objects[0]]
            elif not(self._available(1)) and self._available(0):
                target_id = [objects[0], None]
            elif self._available(0) and self._available(1):
                target_id = self._nearest_object(objects, trans, agent_pos)
        else:
            if self._available(0) and self._available(1):
                target_id = self._nearest_object(objects, trans, agent_pos)
            elif self._available(0):
                target_id = self._nearest_object(objects, trans, [agent_pos[0]])
                target_id = [target_id, None]
            elif self._available(1):
                target_id = self._nearest_object(objects, trans, [agent_pos[1]])
                target_id = [None, target_id]

        return target_id

    def _parse_navigation(self, action_space, object_id, trans, room_map, bound):
        if object_id in action_space.keys():
            return object_id
        pos = trans[object_id].position
        grid = pos_to_grid(pos[0], pos[2], bound)
        room_id = room_map[grid[0], grid[1]]
        return room_id

        
