from copy import deepcopy
from itertools import count
from os import environ
from json import loads
from csv import DictReader
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from scipy.signal import convolve2d
from tdw.librarian import ModelLibrarian
from tdw.tdw_utils import TDWUtils
from tdw.controller import Controller
from magnebot import Magnebot, Arm, ActionStatus, ArmJoint
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from magnebot import Magnebot, ActionStatus
from tdw.tdw_utils import TDWUtils
from icecream import ic
from utils import l2_dis, eular_yaw, a_star_search, pos_to_grid, grid_to_pos
from tdw.output_data import OutputData, NavMeshPath
import math
from bridge import bridge


FORWARD_DIS = 1.5
ARRIVE_AT = 0.2
TURN_ANGLE = 90
class Bot(Magnebot):
    def __init__(self, position, robot_id, debug=True, bound=None, map=None):
        super().__init__(position=position, robot_id=robot_id, check_version=False)

        self._debug = debug
        
        self.navigation_plan = list()
        self.receive_plan = False
        self.navigation_done = False
        self.start_point = None
        self.store_bound(bound)
        self.map = map
        self.previous_nav = None
        self.last_direction = 3
        self.bridge = bridge()

    def _pick_up(self, target: int, arm: Arm):
        if target in self.dynamic.held[arm]:
            if self._debug:
                print(f"Already holding {target}")
            self.action.status = ActionStatus.success
        
        elif len(self.dynamic.held[arm]) > 0:
            if self._debug:
                print(f"Already holding an object in {arm.name}")
            self.action.status = ActionStatus.failed_to_grasp

        else:
            self._grasp(target=target, arm=arm)

    def move_forward(self):
        self.move_by(FORWARD_DIS, ARRIVE_AT)
    
    def turn_left(self):
        self.turn_by(-TURN_ANGLE)

    def turn_right(self):
        self.turn_by(TURN_ANGLE)

    def move_towards(self, target_pos, grid=None):
        map = deepcopy(self.map)
        if grid is not None:
            map[grid[0], grid[1]] = 1
        target = pos_to_grid(target_pos[0], target_pos[2], self.bound)
        start = pos_to_grid(self.dynamic.transform.position[0], self.dynamic.transform.position[2], self.bound)
        #ic(target, start)
        #ic(self.map.shape)
        actions, counts = a_star_search(map, start, target, self.dynamic.transform.position)
        ic(actions, counts)
        if actions == None:
            return

        angle = int(eular_yaw(self.dynamic.transform.rotation))

        if angle < 0:
            angle += 360

        if actions == 0:
            if angle >= 260 and angle <= 280:
                pos = grid_to_pos(start[0]-counts, start[1], self.bound)
                '''if self.map[start[0]-counts-1, start[1]] == 1:
                    pos[0] += 0.2'''
                self.move_to(pos)
            else:
                if angle > 280 or angle <= 45:
                    if angle <= 45:
                        self.turn_by(270-360-angle)
                    else:
                        self.turn_by(270-angle)
                elif angle > 45 and angle <= 90:
                    self.turn_by(0-angle)
                elif angle > 90 and angle <= 135:
                    self.turn_by(180-angle)
                elif angle > 135 and angle <= 260:
                    self.turn_by(270-angle)
        elif actions == 1:
            if angle >= 170 and angle <= 190:
                pos = grid_to_pos(start[0], start[1]-counts, self.bound)
                '''if self.map[start[0], start[1]-counts-1] == 1:
                    pos[2] += 0.2'''
                self.move_to(pos)
            else:
                if angle > 190 and angle <= 315:
                    self.turn_by(180-angle)
                elif angle > 315:
                    self.turn_by(270-angle)
                elif angle <= 45:
                    self.turn_by(90-angle)
                elif angle > 45 and angle <= 170:
                    ic()
                    self.turn_by(180-angle)
        elif actions == 2:
            if angle >= 80 and angle <= 100:
                pos = grid_to_pos(start[0]+counts, start[1], self.bound)
                '''if self.map[start[0]+counts+1, start[1]] == 1:
                    pos[0] -= 0.2'''
                self.move_to(pos)
            else:
                if angle > 100 and angle <= 225:
                    self.turn_by(90-angle)
                elif angle > 225 and angle <= 270:
                    self.turn_by(180-angle)
                elif angle > 270 and angle <= 315:
                    self.turn_by(360-angle)
                elif angle > 315 or angle <= 80:
                    if angle > 315:
                        self.turn_by(80+360-angle)
                    else:
                        self.turn_by(90-angle)
        elif actions == 3:
            if angle <= 10 or angle > 350:
                pos = grid_to_pos(start[0], start[1]+counts, self.bound)
                '''if self.map[start[0], start[1]+counts+1] == 1:
                    pos[2] -= 0.2'''
                self.move_to(pos)
            else:
                if angle > 10 and angle <= 135:
                    self.turn_by(0-angle)
                elif angle > 135 and angle <= 180:
                    self.turn_by(90-angle)
                elif angle > 180 and angle <= 225:
                    self.turn_by(270-angle)
                elif angle > 225 and angle <= 350:
                    self.turn_by(360-angle)
        
        self.last_direction = actions 
    
    """ def move_towards(self, pos, grid=None):
        action = self.bridge.nav([pos[0], pos[2]])
        if action == 0:
            self.move_by(0.5)
        elif action == 1:
            self.turn_by(15)
        elif action == 2:
            self.turn_by(-15) """

    def store_bound(self, bound):
        self.bound = bound

    def pick_up(self, target):
        for arm in [Arm.left, Arm.right]:
            self._pick_up(target, arm)
            if self._check_action_status():
                break

    def drop(self, target: int):
        for arm in [Arm.left, Arm.right]:
            self._drop(target, arm)
            if self._check_action_status():
                break

    def reset_arm(self, arm: Arm):
        status = super().reset_arm(arm=arm)
        return status

    def _drop(self, target: int, arm: Arm):
        if target in self.dynamic.held[arm]:
            super().drop(target=target, arm=arm)
            
        else:
            self.action.status = ActionStatus.not_holding
        
    def turn_by(self, angle: float, aligned_at: float = 3):
        
        super().turn_by(angle=angle, aligned_at=aligned_at)

    def turn_to(self, target: Union[int, Dict[str, float]], aligned_at: float = 3):
        
        super().turn_to(target=target, aligned_at=aligned_at)

    def move_by(self, distance: float, arrived_at: float = 0.3):
        
        super().move_by(distance=distance, arrived_at=arrived_at)

    def _grasp(self, target: int, arm: Arm):
        
        super().grasp(target=target, arm=arm)

    def reset_position(self):
        
        super().reset_position()

    def _get_reset_arm_commands(self, arm: Arm, reset_torso: bool) -> List[dict]:
        return super()._get_reset_arm_commands(arm=arm, reset_torso=reset_torso)

    def _get_bounds_sides(self, target: int) -> Tuple[List[np.array], List[bytes]]:
        sides, resp = super()._get_bounds_sides(target=target)
        # Set the y value to the highest point.
        max_y = -np.inf
        for s in sides:
            if s[1] > max_y:
                max_y = s[1]
        sides = [np.array((s[0], max_y, s[2])) for s in sides]
        # Don't try to pick up the top or bottom of a container.
        return sides, resp

    def _is_stoppable_collision(self, object_id: int) -> bool:
        # Stop for normal reasons or if the Magnebot collides with a container.
        return super()._is_stoppable_collision(object_id=object_id)
    
    def _check_action_status(self):
        return self.action.status == ActionStatus.ongoing or self.action.status == ActionStatus.success
    
    def _check_ongoing(self):
        if self.action.status == ActionStatus.ongoing:
            return True
        else:      
            return False
    
if __name__ == '__main__':
    TDWUtils.set_default_libraries(scene_library="E:/tdw_lib/scenes.json",
                                            model_library='E:/tdw_lib/models.json')
    c = Controller(check_version=False)
    bot_id = [c.get_unique_id() for _ in range(2)]
    pos = [
        {"x": -2, "y": 0, "z": 0},
        {"x": -1, "y": 0, "z": 0}
    ]
    bots = [Bot(pos[i], bot_id[i]) for i in range(2)]
    # Create a camera.
    camera = ThirdPersonCamera(position={"x": 2, "y": 10, "z": -1.5},
                            look_at={"x": 0, "y": 0.5, "z": 0},
                            avatar_id="a")
    # Add two Magnebots.
    c.add_ons.extend([camera])
    c.add_ons.extend(bots)
    # Load the scene.
    obj_id = [c.get_unique_id() for i in range(2)]
    comm1 = c.get_add_object(model_name='iron_box', object_id=obj_id[0], position={"x": 0, "y": 0, "z": 0})
    comm2 = c.get_add_object(model_name='iron_box', object_id=obj_id[1], position={"x": -1, "y": 0, "z": 0})
    c.communicate([{"$type": "load_scene",
                    "scene_name": "ProcGenScene"},
                TDWUtils.create_empty_room(12, 12),
                {"$type": "bake_nav_mesh"},
                c.get_add_object(model_name="iron_box",
                                       object_id=0,
                                       position={"x": -0.5, "y": 0, "z": 0}),
                {"$type": "make_nav_mesh_obstacle",
                       "id": 0,
                       "carve_type": "all",
                       "scale": 1,
                       "shape": "box"},
                {"$type": "set_screen_size",
                "width": 1280,
                "height": 720}])
    # Move the Magnebots.
    comm = bots[0].move_towards([0,0,0])

    while (bots[0]._check_ongoing()):
        c.communicate([])
    
    '''bots[0].turn_to(obj_id[0])
    while (bots[0]._check_ongoing()):
        c.communicate([])
    bots[0].move_to(obj_id[0])
    while (bots[0]._check_ongoing()):
        c.communicate([])
    bots[0].pick_up(obj_id[0])
    while (bots[0]._check_ongoing()):
        c.communicate([])
    c.communicate([])
    bots[0].reset_arm(Arm.left)
    while bots[0]._check_ongoing():
            c.communicate([])

    for i in range(8):
        bots[0].turn_right()
        while bots[0]._check_ongoing():
            c.communicate([])

    bots[0].turn_to(obj_id[1])
    while (bots[0]._check_ongoing()):
        c.communicate([])
    bots[0].move_to(obj_id[1])
    while (bots[0]._check_ongoing()):
        c.communicate([])
    bots[0].pick_up(obj_id[1])
    while (bots[0]._check_ongoing()):
        c.communicate([])
    c.communicate([])
    bots[0].reset_arm(Arm.left)
    while bots[0]._check_ongoing():
            c.communicate([])

    for i in range(8):
        bots[0].turn_right()
        while bots[0]._check_ongoing():
            c.communicate([])
    c.communicate({"$type": "terminate"})'''
'''    for i in range(8):
        bot.turn_right()
        while bot._check_ongoing():
            c.communicate([])
    for i in range(3):
        bot.move_forward()
        while bot._check_ongoing():
            c.communicate([])

    bot.pick_up(obj_id)
    
    while bot._check_ongoing():
            c.communicate([])
    
    bot.reset_arm(Arm.left)
    while bot._check_ongoing():
            c.communicate([])

    for i in range(8):
        bot.turn_right()
        while bot._check_ongoing():
            c.communicate([])
    for i in range(3):
        bot.move_forward()
        while bot._check_ongoing():
            c.communicate([])
    bot.drop(obj_id)
    while bot._check_ongoing():
            c.communicate([])
    bot.reset_arm(Arm.left)
    while bot._check_ongoing():
            c.communicate([])
    print(bot.action.status)
    print(bot.action.done)

    for i in range(8):
        bot.turn_right()
        while bot._check_ongoing():
            c.communicate([])
    for i in range(3):
        bot.move_forward()
        while bot._check_ongoing():
            c.communicate([])
'''
