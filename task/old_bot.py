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
from utils import l2_dis, eular_yaw
from tdw.output_data import OutputData, NavMeshPath
import math


FORWARD_DIS = 2
ARRIVE_AT = 0.2
TURN_ANGLE = 90
class Bot(Magnebot):
    def __init__(self, position, robot_id, debug=True):
        super().__init__(position=position, robot_id=robot_id, check_version=False)

        self._debug = debug
        
        self.navigation_plan = list()
        self.receive_plan = False
        self.navigation_done = False
        self.start_point = None

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

    def navigate_to(self, target_pos):
        self.receive_plan = True
        self.start_point = {
            'pos':self.dynamic.transform.position,
            'rot':eular_yaw(self.dynamic.transform.rotation)
        }
        return {"$type": "send_nav_mesh_path",
                "origin": TDWUtils.array_to_vector3(self.dynamic.transform.position),
                "destination": TDWUtils.array_to_vector3(target_pos)}
  

    def on_send(self, resp: List[bytes]) -> None:
        if self.receive_plan:
            self.receive_plan = False
            for i in range(len(resp) - 1):
                if OutputData.get_data_type_id(resp[i]) == "path":
                    path = NavMeshPath(resp[i]).get_path()
                    break
            for point in path[1:]:
                p = TDWUtils.array_to_vector3(point)
                p['y'] = 0
                self.navigation_plan.append(p)
        else:
            if self.start_point is not None and self.navigation_done == False:
                pos = self.dynamic.transform.position
                dist = l2_dis(pos[0], self.start_point['pos'][0], pos[2], self.start_point['pos'][2])
                if dist >= FORWARD_DIS:
                    self.navigation_done = True
                    self.navigation_plan.clear()
                    self.start_point = None
        return super().on_send(resp)

    def pick_up(self, target):
        for arm in [Arm.left, Arm.right]:
            self._pick_up(target, arm)
            if self._check_action_status():
                break
    
    def move_towards(self):
        self.move_to(self.navigation_plan[0])
        self.navigation_plan = self.navigation_plan[1:]
        return

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
        if self.navigation_done:
            self.navigation_done = False
            return False
        if self.action.status == ActionStatus.ongoing:
            return True
        else:
            if not(len(self.navigation_plan) == 0):
                self.move_to(self.navigation_plan[0])
                self.navigation_plan = self.navigation_plan[1:]
                return True                
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
    comm = bots[0].navigate_to([0,0,0])
    c.communicate([comm])

    while (bots[0]._check_ongoing()):
        c.communicate([])
    
    bots[0].turn_to(obj_id[0])
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
    c.communicate({"$type": "terminate"})
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
