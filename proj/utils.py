from tdw.quaternion_utils import QuaternionUtils
import math
from icecream import ic

def get_pos_and_rot(transform, id):
    for i in range(transform.get_num()):
        if transform.get_id(i) == id:
            return transform.get_position(i), transform.get_rotation(i)
    return None, None


def euler_yaw(qua):
    angle = QuaternionUtils.quaternion_to_euler_angles(qua)[1]
    if abs(qua[3]) < 0.7071068:
        if qua[1] > 0:
            angle = 180-angle
        elif qua[1] < 0:
            angle = -180-angle
    while abs(angle) > 180:
        if angle < 0: angle += 360
        else: angle -= 360
    return angle

def l2_dis(x1, x2, y1, y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

def get_obj_pos(transform, id, dis, delta_angle=None):
    pos, rot = get_pos_and_rot(transform, id)
    angle = euler_yaw(rot)
    if delta_angle is not None:
        angle += (delta_angle)
    angle = math.radians(angle)
    delta_x = dis*math.sin(angle)
    delta_z = dis*math.cos(angle)
    return {'x':float(pos[0]+delta_x),'y':float(1),'z':float(pos[2]+delta_z)}

def all_moving(h, pool):
    for i in range(len(h)):
        if h[i].action_status.ongoing is False and len(pool[i]) == 0:
            return False
    return True

class myRecord():
    def __init__(self, name, num) -> None:
        self.name = name
        self.num = num
    
    def get_num_frames(self):
        return self.num