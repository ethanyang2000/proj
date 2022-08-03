from numpy import angle
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH
from tdw.add_ons.add_on import AddOn
from tdw.output_data import OutputData, Transforms, Rigidbodies
from actions import ActionStatus
from tdw.librarian import HumanoidAnimationLibrarian
from utils import myRecord, get_obj_pos, get_pos_and_rot, l2_dis, euler_yaw
from icecream import ic
import math
from tdw.quaternion_utils import QuaternionUtils

PREFIX_ = 'file:///C:/Users/YangYuxiang/Desktop/proj/resource/'
ROTATE_UNIT_ = 3
OBJECT_DIS_ = 0.5

class myBot(AddOn):
    def  __init__(self, id, **kwargs):
        super().__init__()
        self.args = kwargs
        self.transform = None
        self.id = id
        self.rig = None
        self.action_status = ActionStatus()
        self.lib = HumanoidAnimationLibrarian()
        self.actions_lib = {}
        self.action_seq = []
        self.object_in_hand = None
    
    def get_initialization_commands(self):
        default_value = {
                'name': 'man_suit', 
                'url': PREFIX_ + 'man_suit',
                'position': {'x': -4, 'y': 0, 'z': 7},
                'rotation': {'x': 0, 'y': -190, 'z': 0},
                'id': self.id}
        commands = [{
            '$type': 'add_humanoid'
        }]
        for k, v in default_value.items():
            if k in self.args.keys():
                commands[0][k] = self.args[k]
            else:
                commands[0][k] = v
        commands.append(
            {"$type": "send_humanoids",
            "ids": [self.id],
            'frequency': 'always'}
        )
        self.cam = ThirdPersonCamera(avatar_id=str(self.id)+"_cam",
                           position={"x": -5.5, "y": 5, "z": -2},
                           look_at=self.id)

        
        commands.extend(self.cam.get_initialization_commands())
        return commands
    
    def navigate_to(self, pos):
        curr_pos, curr_angle = get_pos_and_rot(self.transform, self.id)
        tang = (pos[0]-curr_pos[0])/(pos[2]-curr_pos[2])
        theta = math.atan(tang)*180/math.pi
        if pos[0] > curr_pos[0]:
            while theta < 0:
                theta += 180
        elif pos[0] < curr_pos[0]:
            while theta > 0:
                theta -= 180
        else:
            theta = 0 if pos[3] > curr_pos[3] else 180
        angle_delta = theta - euler_yaw(curr_angle)

        dist_delta = l2_dis(pos[0], curr_pos[0], pos[2], curr_pos[2]) - 0.7

        self.action_seq = [
            {'func':self.rotate_by, 'target':angle_delta},
            {'func':self.move_by, 'target':dist_delta}
        ]
        self.action_seq[0]['func'](self.action_seq[0]['target'])
        self.action_seq.pop(0)

    def pick_up(self, obj_id):
        self.object_in_hand = {'id':obj_id}
        commands = []
        if 'pickup' not in self.actions_lib.keys():
            commands.append(
                {"$type": "add_humanoid_animation",
                "name": "hammering",
                'url': 'https://tdw-public.s3.amazonaws.com/humanoid_animations/linux/2019.2/' + 'hammering'}
            )
            self.actions_lib['pickup'] = self.lib.get_record("hammering")

        self.action_status.start(self.transform, self.rig, 0, self.actions_lib['pickup'])
        self.action_status.time_left = 3
        """commands.extend([
            {"$type": "play_humanoid_animation",
                  "name": 'hammering',
                  "id": self.id},
            {"$type": "set_target_framerate",
                 "framerate": self.actions_lib['pickup'].framerate}
        ])"""
        self.commands.extend(commands)



    def on_send(self, resp):
        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            # This is transforms output data.
            if r_id == "tran":
                transforms = Transforms(resp[i])
                for j in range(transforms.get_num()):
                    if transforms.get_id(j) == self.id:
                        # Log the position.
                        self.transform = transforms
                    if self.object_in_hand is not None and transforms.get_id(j) == self.object_in_hand['id']:
                        self.object_in_hand['trans'] = transforms
            elif r_id == "rigi":
                rigidbodies = Rigidbodies(resp[i])
                for j in range(rigidbodies.get_num()):
                    if rigidbodies.get_id(j) == self.id:
                        # Check if the object is sleeping.
                        self.rig = rigidbodies
        
        self.cam.on_send(resp)
        self.commands.extend(self.cam.commands)
        if self.action_status.ongoing:
            self.solve_action_status()
    
    def rotate_by(self, angle):
        self.action_status.start(self.transform, self.rig, angle, myRecord('rotate', angle))
        local_angle = min(ROTATE_UNIT_, abs(angle))
        if angle < 0:
            local_angle = -local_angle
        commands = [{"$type": "rotate_object_by", 
            "angle": local_angle,
            "id": self.id,
            "axis": "yaw",
            },
            {"$type": "rotate_avatar_by", 
            "angle": local_angle,
            "avatar_id": str(self.id)+'_cam',
            "axis": "yaw",
            }]
        
        self.commands.extend(commands)
    
    
    def _solve_rotate_by(self):
        if abs(self.action_status.time_left) <= ROTATE_UNIT_:
            self.action_status.end()
        else:
            if self.action_status.time_left < 0:
                self.action_status.time_left += ROTATE_UNIT_
            else:
                self.action_status.time_left -= ROTATE_UNIT_
            local_angle = min(ROTATE_UNIT_, abs(self.action_status.time_left))
            if self.action_status.time_left < 0:
                local_angle = -local_angle
            self.commands.extend([{"$type": "rotate_object_by", 
            "angle": local_angle,
            "id": self.id,
            "axis": "yaw",
            },
            {"$type": "rotate_avatar_by", 
            "angle": local_angle,
            "avatar_id": str(self.id)+'_cam',
            "axis": "yaw",
            }])
            if self.object_in_hand is not None:
                pos = get_obj_pos(self.transform, self.id, OBJECT_DIS_, local_angle)
                self.commands.extend(
                    [{'$type':'teleport_object',
                    'position':pos,
                    'id':self.object_in_hand['id']},
                    {"$type": "rotate_object_by", 
                    "angle": local_angle,
                    "id": self.object_in_hand['id'],
                    "axis": "yaw",
                    }]
                )

    
    def _solve_look_updown(self):
        if abs(self.action_status.time_left) <= ROTATE_UNIT_:
            self.action_status.end()
        else:
            if self.action_status.time_left < 0:
                self.action_status.time_left += ROTATE_UNIT_
            else:
                self.action_status.time_left -= ROTATE_UNIT_
            local_angle = min(ROTATE_UNIT_, abs(self.action_status.time_left))
            if self.action_status.time_left < 0:
                local_angle = -local_angle
            self.commands.extend([{"$type": "rotate_avatar_by", 
            "angle": local_angle,
            "avatar_id": str(self.id)+'_cam',
            "axis": "pitch",
            'is_world': False
            }])

    def _solve_move_by(self):
        start_pos, _ = get_pos_and_rot(self.action_status.transform, self.id)
        curr_pos, _ = get_pos_and_rot(self.transform, self.id)
        if self.object_in_hand is not None:
            pos = get_obj_pos(self.transform, self.id, OBJECT_DIS_)
            self.commands.extend(
                [{'$type':'teleport_object',
                'position':pos,
                'id':self.object_in_hand['id']}]
            )

        distance = l2_dis(start_pos[0], curr_pos[0], start_pos[2], curr_pos[2])
        self.action_status.time_left -= 1
        if distance >= self.action_status.target:
            if self.action_status.time_left > 0:
                self.commands.append({
                "$type": "stop_humanoid_animation",
                'id': self.id})
            self.action_status.end()
        else:
            if self.action_status.time_left == 0:
                self.commands.extend([
                    {"$type": "play_humanoid_animation",
                        "name": 'walking_1',
                        "id": self.id},
                    {"$type": "set_target_framerate",
                        "framerate": self.actions_lib['walking'].framerate}
                ])
                self.action_status.refresh()



    def solve_action_status(self):
        if self.action_status.record.name == 'walking_1':
            self._solve_move_by()
        elif self.action_status.record.name == 'rotate':
            self._solve_rotate_by()
        elif self.action_status.record.name == 'look_updown':
            self._solve_look_updown()
        elif self.action_status.record.name == 'hammering':
            self._solve_pick_up()
        
        if not(len(self.action_seq) == 0):
            self._solve_action_seq()

    def _solve_pick_up(self):
        if self.action_status.time_left == 0:
            self.action_status.end()
            pos = get_obj_pos(self.transform, self.id, OBJECT_DIS_)
            self.commands.extend(
                [{'$type':'teleport_object',
                'position':pos,
                'id':self.object_in_hand['id']}]
            )
        else:
            self.action_status.time_left -= 1

    def _solve_action_seq(self):
        if not self.action_status.ongoing:
            self.action_seq[0]['func'](self.action_seq[0]['target'])
            self.action_seq.pop(0)
    
    def move_by(self, dis):
        commands = []
        if 'walking' not in self.actions_lib.keys():
            commands.append(
                {"$type": "add_humanoid_animation",
                "name": "walking_1",
                'url': PREFIX_ + 'walking_1'}
            )
            self.actions_lib['walking'] = self.lib.get_record("walking_1")

        self.action_status.start(self.transform, self.rig, dis, self.actions_lib['walking'])
        commands.extend([
            {"$type": "play_humanoid_animation",
                  "name": 'walking_1',
                  "id": self.id},
            {"$type": "set_target_framerate",
                 "framerate": self.actions_lib['walking'].framerate}
        ])
        self.commands.extend(commands)

    def look_updown(self, angle):
        angle = -angle
        self.action_status.start(self.transform, self.rig, angle, myRecord('look_updown', angle))
        local_angle = min(ROTATE_UNIT_, abs(angle))
        if angle < 0:
            local_angle = -local_angle
        commands = [{"$type": "rotate_avatar_by", 
            "angle": local_angle,
            "avatar_id": str(self.id)+'_cam',
            "axis": "pitch",
            'is_world': False
            }]
        
        self.commands.extend(commands)

# Add a camera and enable image capture.
c = Controller(check_version=False)
h = myBot(id=c.get_unique_id())

"""camera = ThirdPersonCamera(avatar_id="observer",
                           position={"x": -5.5, "y": 5, "z": -2},
                           look_at=h.id)"""
path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("test")
print(f"Images will be saved to: {path}")
capture = ImageCapture(avatar_ids=[str(h.id)+"_cam"], path=path)
# Start the controller.
c.add_ons.extend([h, capture])
# Create a scene and add a humanoid.
obj_id = c.get_unique_id()
resp = c.communicate([TDWUtils.create_empty_room(32, 32),
        c.get_add_object(model_name="iron_box",
                    library="models_core.json",
                    position={"x": 0, "y": 0, "z": 0},
                    object_id=obj_id),
        {'$type':'send_transforms',
        'ids':[obj_id],
        'frequency':'always'
        }])# Add an animation.
for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    # This is transforms output data.
    if r_id == "tran":
        transforms = Transforms(resp[i])
        for j in range(transforms.get_num()):
            if transforms.get_id(j) == obj_id:
                # Log the position.
                pos = transforms.get_position(j)
h.navigate_to(pos)
while h.action_status.ongoing:
    c.communicate([])
h.pick_up(obj_id)
while h.action_status.ongoing:
    c.communicate([])
h.move_by(2)

# Play some loops.
while h.action_status.ongoing:
    resp = c.communicate([])
h.rotate_by(1000)
while h.action_status.ongoing:
    resp = c.communicate([])
for i in range(10):
    c.communicate([])

c.communicate({"$type": "terminate"})
