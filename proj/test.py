from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH
from tdw.add_ons.add_on import AddOn
from tdw.output_data import OutputData, Transforms, Rigidbodies
from actions import ActionStatus
from tdw.librarian import HumanoidAnimationLibrarian
from utils import get_pos_and_rot, l2_dis
from icecream import ic

PREFIX_ = 'file:///C:/Users/YangYuxiang/Desktop/proj/resource/'

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

    
    def get_initialization_commands(self):
        default_value = {
                'name': 'man_suit', 
                'url': PREFIX_ + 'man_suit',
                'position': {'x': 0, 'y': 0, 'z': -1},
                'rotation': {'x': 0, 'y': 0, 'z': 0},
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
                           follow_object=self.id,
                           follow_rotate=self.id)
        commands.extend(self.cam.get_initialization_commands())
        return commands
    
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
    
    

    def _solve_move_by(self):
        start_pos, _ = get_pos_and_rot(self.action_status.transform, self.id)
        curr_pos, _ = get_pos_and_rot(self.transform, self.id)

        distance = l2_dis(start_pos[0], curr_pos[0], start_pos[2], curr_pos[2])
        self.action_status.time_left -= 1
        if distance >= self.action_status.target:
            if self.action_status.time_left > 0:
                self.commands.append({"$type": "stop_humanoid_animation", "id": self.id})
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




# Add a camera and enable image capture.
c = Controller(check_version=False)
h = myBot(id=c.get_unique_id())
camera = ThirdPersonCamera(avatar_id="observer",
                           position={"x": -5.5, "y": 5, "z": -2},
                           look_at=h.id)
path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("test")
print(f"Images will be saved to: {path}")
capture = ImageCapture(avatar_ids=["a"], path=path)
# Start the controller.
c.add_ons.extend([h, camera, capture])
# Create a scene and add a humanoid.
c.communicate([TDWUtils.create_empty_room(32, 32)
               ])# Add an animation.
h.move_by(20)

# Play some loops.
while h.action_status.ongoing:
    c.communicate([])

for i in range(100):
    c.communicate([])
c.communicate({"$type": "terminate"})
