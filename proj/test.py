from email.policy import default
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH
from tdw.add_ons.add_on import AddOn
from tdw.output_data import OutputData, Transforms, Rigidbodies
from actions import ActionStatus
from tdw.librarian import HumanoidAnimationLibrarian

PREFIX_ = 'file:///C:/Users/YYX/Desktop/proj/resource/'


class myBot(AddOn):
    def  __init__(self, **kwargs):
        super().__init__()
        self.args = kwargs
        self.transform = None
        self.id = kwargs['id']
        self.rig = None
        self.action_status = ActionStatus()
        self.lib = HumanoidAnimationLibrarian()
        self.actions_lib = {}

    
    def get_initialization_commands(self):
        default_value = {
            'name': 'man_suit', 
            'url': 'file:///resource/man_suit',
            'position': {'x': 0, 'y': 0, 'z': -1},
            'rotation': {'x': 0, 'y': 0, 'z': 0},
        }
        commands = []
        command = {
            '$type': 'add_humanoid'
        }
        for k, v in default_value.items():
            if k in self.args.keys():
                command[k] = self.args[k]
            else:
                command[k] = default_value[k]
        commands.append(command)
        commands.append(
            {"$type": "send_humanoids",
            "ids": [self.id],
            'frequency': 'always'}
        )

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
        if self.action_status.time_left == 0:
            self.action_status.end()
        else:
            self.action_status.time_left -= 1

            


    def move_by(self, dis):
        commands = []
        if 'walking' not in self.actions_lib.keys():
            commands.append(
                {"$type": "add_humanoid_animation",
                "name": "walking_1",
                'url': 'file:///resource/walking_1'}
            )

            record = self.lib.get_record("walking_1")
            self.actions_lib['walking'] = record

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
camera = ThirdPersonCamera(avatar_id="a",
                           position={"x": -5.5, "y": 5, "z": -2},
                           look_at={"x": 0, "y": 1.0, "z": -1})
path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("humanoid_minimal")
print(f"Images will be saved to: {path}")
capture = ImageCapture(avatar_ids=["a"], path=path)
# Start the controller.
c = Controller(check_version=False)
h = myBot(id=c.get_unique_id())
c.add_ons.extend([h, camera, capture])
# Create a scene and add a humanoid.
commands = [TDWUtils.create_empty_room(12, 12)]
# Add an animation.
#h.move_by(2)
while h.action_status.ongoing:
    c.communicate([])
# Play some loops.

c.communicate({"$type": "terminate"})


