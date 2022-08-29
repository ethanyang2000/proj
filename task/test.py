from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from magnebot import Magnebot, ActionStatus

from utils import eular_to_quat, eular_yaw

c = Controller()
# Create a camera.
camera = ThirdPersonCamera(position={"x": 2, "y": 6, "z": -1.5},
                           look_at={"x": 0, "y": 0.5, "z": 0},
                           avatar_id="a")
# Add two Magnebots.
mid = c.get_unique_id()
magnebot_0 = Magnebot(position={"x": -2, "y": 0, "z": 0},
                      robot_id=mid)
magnebot_1 = Magnebot(position={"x": 2, "y": 0, "z": 0},
                      robot_id=c.get_unique_id())
c.add_ons.extend([camera, magnebot_0, magnebot_1])
# Load the scene.
c.communicate([{"$type": "load_scene",
                "scene_name": "ProcGenScene"},
               TDWUtils.create_empty_room(12, 12)])
# Move the Magnebots.
magnebot_0.move_by(-2)
while magnebot_0.action.status == ActionStatus.ongoing:
    c.communicate([])
c.communicate([])
pos = magnebot_0.dynamic.transform.position
rot = eular_yaw(magnebot_0.dynamic.transform.rotation)
rot += 90
if rot > 360:
    rot -= 360
qua = eular_to_quat([0,0,0])
from icecream import ic
pos = [0,0,0]
ic(pos)
ic(qua)
c.communicate([])
c.communicate([{"$type": 'teleport_robot', 'position':list(pos), 'rotation':list(qua), 'id':mid}])
c.communicate({"$type": "terminate"})