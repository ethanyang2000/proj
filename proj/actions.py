
class ActionStatus():
    def __init__(self) -> None:
        self.ongoing = False
        self.transform = None
        self.rig = None
        self.target = None
        self.record = None
        self.time_left = 0
    
    def start(self, transform, rig, target, record):
        self.ongoing = True
        self.transform = transform
        self.rig = rig
        self.target = target
        self.record = record
        self.time_left = record.get_num_frames()
    
    def end(self):
        self.ongoing = False
        self.transform = None
        self.rig = None
        self.target = None
        self.record = None
        self.time_left = 0
    
    def refresh(self):
        self.time_left = self.record.get_num_frames()
