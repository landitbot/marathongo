from abc import ABC, abstractmethod


class JoyState:
    def __init__(self):
        self.vel_forward = 0
        self.vel_rotation = 0
        self.btn_init_receiver = False
        self.btn_manual_control = False
        self.btn_auto_control = False
        self.btn_third_control = False
        self.btn_poweroff = False


class AbstractJOY(ABC):
    def __init__(self):
        super().__init__()
        self.on_update = None
        self.state = JoyState()

    def setCallback(self, fn):
        self.on_update = fn
        
    def stop():
        pass

    def updated(self):
        if self.on_update is not None:
            self.on_update(self.state)
