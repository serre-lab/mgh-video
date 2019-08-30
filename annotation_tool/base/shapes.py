import numpy as np
import json

class Rectangle(object):
    def __init__(self, x=None, y=None, w=None, h=None):
        self.x = x
        self.y = y
        self.width = w
        self.height = h

    def __repr__(self):
        print_dict = {}
        for k, v in self.__dict__.items():
            print_dict[k] = v
        return json.dumps(print_dict, indent=1)

class Circle(object):
    def __init__(self, cx=None, cy=None, rad=None,
            label=None):
        self.x_center = cx
        self.y_center = cy
        self.radius = rad

        # Whether the circle is being dragged or not
        self.drag = False
        
        # Whether it is a marker or another object
        self.is_marker = True

        self.active = True
        self.hold = False
        self.focus = False
        self.label = label

    def __repr__(self):
        print_dict = {}
        for k, v in self.__dict__.items():
            print_dict[k] = v
        return json.dumps(print_dict, indent=1)
