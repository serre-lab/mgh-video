import numpy as np
import json

from .shapes import Rectangle, Circle

class Annotation(object):
    # Limits of the canvas
    keepWithin = Rectangle()

    selectedJoint = None
    size_marker = 2
    initialized = False
    joints = {}
    image = None

    wname = ""
    multiframe = 0
    returnflag = False
    frame_n = 0
    joints_df = []
    colorDict = {}

    def __init__(self, label):
        self.joints[label] = Circle(label=label)
