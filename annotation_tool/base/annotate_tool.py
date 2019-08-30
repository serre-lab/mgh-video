#!/usr/bin/env python

import cv2
import numpy as np

from time import sleep

import pandas as pd
import yaml

from tkinter import *
from tkinter import ttk
import os
import glob
from configparser import ConfigParser

from .shapes import Rectangle, Circle
from .annotations import Annotation
from .events import EventHandler

class AnnotationTool(object):
    def __init__(self, data_dir, data_type,
            data_ext, with_annots,
            annots_file_ext, output_dir,
            yaml_config):
        
        self.data_dir = data_dir
        self.data_type = data_type
        self.data_ext = data_ext
        self.annots_file_ext = annots_file_ext
        self.output_dir = output_dir
        self.with_annots = with_annots
        self.modes = ['VID_ANNOT_SCRATCH', 'VID_ANNOT', 'SEQ_ANNOT_SCRATCH', 'SEQ_ANNOT']

        self.annotObj = Annotation
        self.event = EventHandler()

        if self.data_type == 'Video':
            if self.with_annots:
                self.current_mode = self.modes[1]
            else:
                self.current_mode = self.modes[0]
        elif self.data_type == 'ImageSeq':
            if self.with_annots:
                self.current_mode = self.modes[3]
            else:
                self.current_mode = self.modes[2]

        if self.with_annots:
            self.data_paths, self.annot_paths = self.get_data()
        else:
            self.data_paths = self.get_data()

        with open(yaml_config, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.controls_text = ""
        for k, v in self.config['controls'].items():
            self.controls_text += ("%s: %s," % (k, v))
        self.controls_text = self.controls_text[:-1]

        self.joints = [x.strip() for x  in self.config['joint_names'].split(',')]
        self.joint_radius = int(self.config['joint_radius'])
        self.multiframe = int(self.config['multiframe'])

    def get_data(self):
        vpaths = sorted(glob.glob(os.path.join(self.data_dir, '*.' + self.data_ext)))
        if self.with_annots:
            apaths = sorted(glob.glob(os.path.join(self.data_dir, '*.' + self.annots_file_ext)))
            return vpaths, apaths
        return vpaths

    def initAnnotations(self, joints, radius, annots, player_wname, playerwidth,
            playerheight, colorDict, multiframe):
       
        self.annotObj.wname = player_wname
        self.annotObj.joints_df = annots

        self.annotObj.keepWithin.x = 0
        self.annotObj.keepWithin.y = 0
        self.annotObj.keepWithin.width = playerwidth
        self.annotObj.keepWithin.height = playerheight

        self.annotObj.colorDict = colorDict
        self.annotObj.multiframe = multiframe
    
        for jt in joints:
            self.annotObj(jt)
            self.annotObj.joints[jt].x_center = 0
            self.annotObj.joints[jt].y_center = 0
            self.annotObj.joints[jt].radius = radius
            self.annotObj.active = True
        
    def dragCircle(self, event, x, y, flags, annotObj):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.event.trigger('pressMouseButton')(x, y, annotObj)
        if event == cv2.EVENT_LBUTTONUP:
            self.event.trigger('releaseMouseButton')(x, y, annotObj)
        if event == cv2.EVENT_MOUSEMOVE:
            self.event.trigger('moveMousePointer')(x, y, annotObj)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.event.trigger('mouseDoubleClick')(x, y, annotObj)
