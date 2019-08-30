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

from base import Annotation, AnnotationTool, Rectangle, Circle, EventHandler

def flick(x):
    pass

class AnnotationGUI(object):
    def __init__(self, data_dir, output_dir,
            data_type='Video', data_ext='avi',
            with_annots=True, annots_file_ext='csv',
            yaml_config='mgh.yaml'):
       
        self.annotTool = AnnotationTool(
            data_dir, data_type, data_ext,
            with_annots, annots_file_ext, output_dir,
            yaml_config
        )
        
    def onselect(self, evt):
        # Note here that Tkinter passes an event object to onselect()
        w = evt.widget
        index = int(w.curselection()[0])
        data = self.annotTool.data_paths[index]
        if self.annotTool.current_mode == 'VID_ANNOT':
            annot = self.annotTool.annot_paths[index]
            print('You selected item %d: "%s => %s"' % (index, data, annot))
            self.show_video_with_annots(data, annot)
        elif self.annotTool.current_mode == 'VID_ANNOT_SCRATCH':
            print('You selected item %d: "%s => No annot"' % (index, data))
            self.show_video_scratch(data)
        elif self.annotTool.current_mode == 'SEQ_ANNOT':
            annot = self.annotTool.annot_paths[index]
            print('You selected item %d: "%s => %s"' % (index, data, annot))
            self.show_image_sequence_with_annots(data, annot)
        elif self.annotTool.current_mode == 'SEQ_ANNOT_SCRATCH':
            print('You selected item %d: "%s => No annot"' % (index, data))
            self.show_image_sequence_scratch(data)

    def cv2WindowInit(self, v_path, a_path):
        basepath = os.path.split(v_path)
        player_wname = basepath[1][:-4]
        control_wname = 'Controls'
        color_wname = 'Color mappings'
        
        cv2.destroyAllWindows()
        cv2.namedWindow(player_wname, cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow(player_wname, 400, 335)
        cv2.namedWindow(control_wname)
        cv2.moveWindow(control_wname, 400, 50)
        cv2.namedWindow(color_wname)
        cv2.moveWindow(color_wname, 400, 190)

        self.cap = cv2.VideoCapture(v_path)
        playerwidth = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        playerheight = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.annots = pd.read_csv(a_path)
        colorList = [[0, 0, 255], [0, 255, 170], [0, 170, 255], [0, 255, 0], [255, 0, 170], [255, 255, 0], [255, 0, 0]]
        colorDict = dict(zip(self.annotTool.joints, colorList))

        self.annotTool.initAnnotations(self.annotTool.joints, self.annotTool.joint_radius, self.annots,
                player_wname, playerwidth, playerheight, colorDict, self.annotTool.multiframe)
        cv2.setMouseCallback(player_wname, self.annotTool.dragCircle, self.annotTool.annotObj)
        self.controls = np.zeros((90, int(playerwidth * 2)), np.uint8)
        y0, dy = 20, 25
        for i, line in enumerate(self.annotTool.controls_text.split('\n')):
            y = y0 + i * dy
            cv2.putText(self.controls,
                        line + ' ',
                        (0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        i, x0, y = 0, 0, 20
        x = [0, 85, 270, 400, 510, 680, 800]
        self.color_map = np.zeros((40, int(playerwidth * 2), 3), np.uint8)
        self.color_map[:, :] = 255
        for this_joint in self.annotTool.joints:
            this_color = colorDict[this_joint]
            this_color = tuple(this_color)
            cv2.putText(self.color_map, this_joint, (x[i], y), cv2.FONT_HERSHEY_SIMPLEX, 1, this_color, 2)
            i += 1

        tots = len(self.annots.index)
        cv2.createTrackbar('S', player_wname, 0, int(tots) - 1, flick)
        cv2.setTrackbarPos('S', player_wname, 0)
        cv2.createTrackbar('F', player_wname, 1, 100, flick)
        frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        if frame_rate is None:
            frame_rate = 30
        cv2.setTrackbarPos('F', player_wname, frame_rate)
        
    def show_video_with_annots(self, v_path, a_path):
        self.cv2WindowInit(v_path, a_path)
        basepath = os.path.split(v_path)
        player_wname = basepath[1][:-4]
        control_wname = 'Controls'
        color_wname = 'Color mappings'
        
        tots = len(self.annots.index)
        i = 0
        status = 'stay'
        while True:
            playerwidth = self.annotTool.annotObj.keepWithin.width
            playerheight = self.annotTool.annotObj.keepWithin.height
            cv2.imshow(control_wname, self.controls)
            cv2.imshow(color_wname, self.color_map)
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, im = self.cap.read()
                if i == tots:
                    i = 0
                    status = 'stay'
                r = playerwidth / im.shape[1]
                dim = (int(playerwidth), int(im.shape[0] * r))
                im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

                cv2.imshow(player_wname, im)
                self.annotTool.event.updateAnnots(self.annotTool.annotObj, i, im)

                key = cv2.waitKey(10)
                status = {ord('s'): 'stay', ord('S'): 'stay',
                          ord('w'): 'play', ord('W'): 'play',
                          ord('a'): 'prev_frame', ord('A'): 'prev_frame',
                          ord('d'): 'next_frame', ord('D'): 'next_frame',
                          ord('q'): 'copy', ord('Q'): 'copy',
                          ord('o'): 'occluded', ord('O'): 'occluded',
                          ord('z'): 'save', ord('Z'): 'save',
                          ord('c'): 'quit',
                          ord('0'): 'no_annot',
                          ord('x'): 'incorrect_num',
                          ord('='): 'good',
                          ord('-'): 'bad',
                          # Keycodes for these obtained from online sources
                          ord('i'): 'move_marker_up',
                          ord('m'): 'move_marker_down',
                          ord('j'): 'move_marker_left',
                          ord('l'): 'move_marker_right',
                          255: status,
                          -1: status,
                          27: 'exit'}[key]

                if status == 'move_marker_up':
                    for joint_name in self.annotTool.annotObj.joints:
                        joint = self.annotTool.annotObj.joints[joint_name]
                        if joint.focus:
                            self.annotTool.annotObj.selectedJoint = joint

                    if self.annotTool.annotObj.selectedJoint:
                        joint = self.annotTool.annotObj.selectedJoint
                        curr_x, curr_y = int(joint.x_center), int(joint.y_center)
                        self.annotTool.event.trigger('pressMouseButton')(curr_x, curr_y, self.annotTool.annotObj)
                        self.annotTool.event.trigger('keyboardMoveMarker')(curr_x, curr_y-1, self.annotTool.annotObj)
                        self.annotTool.event.trigger('releaseMouseButton')(curr_x,curr_y-1, self.annotTool.annotObj)

                    status = 'stay'
                    if i % 10 == 0:
                        self.annots.to_csv(a_path, index=False)
                if status == 'move_marker_down':
                    for joint_name in self.annotTool.annotObj.joints:
                        joint = self.annotTool.annotObj.joints[joint_name]
                        if joint.focus:
                            self.annotTool.annotObj.selectedJoint = joint

                    if self.annotTool.annotObj.selectedJoint:
                        joint = self.annotTool.annotObj.selectedJoint
                        curr_x, curr_y = int(joint.x_center), int(joint.y_center)
                        self.annotTool.event.trigger('pressMouseButton')(curr_x, curr_y, self.annotTool.annotObj)
                        self.annotTool.event.trigger('keyboardMoveMarker')(curr_x, curr_y+1, self.annotTool.annotObj)
                        self.annotTool.event.trigger('releaseMouseButton')(curr_x,curr_y+1, self.annotTool.annotObj)

                    status = 'stay'
                    if i % 10 == 0:
                        self.annots.to_csv(a_path, index=False)
                if status == 'move_marker_left':
                    for joint_name in self.annotTool.annotObj.joints:
                        joint = self.annotTool.annotObj.joints[joint_name]
                        if joint.focus:
                            self.annotTool.annotObj.selectedJoint = joint

                    if self.annotTool.annotObj.selectedJoint:
                        joint = self.annotTool.annotObj.selectedJoint
                        curr_x, curr_y = int(joint.x_center), int(joint.y_center)
                        self.annotTool.event.trigger('pressMouseButton')(curr_x, curr_y, self.annotTool.annotObj)
                        self.annotTool.event.trigger('keyboardMoveMarker')(curr_x-1, curr_y, self.annotTool.annotObj)
                        self.annotTool.event.trigger('releaseMouseButton')(curr_x-1,curr_y, self.annotTool.annotObj)

                    status = 'stay'
                    if i % 10 == 0:
                        self.annots.to_csv(a_path, index=False)
                if status == 'move_marker_right':
                    for joint_name in self.annotTool.annotObj.joints:
                        joint = self.annotTool.annotObj.joints[joint_name]
                        if joint.focus:
                            self.annotTool.annotObj.selectedJoint = joint

                    if self.annotTool.annotObj.selectedJoint:
                        joint = self.annotTool.annotObj.selectedJoint
                        curr_x, curr_y = int(joint.x_center), int(joint.y_center)
                        self.annotTool.event.trigger('pressMouseButton')(curr_x, curr_y, self.annotTool.annotObj)
                        self.annotTool.event.trigger('keyboardMoveMarker')(curr_x+1, curr_y, self.annotTool.annotObj)
                        self.annotTool.event.trigger('releaseMouseButton')(curr_x+1,curr_y, self.annotTool.annotObj)

                    status = 'stay'
                    if i % 10 == 0:
                        self.annots.to_csv(a_path, index=False)
                if status == 'play':
                    frame_rate = cv2.getTrackbarPos('F', player_wname)
                    sleep((0.1 - frame_rate / 1000.0) ** 21021)
                    i += 1

                    if i == tots:
                        i = 0
                    cv2.setTrackbarPos('S', player_wname, i)
                    continue
                if status == 'stay':
                    i = cv2.getTrackbarPos('S', player_wname)
                if status == 'save':
                    self.annots.to_csv(a_path, index=False)
                    print('Progress saved!')
                    status = 'stay'
                if status == 'quit':
                    self.annots.to_csv(a_path, index=False)
                    print('Quit. Progress automatically saved!')
                    break
                if status == 'exit':
                    self.annots.to_csv(a_path, index=False)
                    print('Save & Quit!')
                    break
                if status == 'prev_frame':
                    i -= 1
                    cv2.setTrackbarPos('S', player_wname, i)
                    status = 'stay'
                if status == 'occluded':
                    joint = self.annotTool.event.occludedJoint(self.annotTool.annotObj)
                    if joint:
                        print(self.annots.loc[self.annots['frame_n'] == i, joint])
                    self.annots.to_csv(a_path, index=False)
                    status = 'stay'
                if status == 'next_frame':
                    i += 1
                    if i == tots:
                        i = 0
                    cv2.setTrackbarPos('S', player_wname, i)
                    status = 'stay'
                if status == 'copy':
                    if i != 0:
                        self.annots.iloc[i, 3: -1] = self.annots.iloc[i - 1, 3: -1]
                    if i % 10 == 0:
                        self.annots.to_csv(a_path, index=False)
                    status = 'stay'
                if status == 'slow':
                    frame_rate = max(frame_rate - 5, 0)
                    cv2.setTrackbarPos('F', player_wname, frame_rate)
                    status = 'play'
                if status == 'fast':
                    frame_rate = min(100, frame_rate + 5)
                    cv2.setTrackbarPos('F', player_wname, frame_rate)
                    status = 'play'
                if status == 'snap':
                    cv2.imwrite("./" + "Snap_" + str(i) + ".jpg", im)
                    print("Snap of Frame", i, "Taken!")
                    status = 'stay'
                if status == 'good':
                    self.annots.loc[self.annots['frame_n'] == i, 'quality'] = 1
                    i += 1
                    if i == tots:
                        i = 0
                    if i % 10 == 0:
                        self.annots.to_csv(a_path, index=False)
                    cv2.setTrackbarPos('S', player_wname, i)
                    status = 'stay'
                if status == 'bad':
                    self.annots.loc[self.annots['frame_n'] == i, 'quality'] = -1
                    i += 1
                    if i == tots:
                        i = 0
                    if i % 10 == 0:
                        self.annots.to_csv(a_path, index=False)
                    cv2.setTrackbarPos('S', player_wname, i)
                    status = 'stay'
                if status == 'no_annot':
                    self.annots.loc[self.annots['frame_n'] == i, 'quality'] = 0
                    i += 1
                    if i == tots:
                        i = 0
                    if i % 10 == 0:
                        self.annots.to_csv(a_path, index=False)
                    cv2.setTrackbarPos('S', player_wname, i)
                    status = 'stay'
                if status == 'incorrect_num':
                    # TODO: Add this function
                    # debug_list = self.annotTool.event.debug(self.annots)
                    # print('num_incorrect:' + str(len(debug_list)))
                    status = 'stay'
            except KeyError:
                print("Invalid Key was pressed")
            #except ValueError:
            #    print("Don't try going out of the box!")
            #    break

        cv2.destroyWindow(player_wname)
        cv2.destroyWindow(control_wname)
        cv2.destroyWindow(color_wname)

    def build_gui(self):
        root = Tk()
        l = Listbox(root, selectmode=SINGLE, height=30, width=60)
        l.grid(column=0, row=0, sticky=(N, W, E, S))
        s = ttk.Scrollbar(root, orient=VERTICAL, command=l.yview)
        s.grid(column=1, row=0, sticky=(N, S))
        l['yscrollcommand'] = s.set
        ttk.Sizegrip().grid(column=1, row=1, sticky=(S, E))
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)
        root.geometry('350x500+50+50')
        root.title('Select Video')
        for filename in self.annotTool.data_paths:
            l.insert(END, os.path.basename(filename))

        l.bind('<<ListboxSelect>>', self.onselect)
        return root
