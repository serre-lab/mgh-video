import argparse
import logging
import time

import cv2
import math
import numpy as np
import os
import fnmatch

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

import csv
import pandas as pd
import glob
import yaml

from tqdm import tqdm, trange
import multiprocessing
from multiprocessing import Pool

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow pose estimation - Openpose')
    parser.add_argument('--conf', type=str, default='configs/mgh.yaml',
            help='Path to the YAML config file containing the parameters helpful for Openpose inference')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf_vals = yaml.load(f, Loader=yaml.FullLoader)
    conf_vals = conf_vals['estimate_pose']

    logger.debug('initialization %s : %s' % (conf_vals['model_name'], get_graph_path(conf_vals['model_name'])))
    w, h = model_wh(conf_vals['resolution'])
    e = TfPoseEstimator(get_graph_path(conf_vals['model_name']), target_size=(w, h))

    subdirs = os.listdir(conf_vals['data_dir'])

    def process_vids(subs):
        for i in trange(len(subs), desc='Subject ID', position=0):
            sub = subs[i]
            video_root = os.path.join(conf_vals['data_dir'], sub)
            l_vids = []
            for root, dirs, fnames in os.walk(video_root):
                for fname in fnmatch.filter(fnames, '*.' + conf_vals['data_ext']):
                    l_vids.append(os.path.join(root, fname))
            l_vids = sorted(l_vids)

            columnTitle = conf_vals['column_names']
            for j in trange(len(l_vids), desc='Video Number', position=1):
                vid_f = l_vids[j]
                cap = cv2.VideoCapture(vid_f)
                vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
                vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                n_fr = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                if (cap.isOpened() == False):
                    print("Error opening video stream or file")
                    continue

                out_root = os.path.join(conf_vals['pose_dir'], sub)
                if not os.path.exists(out_root):
                    os.makedirs(out_root)

                # open csv file to write joints
                csvfile = open(os.path.join(out_root, os.path.basename(vid_f).replace(conf_vals['data_ext'], 'csv')), "w")

                jointwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                jointwriter.writerow(columnTitle)

                # Define the codec and create VideoWriter object
                i_fr = 0

                fps_time = 0
                imgs = []
                while(cap.isOpened()):
                    ret_val, image = cap.read()

                    if ret_val:
                        imgs.append(image)
                        i_fr += 1
                        # Batch inference: BS of 128
                        if i_fr % 128 == 0:
                            human_list = e.inference(imgs)

                            for i in range(len(human_list)):
                                fnum = 128 * (int(math.ceil(i_fr / 128.)) - 1) + i
                                humans = human_list[i]
                                joints = dict(zip(range(18), [{'x':0, 'y':0, 'score':0} for _ in range(18)]))
                                score_arr = []
                                for human in humans:
                                    for key in range(18):
                                        if key in human.body_parts:
                                            body_part = human.body_parts[key]
                                            center = (int(body_part.x * vid_width + 0.5), int(body_part.y * vid_height + 0.5))
                                            #if not ((40 < center[0] < 260) or (50 < center[1] < 200)):
                                            #    break
                                            joints[key]['x'] = center[0]
                                            joints[key]['y'] = center[1]
                                            joints[key]['score'] = body_part.score

                                        else:
                                            joints[key]['x'] = joints[key]['y'] = joints[key]['score'] = 0

                                score_arr = np.array([j['score'] for j in joints.values()])
                                jointwriter.writerow([str(sub),
                                    os.path.basename(vid_f),
                                    str(fnum),
                                    str(joints[0]['x'])+'-'+str(joints[0]['y'])+'-'+str(joints[0]['score']),
                                    str(joints[2]['x'])+'-'+str(joints[2]['y'])+'-'+str(joints[2]['score']),
                                    str(joints[3]['x'])+'-'+str(joints[3]['y'])+'-'+str(joints[3]['score']),
                                    str(joints[4]['x'])+'-'+str(joints[4]['y'])+'-'+str(joints[4]['score']),
                                    str(joints[5]['x'])+'-'+str(joints[5]['y'])+'-'+str(joints[5]['score']),
                                    str(joints[6]['x'])+'-'+str(joints[6]['y'])+'-'+str(joints[6]['score']),
                                    str(joints[7]['x'])+'-'+str(joints[7]['y'])+'-'+str(joints[7]['score']),
                                    str(0) if not np.any(score_arr) else str(1) if np.mean(score_arr[[0,2,3,4,5,6,7]]) > 5 and np.where(score_arr[[0,2,3,4,5,6,7]] == 0)[0].size == 0 else str(-1)])
                            imgs = []
                    else:
                        human_list = e.inference(imgs)

                        for i in range(len(human_list)):
                            fnum = 128 * (int(math.ceil(i_fr / 128.)) - 1) + i
                            humans = human_list[i]
                            joints = dict(zip(range(18), [{'x':0, 'y':0, 'score':0} for _ in range(18)]))
                            score_arr = []
                            for human in humans:
                                for key in range(18):
                                    if key in human.body_parts:
                                        body_part = human.body_parts[key]
                                        center = (int(body_part.x * vid_width + 0.5), int(body_part.y * vid_height + 0.5))
                                        #if not ((40 < center[0] < 260) or (50 < center[1] < 200)):
                                        #    break
                                        joints[key]['x'] = center[0]
                                        joints[key]['y'] = center[1]
                                        joints[key]['score'] = body_part.score

                                    else:
                                        joints[key]['x'] = joints[key]['y'] = joints[key]['score'] = 0

                            score_arr = np.array([j['score'] for j in joints.values()])
                            jointwriter.writerow([str(sub),
                                os.path.basename(vid_f),
                                str(fnum),
                                str(joints[0]['x'])+'-'+str(joints[0]['y'])+'-'+str(joints[0]['score']),
                                str(joints[2]['x'])+'-'+str(joints[2]['y'])+'-'+str(joints[2]['score']),
                                str(joints[3]['x'])+'-'+str(joints[3]['y'])+'-'+str(joints[3]['score']),
                                str(joints[4]['x'])+'-'+str(joints[4]['y'])+'-'+str(joints[4]['score']),
                                str(joints[5]['x'])+'-'+str(joints[5]['y'])+'-'+str(joints[5]['score']),
                                str(joints[6]['x'])+'-'+str(joints[6]['y'])+'-'+str(joints[6]['score']),
                                str(joints[7]['x'])+'-'+str(joints[7]['y'])+'-'+str(joints[7]['score']),
                                str(0) if not np.any(score_arr) else str(1) if np.mean(score_arr[[0,2,3,4,5,6,7]]) > 5 and np.where(score_arr[[0,2,3,4,5,6,7]] == 0)[0].size == 0 else str(-1)])
                        break

                csvfile.close()
                cap.release()
                cv2.destroyAllWindows()
    
    process_vids(subdirs)
logger.debug('finished+')
