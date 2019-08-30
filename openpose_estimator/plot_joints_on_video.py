import argparse
import logging
import time

import cv2
import os
import fnmatch

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

import yaml

from tqdm import tqdm, trange

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow draw pose on images - Openpose')
    parser.add_argument('--conf', type=str, default='configs/mgh.yaml',
            help='Path to the YAML config file containing the parameters helpful for Openpose inference')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf_vals = yaml.load(f, Loader=yaml.FullLoader)
    conf_vals = conf_vals['draw_pose']

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

            for j in trange(len(l_vids), desc='Video Number', position=1):
                vid_f = l_vids[j]
                cap = cv2.VideoCapture(vid_f)
                vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
                vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                n_fr = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                if (cap.isOpened() == False):
                    print("Error opening video stream or file")

                out_root = os.path.join(conf_vals['data_dir'], sub)
                if not os.path.exists(out_root):
                    os.makedirs(out_root)

                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                outfile = os.path.join(out_root, 'pose_' + os.path.basename(vid_f).replace(conf_vals['data_ext'], 'mp4'))
                out = cv2.VideoWriter(outfile, fourcc, 25, (int(vid_width), int(vid_height)))
                i_fr = 0

                fps_time = 0
                imgs = []
                while(cap.isOpened()):
                    ret_val, image = cap.read()
                    # print("Video: {} frame: {}".format(os.path.basename(vid_f), i_fr))

                    if ret_val:
                        imgs.append(image)
                        i_fr += 1
                        # Batch inference with BS 128
                        if i_fr % 128 == 0:
                            human_list = e.inference(imgs)

                            for humans, image in zip(human_list, imgs):
                                image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

                                cv2.putText(image,
                                            "FPS: %f" % (1.0 / (time.time() - fps_time)),
                                            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 255, 0), 2)

                                # write the flipped frame
                                out.write(image)

                                fps_time = time.time()
                            imgs = []
                    else:
                        human_list = e.inference(imgs)
                        for humans, image in zip(human_list, imgs):
                            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

                            cv2.putText(image,
                                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 2)

                            # write the flipped frame
                            out.write(image)

                            fps_time = time.time()
                        break

                cap.release()
                out.release()
                cv2.destroyAllWindows()

    process_vids(subdirs)
logger.debug('finished+')
