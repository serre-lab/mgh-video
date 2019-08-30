import tensorflow as tf
import numpy as np
import cv2
import time, math, sys, os
import matplotlib.pyplot as plt
import scipy.io as io
import tqdm
from skimage.measure import moments
from matplotlib.pyplot import cm
from central_reservoir.models import unet
from scipy.interpolate import interp1d

if sys.version_info.major == 3:
    PYTHON_VERSION = 3
else:
    PYTHON_VERSION = 2


eval_videos = ['Xxxxxx~ Xxxxxx_ceb2b945-a1b1-4627-8155-7c73937fcdb9_0167.mp4']
res_path = 'mgh_test_res_demo'
basepath = '/media/data_cifs/Kalpit/'#'/media/data_cifs/lakshmi/bmvc2018/convolutional-pose-machines-tensorflow'
bbox_size = 96
batch_size = 196
stride = 10
input_size = 192

def get_bboxes(img):
    boxes = []
    dims = img.shape
    for i in range(bbox_size,dims[0]-1-bbox_size,stride):
        for j in range(bbox_size, dims[1] - 1 - bbox_size,stride):
            im = img[i-bbox_size:i+bbox_size,j-bbox_size:j+bbox_size,:]
            boxes.append(im.astype(np.float32))
    return boxes

def main(argv):
    results_output = open('mgh_test_res_demo.txt','w')
    common_path = '/media/data_cifs/lakshmi/project-worms/mgh/'
    model_path_suffix = 'unet_adam1e-4_v2'
    model_save_dir = os.path.join(common_path,
                                  'model_runs',
                                    model_path_suffix)

    tf_device = '/gpu:0'
    ## Build the model
    with tf.device(tf_device):
        image = tf.placeholder(dtype=tf.float32,
                                shape=(None,448,448,3),
                                name='input_placeholder')
        model = unet.UNET(
                    use_batch_norm=True,
                    use_cross_replica_batch_norm=False,
                    num_classes=8,
		    single_channel=False)
        logits, end_points = model(
                    inputs=image,
                    is_training=False)

    gpuconfig = tf.ConfigProto()
    gpuconfig.gpu_options.allow_growth = True
    gpuconfig.allow_soft_placement = True
    saver = tf.train.Saver()

    """Create session and restore weights
    """
    eval_results = []
    cols = cm.rainbow(np.linspace(0, 1, 9))

    with tf.Session(config=gpuconfig) as sess:

        ckpts = tf.train.latest_checkpoint(model_save_dir)
	#ckpts = os.path.join(model_save_dir,
	#			'model-24000.ckpt-24000')
        saver.restore(sess, ckpts)
        
        # for every video in the test set
        for vid in eval_videos:
            folder = os.path.join(basepath,vid)
            #annotations = io.loadmat(os.path.join(basepath,vid+'_gt.mat'))
            #numFrames = len(annotations['GT_frame_ids'][0])
            numFrames = 3000
            vid = cv2.VideoCapture(folder)

            # for every annotated image within this set
            for im in tqdm.tqdm(range(1,numFrames)):

                ret, test_image = vid.read()
                test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                test_image = cv2.resize(test_image,(448,448))
                ori_im = test_image.copy()

                dims = test_image.shape
                # normalize the image
                test_image = test_image / 255.
                global_map = sess.run(logits, feed_dict={image:np.expand_dims(test_image,axis=0)})
                global_map = global_map.squeeze()
		
		xs, ys = get_prediction(global_map, ori_im.shape)

                disp_map = np.amax(global_map, axis=2)
                disp_map = np.reshape(disp_map, (ori_im.shape[0], ori_im.shape[1], 1))
                disp_map = np.repeat(disp_map, 3, axis=2)
                jnt_im = ori_im.copy()
                jnum = 0
                for x, y in zip(xs, ys):
                    jnt_im[x-5:x+5,y-5:y+5,:] = cols[jnum,:-1]*255. #[255,0,0]
                    jnum += 1

                upper_img = np.concatenate((ori_im, disp_map * 255, jnt_im), axis=1)
                plt.imshow(jnt_im.astype(np.uint8));
                plt.axis('off')
                #plt.show()
                plt.savefig(os.path.join(basepath,
                                         res_path,
                                         'img%06d.png'%im))
                plt.close()
                '''
                results_output.write('%s %s'%(vid,img))
                for (xx,yy) in zip(xs,ys):
                    results_output.write(' %d %d'%(xx,yy))

                #results_output.write('%s %s %d %d'%(vid,img,xs,ys))
                results_output.write('\n')
                '''
        results_output.close()

def eval_prediction_v2(pred_maps, annotations, im, orig_size, tol=20.):
    ### ALERT!
    ### Please note that the order of joints in the pred maps is [j1 -- j8 and then j0]
    #import ipdb; ipdb.set_trace();
    pck_eval = np.zeros((1,FLAGS.num_of_joints))
    xs, ys = [], []

    for jnt in range(FLAGS.num_of_joints):
        jc_pred = np.unravel_index(np.argmax(pred_maps[:, :, (jnt+8)%9]), (orig_size[0], orig_size[1]))
        xs.append(jc_pred[0])
        ys.append(jc_pred[1])

        pred_x, pred_y = jc_pred[0], jc_pred[1]
        gt_y, gt_x = annotations['xGT'][jnt,im], annotations['yGT'][jnt,im]

        #print (pred_x, pred_y, gt_x, gt_y)

        err = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
        if err <= tol:
            pck_eval[0,jnt] = 1.
        else:
            pck_eval[0,jnt] = 0.
    #import ipdb; ipdb.set_trace();
    #print (pck_eval)
    #plt.close(); plt.imshow(image); plt.scatter(ys, xs, c='r'); plt.pause(1); #plt.show(block=False)

    return pck_eval, xs, ys

def get_prediction(pred_maps, orig_size):
    ### ALERT!
    ### Please note that the order of joints in the pred maps is [j1 -- j8 and then j0]
    xs, ys = [], []
    for jnt in range(8):
        jc_pred = np.unravel_index(np.argmax(pred_maps[:, :, jnt]), (orig_size[0], orig_size[1]))
        xs.append(jc_pred[0])
        ys.append(jc_pred[1])
    return xs, ys

def eval_prediction(pred_maps, offsetX, offsetY, annotations, im, image, tol=20.):
    ### ALERT!
    ### Please note that the order of joints in the pred maps is [j1 -- j8 and then j0]

    pck_eval = np.zeros((1,FLAGS.num_of_joints))
    xs, ys = [], []

    for jnt in range(FLAGS.num_of_joints):
        jc_pred = np.unravel_index(np.argmax(pred_maps[:, :, (jnt+8)%9]), (FLAGS.input_size, FLAGS.input_size))
        xs.append(jc_pred[0])
        ys.append(jc_pred[1])

        pred_x, pred_y = jc_pred[0] + (offsetX-bbox_size), jc_pred[1] + (offsetY-bbox_size)
        gt_y, gt_x = annotations['xGT'][jnt,im], annotations['yGT'][jnt,im]

        #print (pred_x, pred_y, gt_x, gt_y)

        err = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
        if (err <= tol):
            pck_eval[0,jnt] = 1.
        else:
            pck_eval[0,jnt] = 0.
    #import ipdb; ipdb.set_trace();
    #print (pck_eval)
    #plt.close(); plt.imshow(image); plt.scatter(ys, xs, c='r'); plt.pause(1); #plt.show(block=False)

    return pck_eval

def vis_predictions(stage_heatmap_np,batch_gt_heatmap_np,batch_x_np):
    plt.close()
    pck_eval = np.zeros((FLAGS.batch_size, FLAGS.num_of_joints))

    for im_num in range(FLAGS.batch_size):

        # demo_stage_heatmaps = []
        # for stage in range(FLAGS.cpm_stages):
        #     demo_stage_heatmap = stage_heatmap_np[0][stage][im_num, :, :, 0:FLAGS.num_of_joints].reshape(
        #         (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
        #     demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size))
        #     demo_stage_heatmap = np.amax(demo_stage_heatmap, axis=2)
        #     demo_stage_heatmap = np.reshape(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size, 1))
        #     demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
        #     demo_stage_heatmaps.append(demo_stage_heatmap)
        #
        # demo_gt_heatmap = batch_gt_heatmap_np[im_num, :, :, 0:FLAGS.num_of_joints].reshape(
        #     (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
        # demo_gt_heatmap = cv2.resize(demo_gt_heatmap, (FLAGS.input_size, FLAGS.input_size))
        # demo_gt_heatmap = np.amax(demo_gt_heatmap, axis=2)
        # demo_gt_heatmap = np.reshape(demo_gt_heatmap, (FLAGS.input_size, FLAGS.input_size, 1))
        # demo_gt_heatmap = np.repeat(demo_gt_heatmap, 3, axis=2)
        # upper_img = np.concatenate((demo_stage_heatmaps[FLAGS.cpm_stages - 1], demo_gt_heatmap, (batch_x_np[im_num] + 0.5)),
        #                            axis=1)


        #### Evaluation
        #import ipdb; ipdb.set_trace();
        #tol = 3.319
        tol = 5.
        gt_heatmap = batch_gt_heatmap_np[im_num, :, :, 0:FLAGS.num_of_joints].reshape((FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
        gt_heatmap = cv2.resize(gt_heatmap, (FLAGS.input_size, FLAGS.input_size))

        pred_heatmap = stage_heatmap_np[0][FLAGS.cpm_stages-1][im_num, :, :, 0:FLAGS.num_of_joints].reshape(
            (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
        pred_heatmap = cv2.resize(pred_heatmap, (FLAGS.input_size, FLAGS.input_size))

        for jnt in range(FLAGS.num_of_joints):
            jc_gt = np.unravel_index(np.argmax(gt_heatmap[:,:,jnt]),(FLAGS.input_size,FLAGS.input_size))
            jc_pred = np.unravel_index(np.argmax(pred_heatmap[:,:,jnt]),(FLAGS.input_size,FLAGS.input_size))
            err = np.sqrt((jc_gt[0] - jc_pred[0])**2 + (jc_gt[1] - jc_pred[1])**2)
            if (err <= tol):
                pck_eval[im_num,jnt] = 1.
            else:
                pck_eval[im_num, jnt] = 0.

        #joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),(test_img.shape[0], test_img.shape[1]))
        #joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

        # plt.subplot(FLAGS.batch_size+1,1,im_num+1)
        # plt.imshow((upper_img * 255).astype(np.uint8))
        # plt.axis('off')
    #plt.show(block = False)
    return pck_eval

if __name__ == '__main__':
    tf.app.run()
