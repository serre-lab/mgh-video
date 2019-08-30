import tensorflow as tf
import numpy as np
import cv2
import time, math, sys, os, csv
import matplotlib.pyplot as plt
import tqdm
from matplotlib.pyplot import cm
from central_reservoir.models import unet
import glob, sys

def inference_fn(sub_names=None,
	basepath='/media/data_cifs/lester/bootstrap',
	model_path_suffix='unet_adam1e-4_preMG51B_B1',
	common_path='/media/data_cifs/lakshmi/project-worms/mgh',
	make_output_thumbs=False):

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
        saver.restore(sess, ckpts)
        for sub in sub_names:
            videos = glob.glob(sub+'/*')
	    # for every video in the set
	    for vid in videos:
                    sub_name = sub.split('/')[-1]
                    eval_video_name = vid.split('/')[-1]
		    res_path = os.path.join(basepath,
				sub_name,
				'pose_'+eval_video_name[:-3]+'csv')
		    results_csv = open(res_path,'w')
		    csv_writer = csv.writer(results_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		    csv_writer.writerow(['sub_n','video_name','frame_n','nose','r_shoulder','r_elbow','r_wrist','l_shoulder','l_elbow','l_wrist','quality'])
	 
		    video_path = os.path.join(basepath,sub_name,eval_video_name)
		    vid = cv2.VideoCapture(video_path)
		    frame_num = -1
		    pbar = tqdm.tqdm(total=3800)
		    while vid.isOpened():

			ret, test_image = vid.read()
			if (ret == False):
                            vid.release()
			    break
			ori_im_size = test_image.shape
			frame_num += 1
			test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
			ori_im = test_image.copy()
			test_image = cv2.resize(test_image,(448,448))
			pbar.update(1)

			dims = test_image.shape
			# normalize the image
			test_image = test_image / 255.
			global_map = sess.run(logits, feed_dict={image:np.expand_dims(test_image,axis=0)})
			global_map = global_map.squeeze()
			#import ipdb; ipdb.set_trace()	
			xs, ys = get_prediction(global_map, test_image.shape)
                        xs = [x * (ori_im_size[0]/448.) for x in xs]
			ys = [y * (ori_im_size[1]/448.) for y in ys]
			if make_output_thumbs:
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

			#results_output.write('%s %s'%(vid,img))
			row_entries = []
			row_entries.append(sub_name)
			row_entries.append(eval_video_name.split('/')[-1])
			row_entries.append(str(frame_num))
			for (xx,yy) in zip(xs,ys):
			    row_entries.append('%d-%d-10'%(yy,xx))
                        row_entries.append('0')
			csv_writer.writerow(row_entries)
	    pbar.close()        
	    results_csv.close()

def get_prediction(pred_maps, orig_size):
    xs, ys = [], []
    for jnt in range(7):
        jc_pred = np.unravel_index(np.argmax(pred_maps[:, :, jnt]), (orig_size[0], orig_size[1]))
        xs.append(jc_pred[0])
        ys.append(jc_pred[1])
    return xs, ys

def main():
    path_to_videos = '/media/data_cifs/lester/bootstrap'
    #subjects = glob.glob(path_to_videos+'/*')
    subject = os.path.join(path_to_videos, sys.argv[1])
    inference_fn(sub_names=[subject])
    '''
    for sub in subjects:
        videos = glob.glob(sub+'/*.mp4')
        for vid in tqdm.tqdm(videos):
            sub_n = sub.split('/')[-1]
            vid_n = vid.split('/')[-1]
            inference_fn(eval_video_name=vid,
				sub_name=sub_n)
    '''
if __name__ == '__main__':
    main()
