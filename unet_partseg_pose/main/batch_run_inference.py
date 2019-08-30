import glob
from inference_fn import inference_fn
import tqdm

def main():
	path_to_videos = '/media/data_cifs/lakshmi/MGH/data'
	subjects = glob.glob(path_to_videos+'/*')
	for sub in subjects:
		videos = glob.glob(sub+'/*.mp4')
		for vid in tqdm.tqdm(videos):
			sub_n = sub.split('/')[-1]
			vid_n = vid.split('/')[-1]
			inference_fn(eval_video_name=vid,
				sub_name=sub_n)
		

if __name__ == '__main__':
	main()
