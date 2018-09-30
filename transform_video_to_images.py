import os
import shutil
from subprocess import call
import time

import numpy as np

from image_processor import process_image

FRAME_RATE = 4
SEQ_LENGTH = 15
image_shape=(128, 128, 3)

def transform_helper(video_path):
	target_dir = os.path.join('./images', video_path)
	image_path_pattern = target_dir + '/%04d.jpg'

	if os.path.isdir(target_dir):
		shutil.rmtree(target_dir)
	os.makedirs(target_dir)
	
	call(["ffmpeg", 
		"-loglevel", "panic", 
		"-i", video_path, 
		"-r", str(FRAME_RATE), 
		image_path_pattern])
	filenames = sorted(os.listdir(target_dir))
	total = len(filenames)
	if total < SEQ_LENGTH:
		return None

	step = total // SEQ_LENGTH
	needed_images = filenames[0:total:step][0:SEQ_LENGTH]
	frames = [process_image(os.path.join(target_dir, x), image_shape) 
				for x in needed_images]
	shutil.rmtree(target_dir)

	image_np_file = target_dir + '.npy'
	np.save(image_np_file, np.array(frames))

	return image_np_file

def transform_video_to_images(file_path):
	valid_lines = []
	count = 0
	with open(file_path) as f:
		for line in f:
			if line.find(',') > 0:
				parts = line.split(',')
				video_path = parts[0]
				image_np_file = transform_helper(video_path)
				if image_np_file:
					line = ','.join([image_np_file] + parts[1:])
					valid_lines.append(line)
				if count % 100 == 0:
					print('process, count={0}, now={1}'.format(count, time.time()))
				count += 1
	with open(file_path+'_npy', 'w') as f:
		for line in valid_lines:
			f.write(line)
	print('end, now=', time.time(), file_path)

# transform_video_to_images('./test_desc')
transform_video_to_images('./train_dataset_desc')
transform_video_to_images('./validation_dataset_desc')
