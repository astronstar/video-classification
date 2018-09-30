# encoding: utf-8
"""
实现读取数据集，包括train、validation、test的读取。
提供读取数据的迭代器。
"""
from subprocess import call
import os
import random
import glob
import math

import numpy as np
import shutil

from image_processor import process_image


class Dao():
	def __init__(self, 
				data_list_file, 
				class_num=63, 
				seq_length=15, 
				frame_rate=5, 
				image_shape=(150, 150, 3)):
		self.class_num = class_num
		self.image_shape = image_shape
		self.seq_length = seq_length
		self.frame_rate = frame_rate

		self.init_data(data_list_file)

	def size(self):
		return len(self.data_list)

	def num_of_classes(self):
		return self.class_num

	def init_data(self, data_list_file):
		self.data_list, self.labels = [], []

		with open(data_list_file, 'r') as f:
			for row in f:
				row = row.strip().split(',')
				video_path = row[0]
				self.data_list.append(video_path)

				_, video_filename = os.path.split(video_path)

				if len(row) > 1:
					labels = map(lambda x: int(x), row[1:])
					multi_hot_labels = [0] * self.class_num
					for label in labels:
						multi_hot_labels[label] = 1
					self.labels.append(multi_hot_labels)

	def frame_generator(self, batch_size):
		while True:
			X, y = [], []
			count = 0

			while True:
				'''sample_video指一个视频文件路径'''
				index = random.randint(0, len(self.labels)-1)
				frames = self.get_frames(self.data_list[index])
				if frames:
					# 这个extend本来是append的。。。只因维度太高了
					X.extend(frames)
					y.extend([self.labels[index]] * len(frames))
					count += 1
					if count >= batch_size:
						break
			yield np.array(X), np.array(y)

	def get_frames(self, sample_video_path):
		dest = sample_video_path + '_tmp/p-%04d.jpg'
		# dest = './tmp/hello' + '_tmp/p-%04d.jpg'
		dest_dir, _ = os.path.split(dest)
		if not os.path.isdir(dest_dir):
			os.makedirs(dest_dir)
		call(["ffmpeg", 
			"-loglevel", "panic", 
			"-i", sample_video_path, 
			"-r", str(self.frame_rate),
			dest])
		# pattern = './tmp/hello' + '_tmp/p-[0-9]*.jpg'
		pattern = sample_video_path + '_tmp/p-[0-9]*.jpg'
		image_files = sorted(glob.glob(pattern, recursive=True))
		step = len(image_files) // self.seq_length
		if step > 0:
			image_files = image_files[0:len(image_files):step][0:self.seq_length]
			frames = [process_image(x, self.image_shape) 
						for x in image_files]
		else:
			frames = None
		shutil.rmtree(dest_dir)
		return frames
