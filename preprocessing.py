import os

def get_file_list(data_dirs):
	file_list = []
	for data_dir in data_dirs:
		for filename in os.listdir(data_dir):
			filename = filename.strip()
			file_path = os.path.join(data_dir, filename)
			file_list.append((file_path, filename))
	return file_list

def gen_subset_data_desc(ann_path, data_dirs, dest):
	name2content = {}
	with open(ann_path, 'r') as f:
		for row in f:
			row = row.strip().split(',')
			filename = row[0]
			name2content[filename] = row[1:]

	file_list = get_file_list(data_dirs)
	with open(dest, 'w') as f:
		hello = 0
		for file_path, filename in file_list:
			labels = name2content.get(filename, None)
			if labels:
				content = [file_path] + labels
				content = ','.join(content) + '\n'
				f.write(content)
			else:
				print(filename)

train_ann = './train_data_settings/short_video_trainingset_annotations.txt.0829'
train_data_dirs = [
	'./train_data/group0', 
	'./train_data/group1'
]

gen_subset_data_desc(train_ann, train_data_dirs, './train_dataset_desc')


validation_ann = './validation_data_settings/short_video_validationset_annotations.txt.0829'
validation_data_dirs = [
	'./validation_data/group4', 
	'./validation_data/group5'
]
gen_subset_data_desc(validation_ann, validation_data_dirs, './validation_dataset_desc')

