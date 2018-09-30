from dao import Dao

d = Dao('./train_dataset_desc')

batch_size = 1
gen = d.frame_generator(batch_size)
for i in range(batch_size):
	a, b = next(gen)
	print(a.shape, b.shape)

