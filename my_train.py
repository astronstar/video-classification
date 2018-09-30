from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


from dao import Dao

seq_length = 15
image_shape = (150, 150, 3)
batch_size = 8

train_data = Dao(
	'./train_dataset_desc', 
	seq_length=seq_length,
	image_shape=image_shape
)

validation_data = Dao(
	'./validation_dataset_desc', 
	seq_length=seq_length,
	image_shape=image_shape
)

train_gen = train_data.frame_generator(batch_size)
val_generator = validation_data.frame_generator(batch_size)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
# model.add(Dense(64))
# model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(train_data.num_of_classes()))
model.add(Activation('sigmoid'))

# 学习如何自定义score函数
def multi_category_score(y_true, y_pred):
	print('multi_category_score', y_true, y_pred)
	return 0.5

# 找一个比较好的loss。loss选择的标准？cat_acc是什么鬼？
model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])

model.fit_generator(
	generator=train_gen,
	steps_per_epoch=1,
	# steps_per_epoch=train_data.size() // batch_size,
	epochs=10000,
	verbose=1,
	validation_data=val_generator,
	validation_steps=1,
	workers=4
)

model.save_weights('first_try.h5')
