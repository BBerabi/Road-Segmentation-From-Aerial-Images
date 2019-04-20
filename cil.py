from __future__ import print_function
import os
import subprocess
import cv2
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

os.environ['KAGGLE_USERNAME'] = ""
os.environ['KAGGLE_KEY'] = ""
FNULL = open(os.devnull, 'w')

def download_data(verbose=False):
	'''
	Downloads data from kaggle if not exists, then unpacks and deletes archives.
	'''
	if os.path.isdir(path_train) and os.path.isdir(path_test):
		return
	stdout = None if verbose else FNULL
	stderr = None if verbose else subprocess.STDOUT
	subprocess.call('kaggle competitions download -c cil-road-segmentation-2019'.split(' '), stdout=stdout, stderr=stderr)
	subprocess.call('unzip training.zip'.split(' '), stdout=stdout, stderr=stderr)
	subprocess.call('unzip sample_submission.csv.zip'.split(' '), stdout=stdout, stderr=stderr)
	subprocess.call('tar xvzf test_images.tar.gz'.split(' '), stdout=stdout, stderr=stderr)
	subprocess.call('rm training.zip sample_submission.csv.zip test_images.tar.gz'.split(' '), stdout=stdout, stderr=stderr)

def threshold_vals(grayscale):
	'''
	Converts the given grayscale images to binary by thresholding.
	'''
	hist = np.bincount(grayscale.flatten())
	normd = hist[1:-1] / np.sum(hist[1:-1])
	cutval = np.where(np.cumsum(normd) > 0.5)[0][0] + 1
	grayscale[grayscale > cutval] = np.max(grayscale)
	grayscale[grayscale <= cutval] = np.min(grayscale)
	grayscale = np.array(grayscale, dtype=np.float32)
	return grayscale

def load_data():
	'''
	Loads data into numpy arrays in sorted order.
	'''
	imgdir = path_train + 'images/'
	imgnames = [imgdir + fname for fname in sorted(os.listdir(imgdir))]
	train_ims = np.array([np.asarray(cv2.imread(imgname)) for imgname in imgnames], dtype=np.float32)

	imgdir = path_train + 'groundtruth/'
	imgnames = [imgdir + fname for fname in sorted(os.listdir(imgdir))]
	train_gt = np.array([np.asarray(cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)) for imgname in imgnames], dtype=int)
	train_gt = threshold_vals(train_gt)
	train_gt = np.expand_dims(train_gt / np.max(train_gt), axis=-1)

	imgdir = path_test
	imgnames = [imgdir + fname for fname in sorted(os.listdir(imgdir), key=lambda x: int( x[x.find('_')+1:x.find('.')] ))]
	test_ims = np.array([np.asarray(cv2.imread(imgname)) for imgname in imgnames], dtype=np.float32)

	return train_ims, train_gt, test_ims

def submit_solution(fname='test', message='test'):
	'''
	Submits solution on kaggle.
	'''
	submitbase = 'kaggle competitions submit -c cil-road-segmentation-2019 -f '
	submit_all = submitbase + fname + '.csv -m "' + message + '"'
	subprocess.call(submit_all.split(' '))

def build_model():
	'''
	Constructs a baseline model.
	'''
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same',
	                 input_shape=x_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(32, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2DTranspose(32, (3,3), strides=(2,2)))
	model.add(Activation('relu'))
	model.add(Conv2DTranspose(16, (3,3), strides=(1,1)))
	model.add(Dropout(0.25))

	model.add(Conv2D(8, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2DTranspose(4, (3,3), strides=(2,2), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2DTranspose(1, (3,3), strides=(1,1)))
	model.add(Dropout(0.25))

	model.add(Activation('sigmoid'))
	return model

# Load train and test data
path_train = './training/'
path_test = './test_images/'
download_data()
x_train, y_train, x_test = load_data()
dim_test = x_test.shape[1]
dim_train = x_train.shape[1]

# Set hyperparameters
batch_size = 16
epochs = 100

# Build and compile the model
model = build_model()
opt = keras.optimizers.Adam(0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

# Set callbacks
file_checkpoint = "baseline_cnn_check.h5"
checkpoint = ModelCheckpoint(file_checkpoint, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=7, verbose=2)
callbacks_list = [checkpoint, early, redonplat]

# Train the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,
          callbacks=callbacks_list,
          shuffle=True)

# Load the checkpoint model
model.save_weights('baseline_cnn_redun.h5')
model.load_weights(file_checkpoint)
