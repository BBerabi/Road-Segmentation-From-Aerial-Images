from __future__ import print_function
import os
import subprocess
import cv2
import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from scipy import ndimage
import random
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from mask_to_submission import mask_to_submission_strings, masks_to_submission

os.environ['KAGGLE_USERNAME'] = ""
os.environ['KAGGLE_KEY'] = ""
FNULL = open(os.devnull, 'w')

argparser = argparse.ArgumentParser()
argparser.add_argument('--verbose', action='store', dest='verbose', help='verbosity of the script', default=True, type=bool)
argparser.add_argument('--batch-size', action='store', dest='batch_size', help='batch size for processing the samples', default=32, type=int)
argparser.add_argument('--early-patience', action='store', dest='early_patience', help='patience for early stopping', default=25, type=int)
argparser.add_argument('--epochs', action='store', dest='epochs', help='number of epochs', default=200, type=int)
argparser.add_argument('--valid-split', action='store', dest='valid_split', help='percentage of validation examples', default=0.1, type=float)
args = argparser.parse_args()
verbose = args.verbose

def download_data(verbosity=True):
	'''
	Downloads data from kaggle if not exists, then unpacks and deletes archives.
	'''
	if os.path.isdir(path_train) and os.path.isdir(path_test):
		return
	if verbose:
		print('Downloading data...')
	stdout = None if verbosity else FNULL
	stderr = None if verbosity else subprocess.STDOUT
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
	Submits solution on kaggle. Command is as follows:
	kaggle competitions submit -c cil-road-segmentation-2019 -f submission.csv -m "Message"
	'''
	submitbase = 'kaggle competitions submit -c cil-road-segmentation-2019 -f '
	submit_all = submitbase + fname + ' -m "' + message + '"'
	subprocess.call(submit_all.split(' '))

def pred_overlap(model, x_test, dim_train, dim_test):
	'''
	Make predictions by averaging over predictions of smaller overlapping segments.
	'''
	n_test = x_test.shape[0]
	tl = x_test[:,:dim_train,:dim_train,:]
	tr = x_test[:,:dim_train,-dim_train:,:]
	bl = x_test[:,-dim_train:,:dim_train,:]
	br = x_test[:,-dim_train:,-dim_train:,:]
	tl_pred = model.predict(tl)
	tr_pred = model.predict(tr)
	bl_pred = model.predict(bl)
	br_pred = model.predict(br)
	test_pred = np.zeros((n_test, dim_test, dim_test, 1))
	test_pred[:,:dim_train,:dim_train,:] += tl_pred
	test_pred[:,:dim_train,-dim_train:,:] += tr_pred
	test_pred[:,-dim_train:,:dim_train,:] += bl_pred
	test_pred[:,-dim_train:,-dim_train:,:] += br_pred
	pred_weights = np.zeros((n_test, dim_test, dim_test, 1))
	pred_weights[:,:dim_train,:dim_train,:] += 1
	pred_weights[:,:dim_train,-dim_train:,:] += 1
	pred_weights[:,-dim_train:,:dim_train,:] += 1
	pred_weights[:,-dim_train:,-dim_train:,:] += 1
	test_pred = np.divide(test_pred, pred_weights)
	return test_pred

def pred_resize(model, x_test, dim_train, dim_test):
	'''
	Downscales the images and upscales the predictions.
	'''
	n_test = x_test.shape[0]

	x_test_downscaled = []
	for i in range(n_test):
		img_ds = cv2.resize(x_test[i], dsize=(dim_train, dim_train), interpolation=cv2.INTER_CUBIC)
		x_test_downscaled.append(img_ds)
	x_test_downscaled = np.asarray(x_test_downscaled)

	test_pred_downscaled = model.predict(x_test_downscaled)

	test_pred_upscaled = []
	for i in range(n_test):
		upscaled = cv2.resize(test_pred_downscaled[i], dsize=(dim_test, dim_test), interpolation=cv2.INTER_CUBIC)
		test_pred_upscaled.append(upscaled)
	test_pred_upscaled = np.asarray(test_pred_upscaled)
	test_pred = np.expand_dims(test_pred_upscaled, -1)
	return test_pred

def create_submission(test_pred, sub_fname='test_submission.csv'):
	test_binary = np.ones(test_pred.shape)
	test_binary[test_pred < 0.5] = 0
	test_binary = test_binary.astype(int)
	# Get file names
	imgnames = [path_pred + fname for fname in sorted(os.listdir(path_test), key=lambda x: int( x[x.find('_')+1:x.find('.')] ))]
	# Save prediction images
	for i in range(len(imgnames)):
		imarr = test_binary[i,:,:,0]
		plt.imsave(imgnames[i], imarr, cmap=cm.gray)
	# Create submission file
	masks_to_submission(sub_fname, *imgnames)

def augment_rotate_full(x_train, y_train):
	'''
	Augments the training data by rotating the images by right angles.
	'''
	x_aug = []
	y_aug = []
	for i in range(x_train.shape[0]):
		for degree in [90,180,270]:
			# degree = random.randint(0,361)
			x_rot = ndimage.rotate(x_train[i], angle=degree, order=1, reshape=False, axes=(0,1))
			y_rot = ndimage.rotate(y_train[i], angle=degree, order=1, reshape=False, axes=(0,1))
			x_aug.append(x_rot)
			y_aug.append(y_rot)
	x_aug = np.array(x_aug)
	y_aug = np.array(y_aug)
	return x_aug, y_aug

def augment_flip(x_train, y_train):
	'''
	Augments the training data by flipping the images.
	'''
	x_aug = []
	y_aug = []
	for i in range(x_train.shape[0]):
		x_ud = np.flipud(x_train[i])
		y_ud = np.flipud(y_train[i])
		x_lr = np.fliplr(x_train[i])
		y_lr = np.fliplr(y_train[i])
		x_aug.append(x_ud)
		y_aug.append(y_ud)
		x_aug.append(x_lr)
		y_aug.append(y_lr)
	x_aug = np.array(x_aug)
	y_aug = np.array(y_aug)
	return x_aug, y_aug

def augment_rotate_zoom(x_train, y_train, trials):
	'''
	Augments the training data by rotating the images arbitrarily, then zooming.
	'''
	crop_len = 60
	x_aug = []
	y_aug = []
	for i in range(x_train.shape[0]):
		for j in range(trials):
			degree = random.randint(0,361)
			rotated_img = ndimage.rotate(x_train[i], angle=degree, order=1, reshape=False, axes=(0,1))
			cropped_img = rotated_img[crop_len:-crop_len,crop_len:-crop_len]
			x_rot = cv2.resize(cropped_img, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)
			x_aug.append(x_rot)
			rotated_img = ndimage.rotate(y_train[i], angle=degree, order=1, reshape=False, axes=(0,1))
			cropped_img = rotated_img[crop_len:-crop_len,crop_len:-crop_len]
			y_rot = cv2.resize(cropped_img, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)
			y_aug.append(y_rot)
	x_aug = np.array(x_aug)
	y_aug = np.array(y_aug)
	y_aug = y_aug[:,:,:,np.newaxis]
	return x_aug, y_aug

def augment_crop_zoom(x_train, y_train, crop_len=100):
	'''
	Augments the training data by cropping and zooming the images.
	'''
	x_aug = []
	y_aug = []
	dim_train = x_train.shape[1]
	if type(crop_len) == int:
		crop_len = [crop_len for x in range(4)]
	for i in range(x_train.shape[0]):
		img = x_train[i]
		cropped_img = img[crop_len[0]:-crop_len[1],crop_len[2]:-crop_len[3]]
		x_crop = cv2.resize(cropped_img, dsize=(dim_train, dim_train), interpolation=cv2.INTER_CUBIC)
		x_aug.append(x_crop)
		img = y_train[i]
		cropped_img = img[crop_len[0]:-crop_len[1],crop_len[2]:-crop_len[3]]
		y_crop = cv2.resize(cropped_img, dsize=(dim_train, dim_train), interpolation=cv2.INTER_CUBIC)
		y_aug.append(y_crop)
	x_aug = np.array(x_aug)
	y_aug = np.array(y_aug)
	return x_aug, y_aug

def hold_out_validation(x_train, y_train, valid_split=args.valid_split):
	'''
	Hold-out validation splitting.
	'''
	valid_split = np.floor(x_train.shape[0] * valid_split).astype(int)
	valid_indices = np.random.choice(x_train.shape[0], valid_split)
	train_indices = np.setdiff1d(np.arange(x_train.shape[0]), valid_indices)
	x_valid = x_train[valid_indices]
	x_train = x_train[train_indices]
	y_valid = y_train[valid_indices]
	y_train = y_train[train_indices]
	return x_train, y_train, x_valid, y_valid

def build_model():
	'''
	Constructs a baseline model.
	'''
	model = Sequential()
	model.add(Conv2D(256, (3, 3), padding='same',
	                 input_shape=x_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(Conv2D(256, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(128, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(128, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2DTranspose(64, (3,3), strides=(2,2)))
	model.add(Activation('relu'))
	model.add(Conv2DTranspose(32, (3,3), strides=(1,1)))
	model.add(Dropout(0.2))

	model.add(Conv2D(16, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2DTranspose(4, (3,3), strides=(2,2), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2DTranspose(1, (3,3), strides=(1,1)))
	model.add(Dropout(0.2))

	model.add(Activation('sigmoid'))
	return model

# Load train and test data
path_train = './training/'
path_test = './test_images/'
path_pred = './pred_ims/'
path_out = './outdir/'

download_data()
for directory in [path_pred, path_out]:
	if not os.path.exists(directory):
		print(directory, ' not exists')
		os.makedirs(directory)

x_train, y_train, x_test = load_data()
dim_test = x_test.shape[1]
dim_train = x_train.shape[1]
if verbose:
	print('Loaded the data...')

# Hold-out validation
if args.valid_split > 0:
	x_train, y_train, x_valid, y_valid = hold_out_validation(x_train, y_train, valid_split=0.1)

# Augment the data
if verbose:
	print(x_train.shape[0], 'original training examples')

# Symmetry
x_flip, y_flip = augment_flip(x_train, y_train)
x_train = np.concatenate((x_train, x_flip), axis=0)
y_train = np.concatenate((y_train, y_flip), axis=0)
if verbose:
	print(x_flip.shape[0], 'flipped training examples')

# Rotation
x_rot, y_rot = augment_rotate_full(x_train, y_train)
x_aug, y_aug = augment_rotate_zoom(x_train, y_train, trials=1)
if verbose:
	print(x_rot.shape[0], 'rotated training examples')
	print(x_aug.shape[0], 'rotated and zoomed training examples')

x_train = np.concatenate((x_train, x_rot, x_aug), axis=0)
y_train = np.concatenate((y_train, y_rot, y_aug), axis=0)
if verbose:
	print(x_train.shape[0], 'training examples in total')
	print('Augmented the training data...')

# Build and compile the model
model = build_model()
opt = keras.optimizers.Adam(0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

# Set callbacks
file_bestval_checkpoint = path_out + 'baseline_bestval.h5'
file_periodic_checkpoint = path_out + 'baseline_periodic-{epoch:02d}.h5'
bestval_checkpoint = ModelCheckpoint(file_bestval_checkpoint, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
periodic_checkpoint = ModelCheckpoint(file_periodic_checkpoint, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=5)
early = EarlyStopping(monitor="val_acc", mode="max", patience=args.early_patience, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=15, verbose=2)
callbacks_list = [periodic_checkpoint]
if args.valid_split > 0:
	callbacks_list.append(early)
	callbacks_list.append(bestval_checkpoint)
	callbacks_list.append(redonplat)
if verbose:
	print('Compiled the model...')

# Train the model
if args.valid_split > 0:
	history = model.fit(x_train, y_train,
	          batch_size=args.batch_size,
	          epochs=args.epochs,
	          validation_data=(x_valid, y_valid),
	          callbacks=callbacks_list,
	          verbose=2,
	          shuffle=True)
else:
	model.fit(x_train, y_train,
	          batch_size=args.batch_size,
	          epochs=args.epochs,
	          callbacks=callbacks_list,
	          verbose=2,
	          shuffle=True)
if verbose:
	print('Finished training...')

# Save the history file
if args.valid_split > 0:
	np.savez(path_out + 'history.npz',
		acc=history.history['acc'],
		val_acc=history.history['val_acc'],
		loss=history.history['loss'],
		val_loss=history.history['val_loss'])

# Load the checkpoint model
if args.valid_split > 0:
	model.load_weights(file_bestval_checkpoint)

# Predict on test data
test_pred_overlap = pred_overlap(model, x_test, dim_train, dim_test)
np.save(path_out + 'test_pred_overlap.npy', test_pred_overlap)
test_pred_resize = pred_resize(model, x_test, dim_train, dim_test)
np.save(path_out + 'test_pred_resize.npy', test_pred_resize)

# Create submission
sub_fname = path_out + 'submission_overlap.csv'
create_submission(test_pred_overlap, sub_fname=sub_fname)
sub_fname = path_out + 'submission_resize.csv'
create_submission(test_pred_resize, sub_fname=sub_fname)
if verbose:
	print('Created the submission file...')

# Send submission file
# submit_solution(fname=submission_filename, message='test')
