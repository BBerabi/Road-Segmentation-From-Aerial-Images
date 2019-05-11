import os
import cv2
import random
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from scipy import ndimage
from mask_to_submission import mask_to_submission_strings, masks_to_submission
# from keras.preprocessing.image import ImageDataGenerator

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

def load_data(path_train, path_test):
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

def download_data(path_train, path_test, verbosity=True):
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

def hold_out_validation(x_train, y_train, valid_split=0.1):
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

def pred_overlap(model, x_test, dim_train, dim_test, batch_size):
	'''
	Make predictions by averaging over predictions of smaller overlapping segments.
	'''
	n_test = x_test.shape[0]
	tl = x_test[:,:dim_train,:dim_train,:]
	tr = x_test[:,:dim_train,-dim_train:,:]
	bl = x_test[:,-dim_train:,:dim_train,:]
	br = x_test[:,-dim_train:,-dim_train:,:]
	tl_pred = model.predict(tl, batch_size)
	tr_pred = model.predict(tr, batch_size)
	bl_pred = model.predict(bl, batch_size)
	br_pred = model.predict(br, batch_size)
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

def pred_resize(model, x_test, dim_train, dim_test, batch_size):
	'''
	Downscales the images and upscales the predictions.
	'''
	n_test = x_test.shape[0]

	x_test_downscaled = []
	for i in range(n_test):
		img_ds = cv2.resize(x_test[i], dsize=(dim_train, dim_train), interpolation=cv2.INTER_CUBIC)
		x_test_downscaled.append(img_ds)
	x_test_downscaled = np.asarray(x_test_downscaled)

	test_pred_downscaled = model.predict(x_test_downscaled, batch_size)

	test_pred_upscaled = []
	for i in range(n_test):
		upscaled = cv2.resize(test_pred_downscaled[i], dsize=(dim_test, dim_test), interpolation=cv2.INTER_CUBIC)
		test_pred_upscaled.append(upscaled)
	test_pred_upscaled = np.asarray(test_pred_upscaled)
	test_pred = np.expand_dims(test_pred_upscaled, -1)
	return test_pred

def create_submission(test_pred, path_test, path_pred, sub_fname='test_submission.csv'):
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

def aug_flip(x_train, y_train):
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

def aug_rot_zoom(x_train, y_train, trials):
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

def aug_crop_zoom(x_train, y_train, crop_len=100):
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

def aug_rot_full(x_train, y_train):
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

# Currently not working

def submit_solution(fname='test', message='test'):
	'''
	Submits solution on kaggle. Command is as follows:
	kaggle competitions submit -c cil-road-segmentation-2019 -f submission.csv -m "Message"
	'''
	submitbase = 'kaggle competitions submit -c cil-road-segmentation-2019 -f '
	submit_all = submitbase + fname + ' -m "' + message + '"'
	subprocess.call(submit_all.split(' '))