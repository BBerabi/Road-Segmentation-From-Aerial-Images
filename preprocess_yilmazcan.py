import numpy as np
from sklearn.model_selection import train_test_split
import os
import skimage.io as io
from PIL import Image #use for augs
import cv2
from scipy import ndimage
import random
np.random.seed(42)
def create_data_dirs(train_path,val_ratio=0.2, seed=42, build_augs_folder=True):
	#this method creates validation folder so later on we can call the data generator from there,
	#but also creates a train folder with augmentations and you can use that if you want

	#give train_path, and the ratio of images to be validated
	#seed helps keeping track of the order for training, makes more sense for test
	#if build_val_folder, keep a new folder for that particular val set to see what's in it
	#if build_augs_folder, keep a new folder for the actual set that is used with all the augmentations
	#I also used that for aug flag, if you don't want augs set to false

	N= 100 #hard coded since we have a particular number of train images to use, ordered
	val_path= train_path +"/../validation"

	#create validation folders and move data there
	if not os.path.exists(val_path):
		os.makedirs(val_path)
	if not os.path.exists(os.path.join(val_path, 'images')):
		os.makedirs(os.path.join(val_path, 'images'))
	if not os.path.exists(os.path.join(val_path, 'groundtruth')):
		os.makedirs(os.path.join(val_path, 'groundtruth'))                
	
	train, val = train_test_split(range(1,N+1), test_size=val_ratio, random_state=seed)

	for i in val:
		image = Image.open(os.path.join(train_path, 'images', 'satImage_%.3d.png'%i))
		label = Image.open(os.path.join(train_path, 'groundtruth', 'satImage_%.3d.png'%i))

		io.imsave(os.path.join(val_path, 'images', 'satImage_%.3d.png'%i), np.array(image))
		io.imsave(os.path.join(val_path, 'groundtruth', 'satImage_%.3d.png'%i), np.array(label))


	if build_augs_folder:
	#each time you change something in augs, (add or subtract stuff), change below path so its
	#saved somewhere else
		aug_path= train_path + "/../aug_train"
		if not os.path.exists(aug_path):
			os.makedirs(aug_path)
			os.makedirs(os.path.join(aug_path, 'images'))
			os.makedirs(os.path.join(aug_path, 'groundtruth'))

		for i in train:
			image = Image.open(os.path.join(train_path, 'images', 'satImage_%.3d.png'%i))
			label = Image.open(os.path.join(train_path, 'groundtruth', 'satImage_%.3d.png'%i))

			#Copy original images to aug_path
			io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d.png'%i), np.array(image))
			io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d.png'%i), np.array(label))

			#create horizontally flipped images
			im_flip_lr = image.transpose(Image.FLIP_LEFT_RIGHT)
			io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_flr.png'%i), np.array(im_flip_lr))

			label_flip_lr = label.transpose(Image.FLIP_LEFT_RIGHT)
			io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_flr.png'%i), np.array(label_flip_lr))

			#create vertically flipped images
			im_flip_tb = image.transpose(Image.FLIP_TOP_BOTTOM)
			io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_ftb.png'%i), np.array(im_flip_tb))

			label_flip_tb = label.transpose(Image.FLIP_TOP_BOTTOM)
			io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_ftb.png'%i), np.array(label_flip_tb))
		
			""" 90 rotations + 5 random rotation
			#create rotated images, also rotated ones
			for angle in [90, 180, 270]:
				im_r = image.rotate(angle)
				io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_%.3d.png'%(i, angle)), np.array(im_r))

				im_f_r = im_flip.rotate(angle)
				io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_f_%.3d.png'%(i, angle)), np.array(im_f_r))

				label_r = label.rotate(angle)
				io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_%.3d.png'%(i, angle)), np.array(label_r))

				label_f_r = label_flip.rotate(angle)
				io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_f_%.3d.png'%(i, angle)), np.array(label_f_r))

			#create rotated images in weird angles, and zoom appropriately
			for trial in range(5):
				angle = random.randint(0,259)
				#to get an angle other han 90,180 or 270
				while angle in [90, 180, 270, 360]:
					angle = random.randint(0,359)
				image_rotated, label_rotated = aug_rot_zoom(np.array(image), np.array(label), angle)
				io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_%.3d.png'%(i, angle)), np.array(image_rotated))
				io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_%.3d.png'%(i, angle)), np.array(label_rotated))
			"""

			#Rotations of every 5 degrees
			for angle in range(5,360,5):

				image_rotated, label_rotated = aug_rot_zoom(np.array(image), np.array(label), angle)
				io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_%.3d.png'%(i, angle)), np.array(image_rotated))
				io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_%.3d.png'%(i, angle)), np.array(label_rotated))

				image_rotated_flr, label_rotated_flr = aug_rot_zoom(np.array(im_flip_lr), np.array(label_flip_lr), angle)
				io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_flr_%.3d.png'%(i, angle)), np.array(image_rotated_flr))
				io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_flr_%.3d.png'%(i, angle)), np.array(label_rotated_flr))

				image_rotated_ftb, label_rotated_ftb = aug_rot_zoom(np.array(im_flip_tb), np.array(label_flip_tb), angle)
				io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_ftb_%.3d.png'%(i, angle)), np.array(image_rotated_ftb))
				io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_ftb_%.3d.png'%(i, angle)), np.array(label_rotated_ftb))

			for crop_len in [40, 60, 80, 100]:
				x_alignment = random.randint(-crop_len+1,crop_len-1)
				y_alignment = random.randint(-crop_len+1,crop_len-1)
				image_zoomed, label_zoomed = aug_zoom(np.array(image), np.array(label), crop_len, x_alignment, y_alignment)
				io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_z%.3d.png'%(i, crop_len)), np.array(image_zoomed))
				io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_z%.3d.png'%(i, crop_len)), np.array(label_zoomed))
			

		train_images = os.listdir(os.path.join(aug_path, 'images'))
		train_path= aug_path
	else:
		train_images = os.listdir(os.path.join(train_path, 'images'))


def aug_rot_zoom(x_train, y_train, degree):
	'''
	Augments the training data by rotating the images arbitrarily, then zooming.
	'''
	crop_len = 60

	rotated_img = ndimage.rotate(x_train, angle=degree, order=1, reshape=False, axes=(0,1))
	cropped_img = rotated_img[crop_len:-crop_len,crop_len:-crop_len]
	x_rot = cv2.resize(cropped_img, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)

	rotated_img = ndimage.rotate(y_train, angle=degree, order=1, reshape=False, axes=(0,1))
	cropped_img = rotated_img[crop_len:-crop_len,crop_len:-crop_len]
	y_rot = cv2.resize(cropped_img, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)

	return x_rot, y_rot

def aug_zoom(x_train, y_train, crop_len, x_alignment, y_alignment):
	'''
	Zoom for image with the given crop_len
	'''
	x_zoom = ndimage.rotate(x_train, angle=0, order=1, reshape=False, axes=(0,1))
	x_zoom = x_zoom[(crop_len+x_alignment):(x_alignment-crop_len),(crop_len+y_alignment):(y_alignment-crop_len)]
	x_zoom = cv2.resize(x_zoom, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)

	y_zoom = ndimage.rotate(y_train, angle=0, order=1, reshape=False, axes=(0,1))
	y_zoom = y_zoom[(crop_len+x_alignment):(x_alignment-crop_len),(crop_len+y_alignment):(y_alignment-crop_len)]
	y_zoom = cv2.resize(y_zoom, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)

	return x_zoom, y_zoom


def generate_train(train_path, batch_size):
	while True:
		ls= os.listdir(os.path.join(train_path, "images"))
		selected_ls= np.random.choice(ls, batch_size)
		train_data=[]
		train_labels= []

		
		for n in (selected_ls):
			
			img = io.imread(os.path.join(train_path,"images",n))
			img = img / 255

			train_data.append(img)
			label= io.imread(os.path.join(train_path,"groundtruth",n))
			label = label/255
			label[label > 0.5] = 1
			label[label <= 0.5] = 0
			label = np.expand_dims(label, axis=2)

			train_labels.append(label)
		yield (np.asarray((train_data)), np.asarray((train_labels)))

def generate_valid(valid_path, batch_size):
	while True:
		ls= os.listdir(os.path.join(valid_path, "images"))
		selected_ls= np.random.choice(ls, batch_size)
		valid_data=[]
		valid_labels= []
		for n in (selected_ls):
			
			img = io.imread(os.path.join(valid_path,"images",n))
			img = img / 255

			valid_data.append(img)

			label= io.imread(os.path.join(valid_path,"groundtruth",n))
			label = label/255
			label[label > 0.5] = 1
			label[label <= 0.5] = 0
			label = np.expand_dims(label, axis=2)

			valid_labels.append(label)
		yield (np.asarray((valid_data)), np.asarray((valid_labels)))

def generate_test(test_path):

	for i in os.listdir(test_path):
		print(i) #just in case, if the random seed does not work
		img = io.imread(os.path.join(test_path,i))
		img = img / 255
		img = np.reshape(img,(1,)+img.shape)
		
		yield img


