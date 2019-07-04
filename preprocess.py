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
	#It is also used for aug flag, if you don't want augs set to false

	N= 100 #hard coded since we have a particular number of train images to use, ordered
	val_path= train_path +"/../validation"

	#create validation folders and move data there
	if not os.path.exists(val_path):
		os.makedirs(val_path)
	if not os.path.exists(os.path.join(val_path, 'images')):
		os.makedirs(os.path.join(val_path, 'images'))
	if not os.path.exists(os.path.join(val_path, 'groundtruth')):
		os.makedirs(os.path.join(val_path, 'groundtruth'))                
	
	#Get the train and validation sets by the IDs of images (from 1 to 100)
	train, val = train_test_split(range(1,N+1), test_size=val_ratio, random_state=seed)

	#Images chosen for validation are moved to corresponding folder
	for i in val:
		image = Image.open(os.path.join(train_path, 'images', 'satImage_%.3d.png'%i))
		label = Image.open(os.path.join(train_path, 'groundtruth', 'satImage_%.3d.png'%i))

		io.imsave(os.path.join(val_path, 'images', 'satImage_%.3d.png'%i), np.array(image))
		io.imsave(os.path.join(val_path, 'groundtruth', 'satImage_%.3d.png'%i), np.array(label))

	#Create augmentation folder and put all the augmented images into it	
	if build_augs_folder:
	#each time you change something in augs, (add or subtract stuff), change below path so its
	#saved somewhere else
		aug_path= train_path + "/../aug_train"
		if not os.path.exists(aug_path):
			os.makedirs(aug_path)
			os.makedirs(os.path.join(aug_path, 'images'))
			os.makedirs(os.path.join(aug_path, 'groundtruth'))
		#Augmentation is done only for images in training set
		for i in train:
			#Read the original image and its label
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

			#Rotations of every 5 degrees
			for angle in range(5,360,5):
				#Rotate the original image and its label
				image_rotated, label_rotated = aug_rot(np.array(image), np.array(label), angle)
				io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_%.3d.png'%(i, angle)), np.array(image_rotated))
				io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_%.3d.png'%(i, angle)), np.array(label_rotated))
				#Rotate the horizontally flipped image and its label
				image_rotated_flr, label_rotated_flr = aug_rot(np.array(im_flip_lr), np.array(label_flip_lr), angle)
				io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_flr_%.3d.png'%(i, angle)), np.array(image_rotated_flr))
				io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_flr_%.3d.png'%(i, angle)), np.array(label_rotated_flr))
				#Rotate the vertically flipped image and its label
				image_rotated_ftb, label_rotated_ftb = aug_rot(np.array(im_flip_tb), np.array(label_flip_tb), angle)
				io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_ftb_%.3d.png'%(i, angle)), np.array(image_rotated_ftb))
				io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_ftb_%.3d.png'%(i, angle)), np.array(label_rotated_ftb))

				#For every 30 degrees in angle
				#Zoom in rotated images 
				if angle in range(30,360,30):
					for crop_len in range(10,60,10):
						#After the crop length is chosen (to crop from left,right,up and down of the image),
						#Randomly choose which part of the image will be zoomed in
						#For both x_alignment and y_alignment:
						#The value 0 means the middle of the corresponding dimension
						#Negative value means the shift to the back of middle, pozitive means to the forward of the middle
						x_alignment = random.randint(-crop_len+1,crop_len-1)
						y_alignment = random.randint(-crop_len+1,crop_len-1)

						#Zoom is done for the original image and its label
						image_zoomed, label_zoomed = aug_zoom(np.array(image_rotated), np.array(label_rotated), crop_len, x_alignment, y_alignment)
						io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_%.3d_z%.3d.png'%(i, angle, crop_len)), np.array(image_zoomed))
						io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_%.3d_z%.3d.png'%(i, angle, crop_len)), np.array(label_zoomed))
						#Zoom is done for the horizontally flipped image and its label
						image_zoomed, label_zoomed = aug_zoom(np.array(image_rotated_flr), np.array(label_rotated_flr), crop_len, x_alignment, y_alignment)
						io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_flr_%.3d_z%.3d.png'%(i, angle, crop_len)), np.array(image_zoomed))
						io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_flr_%.3d_z%.3d.png'%(i, angle, crop_len)), np.array(label_zoomed))
						#Zoom is done for the vertically flipped image and its label
						image_zoomed, label_zoomed = aug_zoom(np.array(image_rotated_ftb), np.array(label_rotated_ftb), crop_len, x_alignment, y_alignment)
						io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_ftb_%.3d_z%.3d.png'%(i, angle, crop_len)), np.array(image_zoomed))
						io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_ftb_%.3d_z%.3d.png'%(i, angle, crop_len)), np.array(label_zoomed))

			#Here the zoom is done for non-rotated image and its label as well. 
			for crop_len in range(10,60,10):
				x_alignment = random.randint(-crop_len+1,crop_len-1)
				y_alignment = random.randint(-crop_len+1,crop_len-1)

				#Zoom is done for the original image and its label
				image_zoomed, label_zoomed = aug_zoom(np.array(image), np.array(label), crop_len, x_alignment, y_alignment)
				io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_z%.3d.png'%(i, crop_len)), np.array(image_zoomed))
				io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_z%.3d.png'%(i, crop_len)), np.array(label_zoomed))
				#Zoom is done for the horizontally flipped image and its label
				image_zoomed, label_zoomed = aug_zoom(np.array(im_flip_lr), np.array(label_flip_lr), crop_len, x_alignment, y_alignment)
				io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_flr_z%.3d.png'%(i, crop_len)), np.array(image_zoomed))
				io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_flr_z%.3d.png'%(i, crop_len)), np.array(label_zoomed))
				#Zoom is done for the vertically flipped image and its label
				image_zoomed, label_zoomed = aug_zoom(np.array(im_flip_tb), np.array(label_flip_tb), crop_len, x_alignment, y_alignment)
				io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_ftb_z%.3d.png'%(i, crop_len)), np.array(image_zoomed))
				io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_ftb_z%.3d.png'%(i, crop_len)), np.array(label_zoomed))

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

def aug_rot(x_train, y_train, degree):
	'''
	Augments the training data by rotation: no zoom, black corners are filled with mode:mirror
	'''
	x_rot = ndimage.rotate(x_train, angle=degree, order=1, reshape=False, axes=(0,1), mode='mirror')
	y_rot = ndimage.rotate(y_train, angle=degree, order=1, reshape=False, axes=(0,1), mode='mirror')
	return x_rot, y_rot


def aug_zoom(x_train, y_train, crop_len, x_alignment, y_alignment):
	'''
	Zoom for image with the given crop_len, x_alignment and y_alignment
	'''
	x_zoom = ndimage.rotate(x_train, angle=0, order=1, reshape=False, axes=(0,1))
	x_zoom = x_zoom[(crop_len+x_alignment):(x_alignment-crop_len),(crop_len+y_alignment):(y_alignment-crop_len)]
	x_zoom = cv2.resize(x_zoom, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)

	y_zoom = ndimage.rotate(y_train, angle=0, order=1, reshape=False, axes=(0,1))
	y_zoom = y_zoom[(crop_len+x_alignment):(x_alignment-crop_len),(crop_len+y_alignment):(y_alignment-crop_len)]
	y_zoom = cv2.resize(y_zoom, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)

	return x_zoom, y_zoom


def generate_train(train_path, batch_size):
	'''
	For every step in every epoch, batch of images are provided by this function.
	Training images are normalized before they are provided for training (dividing by 255)
	Labels are provided in binary format.
	'''
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
	'''
	For every epoch, validation images are provided by this function.
	Validation images are normalized before they are provided for training (dividing by 255)
	Labels are provided in binary format.
	'''

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
	'''
	The images to be used for prediction after the training are provided by this function.
	Test images are normalized as well.
	'''

	for i in os.listdir(test_path):
		print(i) #just in case, if the random seed does not work
		img = io.imread(os.path.join(test_path,i))
		img = img / 255
		img = np.reshape(img,(1,)+img.shape)
		
		yield img


