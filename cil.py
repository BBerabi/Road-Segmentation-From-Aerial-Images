import os
import keras
import argparse
import numpy as np
from auxiliary import load_data, download_data, create_submission
from auxiliary import pred_overlap, pred_resize, hold_out_validation
from auxiliary import aug_flip, aug_rot_zoom, aug_crop_zoom, aug_rot_full
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Conv2D, MaxPooling2D, Conv2DTranspose, Input, Add
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, TensorBoard

os.environ['KAGGLE_USERNAME'] = ""
os.environ['KAGGLE_KEY'] = ""
FNULL = open(os.devnull, 'w')

argparser = argparse.ArgumentParser()
argparser.add_argument('--verbose', action='store', dest='verbose', help='verbosity of the script', default=True, type=bool)
argparser.add_argument('--augment', action='store', dest='augment', help='augment the training data', default=True, type=bool)
argparser.add_argument('--batch-size', action='store', dest='batch_size', help='batch size for processing the samples', default=16, type=int)
argparser.add_argument('--early-patience', action='store', dest='early_patience', help='patience for early stopping', default=25, type=int)
argparser.add_argument('--epochs', action='store', dest='epochs', help='number of epochs', default=100, type=int)
argparser.add_argument('--valid-split', action='store', dest='valid_split', help='percentage of validation examples', default=0.1, type=float)
#argparser.add_argument('--tensor-log', action='store', dest='tensor_log', help='directory for tensorboard log', default='logfiles')
argparser.add_argument('--lr', action='store', dest='lr', help='learning rate of optimizer', default=-4, type=int)


args = argparser.parse_args()
verbose = args.verbose

def augment_train(x_train, y_train):
	if verbose:
		print(x_train.shape[0], 'original training examples')

	# Symmetry
	x_flip, y_flip = aug_flip(x_train, y_train)
	x_train = np.concatenate((x_train, x_flip), axis=0)
	y_train = np.concatenate((y_train, y_flip), axis=0)
	if verbose:
		print(x_flip.shape[0], 'flipped training examples')

	# Rotation
	x_rot, y_rot = aug_rot_full(x_train, y_train)
	x_aug, y_aug = aug_rot_zoom(x_train, y_train, trials=1)
	if verbose:
		print(x_rot.shape[0], 'rotated training examples')
		print(x_aug.shape[0], 'rotated and zoomed training examples')

	x_train = np.concatenate((x_train, x_rot, x_aug), axis=0)
	y_train = np.concatenate((y_train, y_rot, y_aug), axis=0)
	if verbose:
		print(x_train.shape[0], 'training examples in total')
		print('Augmented the training data...')

	return x_train, y_train

def baseline_model():
	'''
	Constructs a baseline model.
	'''
	inputs = Input(shape=x_train.shape[1:])

	x = Conv2D(256, (3, 3), padding='same')(inputs)
	x = Activation('relu')(x)
	x = Conv2D(256, (3, 3))(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(0.2)(x)

	x = Conv2D(128, (3, 3), padding='same')(x)
	x = Activation('relu')(x)
	x = Conv2D(128, (3, 3))(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(0.2)(x)

	x = Conv2D(64, (3, 3), padding='same')(x)
	x = Activation('relu')(x)
	x = Conv2DTranspose(64, (3,3), strides=(2,2))(x)
	x = Activation('relu')(x)
	x = Conv2DTranspose(32, (3,3), strides=(1,1))(x)
	x = Dropout(0.2)(x)

	x = Conv2D(16, (3, 3), padding='same')(x)
	x = Activation('relu')(x)
	x = Conv2DTranspose(4, (3,3), strides=(2,2), padding='same')(x)
	x = Activation('relu')(x)
	x = Conv2DTranspose(1, (3,3), strides=(1,1))(x)
	x = Dropout(0.2)(x)

	predictions = Activation('sigmoid')(x)
	model = Model(inputs=inputs, outputs=predictions)

	return model

def fcn_model():
	'''
	Constructs a model based on Hofmann's paper
	'''

	#Output: (3,400,400)
	inputs = Input(shape=x_train.shape[1:])

	#Output: (64,400,400) for this block
	x = Conv2D(64, (15, 15), padding='same')(inputs)
	x = Activation('relu')(x)
	x = Conv2D(64, (15, 15), padding='same')(x)
	x = Activation('relu')(x)

	#Output from first 2 convolutions (will be used at the end)
	#out_first = Conv2D(3, (3, 3), padding='same')(x)

	#Output: (64,200,200)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	#INTERMEDIARY OUTPUT TO BE FUSED with size (3,200,200)
	out_pool1 = Conv2D(3, (11, 11), padding='same')(x)

	#Output: (128,200,200) for this block
	x = Conv2D(128, (11, 11), padding='same')(x)
	x = Activation('relu')(x)
	x = Conv2D(128, (11, 11), padding='same')(x)
	x = Activation('relu')(x)

	#Output: (128,100,100)
	x = MaxPooling2D(pool_size=(2,2))(x)

	#INTERMEDIARY OUTPUT TO BE FUSED with size (3,100,100)
	out_pool2 = Conv2D(3, (7,7), padding='same')(x)		

	#Output: (256,100,100) for this block
	x = Conv2D(256, (7, 7), padding='same')(x)
	x = Activation('relu')(x)
	x = Conv2D(256, (7, 7), padding='same')(x)
	x = Activation('relu')(x)
	x = Conv2D(256, (7, 7), padding='same')(x)
	x = Activation('relu')(x)

	#Output: (256,50,50)
	x = MaxPooling2D(pool_size=(2,2))(x)

	#INTERMEDIARY OUTPUT TO BE FUSED with size (3,50,50)
	out_pool3 = Conv2D(3, (5, 5), padding='same')(x)		

	#Output: (512,50,50) for this block
	x = Conv2D(512, (5, 5), padding='same')(x)
	x = Activation('relu')(x)
	x = Conv2D(512, (5, 5), padding='same')(x)
	x = Activation('relu')(x)
	x = Conv2D(512, (5, 5), padding='same')(x)
	x = Activation('relu')(x)

	#Output: (512,25,25)
	x = MaxPooling2D(pool_size=(2,2))(x)

	#INTERMEDIARY OUTPUT TO BE FUSED with size (3,25,25)
	out_pool4 = Conv2D(3, (3,3), padding='same')(x)

	#Output: (512,25,25) for this block
	x = Conv2D(512, (3, 3), padding='same')(x)
	x = Activation('relu')(x)
	x = Conv2D(512, (3, 3), padding='same')(x)
	x = Activation('relu')(x)
	x = Conv2D(512, (3, 3), padding='same')(x)
	x = Activation('relu')(x)

	#Output: (512,12,12)
	x = MaxPooling2D(pool_size=(2,2))(x)

	#Output: (4096,10,10) for this block
	x = Conv2D(4096, (3, 3), padding='valid')(x)
	x = Activation('relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(4096, (3, 3), padding='same')(x)
	x = Activation('relu')(x)
	x = Dropout(0.5)(x)

	#Output: (3,10,10)
	x = Conv2D(3, (3, 3), padding='same')(x)
	x = Activation('relu')(x)

	#Output: (3,12,12) ? 
	x = Conv2DTranspose(3, (3,3), strides=(1,1))(x)
	x = Activation('relu')(x)

	#Output: (3,25,25) ? 
	x = Conv2DTranspose(3, (3,3), strides=(2,2))(x)
	x = Add()([out_pool4, x])
	x = Activation('relu')(x)
	#x = Conv2D(3, (1,1), padding='same')(x)
	x = Conv2D(3, (3,3), padding='same')(x)
	x = Activation('relu')(x)

	#Output (3,50,50) ? 
	x = Conv2DTranspose(3, (3,3), strides=(2,2), padding='same')(x)
	x = Add()([out_pool3, x])
	x = Activation('relu')(x)
	#x = Conv2D(3, (1,1), padding='same')(x)
	x = Conv2D(3, (3,3), padding='same')(x)
	x = Activation('relu')(x)

	#Output (3,100,100) ? 
	x = Conv2DTranspose(3, (3,3), strides=(2,2), padding='same')(x)
	x = Add()([out_pool2, x])
	x = Activation('relu')(x)
	#x = Conv2D(3, (1,1), padding='same')(x)
	x = Conv2D(3, (3,3), padding='same')(x)
	x = Activation('relu')(x)

	"""
	#First trial
	#Output (1,400,400) ? 
	x = Conv2DTranspose(1, (1,1), strides=(4,4), padding='same')(x)

	"""

	#Second trial
	#Output (3,200,200) ? 
	x = Conv2DTranspose(3, (3,3), strides=(2,2), padding='same')(x)
	x = Add()([out_pool1, x])
	x = Activation('relu')(x)
	#x = Conv2D(3, (1,1), padding='same')(x)
	x = Conv2D(3, (3,3), padding='same')(x)
	x = Activation('relu')(x)

	x = Conv2DTranspose(3, (3,3), strides=(2,2), padding='same')(x)
	#x = Add()([out_first,x])
	#x = Activation('relu')(x)
	#x = Conv2D(3, (3,3), padding='same')(x)
	#x = Activation('relu')(x)
	x = Conv2D(1, (3,3), padding='same')(x)


	#For Now
	predictions = Activation('sigmoid')(x)

	model = Model(inputs=inputs, outputs=predictions)

	return model


path_train = './training/'
path_test = './test_images/'
path_pred = './pred_ims/batch'+str(args.batch_size)+'_lr'+str(args.lr)+'_conv15_full-1_gpuT'
path_out = './outdir/batch'+str(args.batch_size)+'_lr'+str(args.lr)+'_conv15_full-1_gpuT'
download_data(path_train, path_test)
for directory in [path_pred, path_out]:
	if not os.path.exists(directory):
		print(directory, ' not exists')
		os.makedirs(directory)

# Load train and test data
x_train, y_train, x_test = load_data(path_train, path_test)
dim_test = x_test.shape[1]
dim_train = x_train.shape[1]
if verbose:
	print('Loaded the data...')

# Hold-out validation
if args.valid_split > 0:
	x_train, y_train, x_valid, y_valid = hold_out_validation(x_train, y_train, valid_split=args.valid_split)

# Augment the data
if args.augment:
	x_train, y_train = augment_train(x_train, y_train)

# Build and compile the model
model = fcn_model()
#opt = keras.optimizers.Adam(0.001)
opt = keras.optimizers.SGD(lr=10**args.lr, momentum=0.9, decay=0.06)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()


# Set callbacks
file_bestval_cp = path_out + 'bestval.h5'
file_periodic_cp = path_out + 'periodic-{epoch:02d}.h5'
bestval_cp = ModelCheckpoint(file_bestval_cp, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
periodic_cp = ModelCheckpoint(file_periodic_cp, monitor='val_loss', verbose=1, save_best_only=False, mode='auto', period=5)
early = EarlyStopping(monitor="val_acc", mode="max", patience=args.early_patience, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=15, verbose=2)

tensor_logdir = 'logfiles/batch'+str(args.batch_size)+'_lr'+str(args.lr)+'_conv15_full-1_gpuT'
tensorboard = TensorBoard(log_dir=tensor_logdir, histogram_freq=1, batch_size=args.batch_size, write_graph=True, write_grads=True)
callbacks_list = [periodic_cp]
if args.valid_split > 0:
	callbacks_list.append(early)
	callbacks_list.append(bestval_cp)
	callbacks_list.append(redonplat)
	callbacks_list.append(tensorboard)
if verbose:
	print('Compiled the model...')

# Train the model
if args.epochs > 0:
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
	model.load_weights(file_bestval_cp)

# Predict on test data
test_pred_resize = pred_resize(model, x_test, dim_train, dim_test, args.batch_size)
np.save(path_out + 'test_pred_resize.npy', test_pred_resize)

# Create submission
sub_fname = path_out + 'submission_resize.csv'
create_submission(test_pred_resize, path_test, path_pred, sub_fname=sub_fname)
if verbose:
	print('Created the submission file...')

# Currently not used:
# test_pred_overlap = pred_overlap(model, x_test, dim_train, dim_test, args.batch_size)
# np.save(path_out + 'test_pred_overlap.npy', test_pred_overlap)
# sub_fname = path_out + 'submission_overlap.csv'
# create_submission(test_pred_overlap, path_test, path_pred, sub_fname=sub_fname)