import os
import keras
import argparse
import numpy as np
from auxiliary import load_data, download_data, create_submission
from auxiliary import pred_overlap, pred_resize, hold_out_validation
from auxiliary import aug_flip, aug_rot_zoom, aug_crop_zoom, aug_rot_full
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Conv2D, MaxPooling2D, Conv2DTranspose, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

os.environ['KAGGLE_USERNAME'] = ""
os.environ['KAGGLE_KEY'] = ""
FNULL = open(os.devnull, 'w')

argparser = argparse.ArgumentParser()
argparser.add_argument('--verbose', action='store', dest='verbose', help='verbosity of the script', default=True, type=bool)
argparser.add_argument('--augment', action='store', dest='augment', help='augment the training data', default=False, type=bool)
argparser.add_argument('--batch-size', action='store', dest='batch_size', help='batch size for processing the samples', default=4, type=int)
argparser.add_argument('--early-patience', action='store', dest='early_patience', help='patience for early stopping', default=25, type=int)
argparser.add_argument('--epochs', action='store', dest='epochs', help='number of epochs', default=0, type=int)
argparser.add_argument('--valid-split', action='store', dest='valid_split', help='percentage of validation examples', default=0.0, type=float)
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

path_train = './training/'
path_test = './test_images/'
path_pred = './pred_ims/'
path_out = './outdir/'
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
model = baseline_model()
opt = keras.optimizers.Adam(0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

# Set callbacks
file_bestval_cp = path_out + 'baseline_bestval.h5'
file_periodic_cp = path_out + 'baseline_periodic-{epoch:02d}.h5'
bestval_cp = ModelCheckpoint(file_bestval_cp, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
periodic_cp = ModelCheckpoint(file_periodic_cp, monitor='val_loss', verbose=1, save_best_only=False, mode='auto', period=5)
early = EarlyStopping(monitor="val_acc", mode="max", patience=args.early_patience, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=15, verbose=2)
callbacks_list = [periodic_cp]
if args.valid_split > 0:
	callbacks_list.append(early)
	callbacks_list.append(bestval_cp)
	callbacks_list.append(redonplat)
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
model.load_weights('./temp-outdir/baseline_periodic-60.h5')

if args.valid_split > 0:
	model.load_weights(file_bestval_cp)

# Predict on test data
# test_pred_overlap = pred_overlap(model, x_test, dim_train, dim_test, args.batch_size)
# np.save(path_out + 'test_pred_overlap.npy', test_pred_overlap)
test_pred_resize = pred_resize(model, x_test, dim_train, dim_test, args.batch_size)
np.save(path_out + 'test_pred_resize.npy', test_pred_resize)

# Create submission
# sub_fname = path_out + 'submission_overlap.csv'
# create_submission(test_pred_overlap, path_test, path_pred, sub_fname=sub_fname)
sub_fname = path_out + 'submission_resize.csv'
create_submission(test_pred_resize, path_test, path_pred, sub_fname=sub_fname)
if verbose:
	print('Created the submission file...')

# Send submission file
# submit_solution(fname=submission_filename, message='test')
