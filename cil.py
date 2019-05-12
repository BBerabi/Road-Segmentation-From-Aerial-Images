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

from auxiliary import pred_maximum_overlap, pred_minimum_overlap, save_maps

os.environ['KAGGLE_USERNAME'] = ""
os.environ['KAGGLE_KEY'] = ""
FNULL = open(os.devnull, 'w')

argparser = argparse.ArgumentParser()
argparser.add_argument('--verbose', action='store', dest='verbose', help='verbosity of the script', default=True, type=bool)
argparser.add_argument('--train-ae', action='store', dest='train_ae', help='train the autoencoder first', default=True, type=bool)
argparser.add_argument('--augment', action='store', dest='augment', help='augment the training data', default=True, type=bool)
argparser.add_argument('--batch-size', action='store', dest='batch_size', help='batch size for processing the samples', default=16, type=int)
argparser.add_argument('--early-patience', action='store', dest='early_patience', help='patience for early stopping', default=20, type=int)
argparser.add_argument('--epochs', action='store', dest='epochs', help='number of epochs', default=70, type=int)
argparser.add_argument('--valid-split', action='store', dest='valid_split', help='percentage of validation examples', default=0.1, type=float)
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

class SegmentModel:
	def __init__(self):
		self.dropout_rate = 0.3
		self.build()

	def build(self):
		'''
		Constructs a baseline model.
		'''
		inputs = Input(shape=x_train.shape[1:])

		### SHARED ENCODER MODEL

		x = Conv2D(128, (3, 3), padding='same')(inputs)
		x = Activation('relu')(x)
		x = Conv2D(128, (3, 3))(x)
		x = Activation('relu')(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.2)(x)

		x = Conv2D(64, (3, 3), padding='same')(x)
		x = Activation('relu')(x)
		x = Conv2D(64, (3, 3))(x)
		x = Activation('relu')(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		hidden = Dropout(0.2)(x)

		### PREDICTION MODEL

		x = Conv2D(64, (3, 3), padding='same')(hidden)
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

		preds = Activation('sigmoid')(x)

		### AUTOENCODER MODEL

		x = Conv2D(64, (3, 3), padding='same')(hidden)
		x = Activation('relu')(x)
		x = Conv2DTranspose(64, (3,3), strides=(2,2))(x)
		x = Activation('relu')(x)
		x = Conv2DTranspose(32, (3,3), strides=(1,1))(x)
		x = Dropout(0.2)(x)

		x = Conv2D(16, (3, 3), padding='same')(x)
		x = Activation('relu')(x)
		x = Conv2DTranspose(4, (3,3), strides=(2,2), padding='same')(x)
		x = Activation('relu')(x)
		x = Conv2DTranspose(3, (3,3), strides=(1,1))(x)
		x = Dropout(0.2)(x)

		reconst = Activation('linear')(x)

		self.preds = preds
		self.reconst = reconst
		self.inputs = inputs

	def pred_model(self):
		return Model(inputs=self.inputs, outputs=self.preds)
	
	def autoenc_model(self):
		return Model(inputs=self.inputs, outputs=self.reconst)

path_train = './training/'
path_test = './test_images/'
path_pred = './pred_ims/'
path_out = './outdir/'
path_maps = './outdir/maps/'
download_data(path_train, path_test)
for directory in [path_pred, path_out, path_maps]:
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
if args.epochs > 0 and args.augment:
	x_train, y_train = augment_train(x_train, y_train)

# Build and compile the model
model = SegmentModel()
pred_model = model.pred_model()
autoenc_model = model.autoenc_model()
# pred_model = baseline_model()

if args.train_ae:
	enc_epochs = 30
	opt = keras.optimizers.Adam(0.001)
	autoenc_model.compile(loss='mean_squared_error', optimizer=opt)
	# autoenc_model.summary()

	# Set callbacks
	file_bestval_cp = path_out + 'autoenc_bestval.h5'
	file_periodic_cp = path_out + 'autoenc_periodic-{epoch:02d}.h5'
	bestval_cp = ModelCheckpoint(file_bestval_cp, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	periodic_cp = ModelCheckpoint(file_periodic_cp, monitor='val_loss', verbose=1, save_best_only=False, mode='auto', period=5)
	callbacks_list = [periodic_cp]
	if args.valid_split > 0:
		callbacks_list.append(bestval_cp)
	if verbose:
		print('Compiled the autoenc_model...')

	# Train the autoenc_model
	if args.epochs > 0:
		if args.valid_split > 0:
			history = autoenc_model.fit(x_train, x_train,
			          batch_size=args.batch_size,
			          epochs=enc_epochs,
			          validation_data=(x_valid, x_valid),
			          callbacks=callbacks_list,
			          verbose=2,
			          shuffle=True)
		else:
			autoenc_model.fit(x_train, x_train,
			          batch_size=args.batch_size,
			          epochs=enc_epochs,
			          callbacks=callbacks_list,
			          verbose=2,
			          shuffle=True)

opt = keras.optimizers.Adam(0.001)
pred_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# pred_model.summary()

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
	print('Compiled the pred_model...')

# Train the pred_model
if args.epochs > 0:
	if args.valid_split > 0:
		history = pred_model.fit(x_train, y_train,
		          batch_size=args.batch_size,
		          epochs=args.epochs,
		          validation_data=(x_valid, y_valid),
		          callbacks=callbacks_list,
		          verbose=2,
		          shuffle=True)
	else:
		pred_model.fit(x_train, y_train,
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

# Load the checkpoint pred_model
if args.valid_split > 0:
	pred_model.load_weights(file_bestval_cp)

# Predict on test data
test_pred_resize = pred_resize(pred_model, x_test, dim_train, dim_test, args.batch_size)
np.save(path_out + 'test_pred_resize.npy', test_pred_resize)
# Create submission
sub_fname = path_out + 'submission_resize.csv'
create_submission(test_pred_resize, path_test, path_pred, sub_fname=sub_fname)

save_maps(test_pred_resize, path_test, path_maps)

if verbose:
	print('Created the submission file...')
