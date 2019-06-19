
import numpy as np
import os
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from preprocess import create_data_dirs, generate_train, generate_valid, generate_test
from model import unet
from mask_to_submission import make_submission
from metrics import save_result, dice_loss, dice_coeff, bce_dice_loss
np.random.seed(42)

NUM_EPOCH = 15
NUM_TRAINING_STEP = 100
NUM_VALIDATION_STEP = 80
TEST_SIZE = 94
BATCH_SIZE= 2
train_path = os.path.join("data", "training")
train_aug_path = os.path.join("data", "aug_train")
val_path= os.path.join("data", "validation")
test_path = os.path.join("data", "test_images")

submission_path = "subs"
weight_path = "weights"


if not os.path.exists(val_path):
    print("Creating validation directory because there weren't any.")
    #set here if you want to do augmentation folder, you can sen augmentations in preprocess.py
    #if you set build_augs_folder=False, you'll only train with the original data
    create_data_dirs(train_path, val_ratio=0.2, build_augs_folder= True)
else:
    print("All directories were already there.")


# Build generator for training and validation set
#change train path if you want to use the augmentations
train_generator= generate_train(train_aug_path, batch_size=BATCH_SIZE)
val_generator= generate_valid(val_path, batch_size=BATCH_SIZE)


#start building model
print("Build and train started.")

model = unet(n_filter=32, activation='elu', dropout_rate=0.2, loss=dice_loss)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=9, verbose=1, min_delta=1e-4),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4),
    ModelCheckpoint(os.path.join(weight_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss', save_best_only=True, save_weights_only=False,verbose=1)
]
#training
hist = model.fit_generator(generator=train_generator, steps_per_epoch=NUM_TRAINING_STEP,
                                    validation_data=val_generator, validation_steps=NUM_VALIDATION_STEP,
                                    epochs=NUM_EPOCH, callbacks=callbacks)



print("Predicting now.")
test_generator= generate_test(test_path)
result = model.predict_generator(test_generator, TEST_SIZE, verbose=1)


#set to true if you want to save your test images
if True:
    predict_path= "preds"
    save_result(predict_path, result) #this will save them in the order of reading them, also printing that to see


print("Making subs.")
#this method is added to mask_to_submission.py
make_submission(predict_path, test_size=TEST_SIZE, submission_filename=os.path.join(submission_path, "submission.csv"))
