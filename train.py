
import numpy as np
import os
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from preprocess import create_data_dirs, generate_train, generate_valid, generate_test
from model import unet
from mask_to_submission import make_submission
from metrics import save_result, dice_loss, dice_coeff, bce_dice_loss, f1
from keras.optimizers import Adam
from keras.models import load_model
np.random.seed(42)

NUM_EPOCH = 40
BATCH_SIZE= 32
#Training step is found by #of images in augmented training data divided by batch size.
#Still the training step is hard-coded here because of the changes done on augmented training data.
NUM_TRAINING_STEP = 1000
#Below values are hard-coded since they will never change.
NUM_VALIDATION_STEP = 80
TEST_SIZE = 94

#The paths to read images are given below.
train_path = os.path.join("data", "training")
train_aug_path = os.path.join("data", "aug_train")
val_path= os.path.join("data", "validation")
test_path = os.path.join("data", "test_images")

#Those are the paths to write submission and weights of the network model.
submission_path = "subs"
weight_path = "weights"

#If the validation path does not exist,
#It means that preprocess is not done yet. create_data_dirs function is called from preprocess.py
if not os.path.exists(val_path):
    print("Creating validation directory because there weren't any.")
    #set here if you want to do augmentation folder, you can have augmentations in preprocess.py
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
#Below info is hard-coded just to keep track of the experiments
print("Augmentation: rot and zoom. Dropout rate = 0.2, Learning rate = 1e-5, num_epoch=40, num_steps=1000, batch_size=32")

model = unet(n_filter=32, activation='elu', dropout_rate=0.2, loss=dice_loss, optimizer = Adam(lr=1e-5))
#Train from where it is left (Uncomment below and choose corresponding weight for it)
#Sometimes, the allocated time is exceeded on Leonhard, therefore, loading the model can become necessary.
#model = load_model(os.path.join(weight_path,"weights.12-0.18.hdf5"), custom_objects={"dice_loss": dice_loss, "f1": f1})
callbacks = [
    EarlyStopping(monitor='val_loss', patience=9, verbose=1, min_delta=1e-4),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1),
    ModelCheckpoint(os.path.join(weight_path, 'weights.aug-{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss', save_best_only=True, save_weights_only=False,verbose=1)
]
#Start training

hist = model.fit_generator(generator=train_generator, steps_per_epoch=NUM_TRAINING_STEP,
                                    validation_data=val_generator, validation_steps=NUM_VALIDATION_STEP,
                                    epochs=NUM_EPOCH, callbacks=callbacks)


print("Predicting now.")
test_generator= generate_test(test_path)
#Start the prediction step
result = model.predict_generator(test_generator, TEST_SIZE, verbose=1)


#set to true if you want to save your test images
if True:
    predict_path= "preds"
    save_result(predict_path, result) #this will save them in the order of reading them, also printing that to see


print("Making subs.")
#this method is added to mask_to_submission.py
make_submission(predict_path, test_size=TEST_SIZE, submission_filename=os.path.join(submission_path, "submission.csv"))
