
import numpy as np
np.random.seed(42)
import os
import tensorflow as tf
from keras.models import load_model

from preprocess import generate_test
from metrics import save_result
from metrics import f1, dice_loss
from mask_to_submission import make_submission

'''
This scripts is used when user wants to make prediction from already trained models.
In case that multiple weight files are provided into weights array, the predictions are averaged over models. 
One further step could be done to give different weights to models for prediction averaging, but we didn't need it.
'''

TEST_SIZE = 94
test_path = os.path.join("data", "test_images")

predict_path = "preds"
submission_path = "subs"
weight_path = "weights"
#weights = ["weights.11-0.17.hdf5"]
#weights = ["weights.22-0.15.hdf5"]
weights = ["weights.12-0.18.hdf5"]
#to average over multiple weights

results = 0
for w in weights:
	#For the whole models (or weights) provided, the predictions are done seperately and accumulated on 'results'.
    print("loading weights " + w )
    model = load_model(os.path.join(weight_path, w), custom_objects={"dice_loss": dice_loss, "f1": f1})
    print("predicting")
    test_generator = generate_test(test_path)
    results += model.predict_generator(test_generator, TEST_SIZE, verbose=1)
#Accumulated predictions are now averaged to get proper predictions. 
results /= len(weights)
save_result(predict_path, results)
make_submission(predict_path, test_size=TEST_SIZE, submission_filename=os.path.join(submission_path, "submission.csv"))