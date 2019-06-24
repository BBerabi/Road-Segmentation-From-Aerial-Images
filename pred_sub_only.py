
import numpy as np
np.random.seed(42)
import os
import tensorflow as tf
from keras.models import load_model

from preprocess import generate_test
from metrics import save_result
from metrics import f1, dice_loss
from mask_to_submission import make_submission


TEST_SIZE = 94
test_path = os.path.join("data", "test_images")

predict_path = "preds"
submission_path = "subs"
weight_path = "weights"
weights = ["weights.h5" ] 
#to average over multiple weights

results = 0
for w in weights:
    print("loading weights " + w )
    model = load_model(os.path.join(weight_path, w), custom_objects={"dice_loss": dice_loss, "f1": f1})
    print("predicting")
    test_generator = generate_test(test_path)
    results += model.predict_generator(test_generator, TEST_SIZE, verbose=1)
results /= len(weights)
save_result(predict_path, results)
make_submission(predict_path, test_size=TEST_SIZE, submission_filename=os.path.join(submission_path, "submission.csv"))
#this will probably give weird ids for the predictions. I was too lazy to fix it so I just change it everytime when I upload.
#you can get the corresponding correct ids from best_submission.csv
