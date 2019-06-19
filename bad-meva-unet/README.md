~cil project road segmentation using aerial images with unet~

this folder should have following subfolders and files:

--data: put the downloaded data under this folder after unzipping training
in the subfolders called "training" and "test_set_images"
after augmentations, one can choose to save the augmented files as well (obviously under training)
valdiation data created in train.py

--weights: use this folder to save weights for models,
the model weights under which was trained for 10 epochs and gives:
loss: 0.1135 - f1: 0.8865 - acc: 0.9562 - val_loss: 0.1408 - val_f1: 0.8592 - val_acc: 0.9538
download from here and put under folder:https://polybox.ethz.ch/index.php/s/bQlJCO1ovp3JzSs

--subs: use this to send submission after prediction is done

--preprocess.py: has generators for train, val and test data, with augmentations in train generator

--model.py: unet implementation

--train.py: trains the model and makes predictions. saves the submission

--pred_sub_only.py: for those who don't want to train but use da weights and rocknroll!!

--mask_to_submission.py: contains helpers for submission

--metrics.py: contains f1 score calc

