import numpy as np
from sklearn.model_selection import train_test_split
import os
import skimage.io as io
from PIL import Image #use for augs
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

    if build_augs_folder:
        #each time you change something in augs, (add or subtract stuff), change below path so its
        #saved somewhere else
        aug_path= train_path + "/../aug_train"
        if not os.path.exists(aug_path):
            os.makedirs(aug_path)
            os.makedirs(os.path.join(aug_path, 'images'))
            os.makedirs(os.path.join(aug_path, 'groundtruth'))

        for i in range(1, N+1):

            image = Image.open(os.path.join(train_path, 'images', 'satImage_%.3d.png'%i))
            label = Image.open(os.path.join(train_path, 'groundtruth', 'satImage_%.3d.png'%i))
            #create horizontally flipped images
            im_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
            io.imsave(os.path.join(aug_path, 'images', 'satImage_%.3d_f.png'%i), np.array(im_flip))

            label_flip = label.transpose(Image.FLIP_LEFT_RIGHT)
            io.imsave(os.path.join(aug_path, 'groundtruth', 'satImage_%.3d_f.png'%i), np.array(label_flip))
        
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

        train_images = os.listdir(os.path.join(aug_path, 'images'))
        train_path= aug_path
    else:
        train_images = os.listdir(os.path.join(train_path, 'images'))

    train, val = train_test_split(train_images, test_size=val_ratio, random_state=seed)

    #create validation folders and move data there
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    if not os.path.exists(os.path.join(val_path, 'images')):
        os.makedirs(os.path.join(val_path, 'images'))
    if not os.path.exists(os.path.join(val_path, 'groundtruth')):
        os.makedirs(os.path.join(val_path, 'groundtruth'))                
    for im in val:
        os.rename(os.path.join(train_path, 'images', im), os.path.join(val_path, 'images', im))
        os.rename(os.path.join(train_path, 'groundtruth', im), os.path.join(val_path, 'groundtruth', im))



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
    ls= os.listdir(test_path)
    np.save("test_indices.npy", ls)
    for i in ls:
        print(i) #just in case, if the random seed does not work
        img = io.imread(os.path.join(test_path,i))
        img = img / 255
        img = np.reshape(img,(1,)+img.shape)
        
        yield img


