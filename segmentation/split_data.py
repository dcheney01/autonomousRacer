import glob
import shutil
import splitfolders
import os

from tqdm import tqdm

#loop through the folders in the images directory and split the data for each folder
folders = os.listdir('./resized_tmp/images/')
print(folders)

#put all of the images into a single folder
for folder in folders:
    input_fold_path = f'./resized_tmp/images/{folder}/'
    split_data_path = f'./resized_tmp/split_images/{folder}/'
    splitfolders.ratio(input_fold_path, output=split_data_path,
        seed=1337, ratio=(.8, .2), group_prefix=None, move=False)
    
#loop through the folders in the labels directory and split the data for each folder
folders = os.listdir('./resized_tmp/labels/')
print(folders)

#put all of the labels into a single folder
for folder in folders:
    input_fold_path = f'./resized_tmp/labels/{folder}/'
    split_data_path = f'./resized_tmp/split_labels/{folder}/'
    splitfolders.ratio(input_fold_path, output=split_data_path,
        seed=1337, ratio=(.8, .2), group_prefix=None, move=False)
    


#copy all of the images from the train folders to the train folder in data
train_images = glob.glob('./resized_tmp/split_images/*/train/drivable_area/*.jpg')
val_images = glob.glob('./resized_tmp/split_images/*/val/drivable_area/*.jpg')
train_labels = glob.glob('./resized_tmp/split_labels/*/train/drivable_area/*.txt')
val_labels = glob.glob('./resized_tmp/split_labels/*/val/drivable_area/*.txt')
    
#move train images to /data/train/images and train labels to /data/train/labels
os.makedirs('./data/images/train', exist_ok=True)
os.makedirs('./data/images/val', exist_ok=True)
os.makedirs('./data/labels/train', exist_ok=True)
os.makedirs('./data/labels/val', exist_ok=True)

for image in tqdm(train_images):
    shutil.copy(image, './data/images/train')

for label in tqdm(train_labels):
    shutil.copy(label, './data/labels/train')

for image in tqdm(val_images):
    shutil.copy(image, './data/images/val')

for label in tqdm(val_labels):
    shutil.copy(label, './data/labels/val')

    

