# Laura Bolduc
# Summer Fellowship 2022
# BINARY IMAGE PREPROCESS

# This python file when run, will concatenate series of game camera images (1, 2 or 3 images)
# It will seperate the images based on animal or no animal
# If a species has less than 10 images contaning that animal, the species type will be changed to just "SOMETHINGHERE"
# It will then split the images into the first 80% for training, 10% for validation and last 10% for testing
# The sets directory will be cleared and the directories to save these images to, will be created by the program
# I did use this to preprocess images as my program developed because the end goal of my project was to identify species 

# Open the CSV
import pandas as pd
zoo = pd.read_csv("zooniverse.csv")

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
from numpy import char
import sklearn
import skimage
from skimage import transform, io
from functools import partial
import cv2
import os 
import shutil

# assign 1 to animal and 0 to no animal
bin = []
for spec in species:
    if spec == "NOTHINGHERE":
        bin.append(0)
    else:
        bin.append(1)

# labels for directories
bin_labels = []
for i in bin:
    if i == 0:
        bin_labels.append("no_animal")
    else:
        bin_labels.append("animal")

# clean out train/test/val directories
try:
    shutil.rmtree('./sets/binary/train')
    shutil.rmtree('./sets/binary/val')
    shutil.rmtree('./sets/binary/test')
except IOError as ioe:
    print(ioe)

# Concatenate each events filenames
temp_rows = zoo[["Imj1", "Imj2", "Img3"]]
print(temp_rows)

# fill na filenames
temp_rows.Imj2 = temp_rows.Imj2.fillna(temp_rows.Imj1)
temp_rows.Img3 = temp_rows.Img3.fillna(temp_rows.Imj2) 

x = []
y = []
missing = 0

# create a list of filenames
species = (zoo['choice'])

# Define function to standardize filenames
def standardized(filename):
    filename = filename.upper()
    if ".JPG" not in filename:
        filename += ".JPG"
    return filename

# concatenate images
for i, row in temp_rows[0:len(temp_rows)].iterrows():
    try: 
        merged = np.concatenate([skimage.io.imread(standardized(filename)) for filename in row])
        merged = tf.image.resize(merged, (450, 200))
        x.append(merged)

        if species[i] == 'NOTHINGHERE':
            y.append(0)

        else:
            y.append(1)

    except IOError as ioe:
        print(ioe)

# break into train/val/test sets
from sklearn.model_selection import train_test_split
x_train_, x_test, y_train_, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=species)
x_train, x_val, y_train, y_val = train_test_split(x_train_, y_train_, test_size=0.1, random_state=42, stratify=y_train_)

count = 0

# create a directory for  animal/no animal in train
os.makedirs("sets/binary/train/no_animal/")
os.makedirs("sets/binary/train/animal/")

# create a directory for animal/no_animal in val
os.makedirs("sets/binary/val/no_animal/")
os.makedirs("sets/binary/val/animal/")

# create a directory for animal/no_animal in test
os.makedirs("sets/binary/test/no_animal/")
os.makedirs("sets/binary/test/animal/")

# print lengths of xtrain, train
print(len(x_train), len(y_train))
print(len(x_val), len(y_val))
print(len(x_test), len(y_test))

count = 0

# add images to their correct directory for train
for i in range(len(x_train)):
    if y[i] == 0:
        new_name = "sets/binary/train/no_animal/" +  "/" + str(count) + ".JPG"
        skimage.io.imsave(new_name, x_train[i])
        count += 1
    else:
        new_name = "sets/binary/train/animal/" + "/" + str(count) + ".JPG"
        skimage.io.imsave(new_name, x_train[i])
        count += 1

# add images to their correct directory for val
for j in range(len(x_val)):
    if y[j] == 0:
        new_name = "sets/binary/val/no_animal/"  + "/" + str(count) + ".JPG"
        skimage.io.imsave(new_name, x_val[j])
        count += 1
    else:
        new_name = "sets/binary/val/animal/" + "/"  + str(count) + ".JPG"
        skimage.io.imsave(new_name, x_val[j])
        count += 1

# add images to their correct directory for val
for k in range(len(x_test)):
    if y[k] == 0:
        new_name = "sets/binary/test/no_animal" + "/" + str(count) + ".JPG"
        skimage.io.imsave(new_name, x_test[k])
        count += 1
    else:
        new_name = "sets/binary/test/animal/" + "/" + str(count) + ".JPG"
        skimage.io.imsave(new_name, x_test[k])
        count += 1 
