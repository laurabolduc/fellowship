# Laura Bolduc
# Summer Fellowship 2022
# MULTI IMAGE PREPROCESSING

# This python file when run, will concatenate series of game camera images (1, 2 or 3 images)
# It will seperate the images based on species
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

# Create a list of species labels
species = (zoo['choice'])

# all 12814 rows
print(species)

# each unique species
print(species.unique())

# print how many times each unique species occurs
counts = species.value_counts()

new_name = "SOMETHINGHERE"

for index, value in counts.items():
    print(index, value)
    if value < 10:
       species = species.replace(index, new_name)

counts = species.value_counts()
unique = species.unique()

# clean out train/test/val directories
try:
    shutil.rmtree('./sets/multi/train')
    shutil.rmtree('./sets/multi/val')
    shutil.rmtree('./sets/multi/test')
except IOError as ioe:
    print(ioe)

# Concatenate each events filenames
temp_rows = zoo[["Imj1", "Imj2", "Img3"]]
print(temp_rows)

# fill na filenames
temp_rows.Imj2 = temp_rows.Imj2.fillna(temp_rows.Imj1)
temp_rows.Img3 = temp_rows.Img3.fillna(temp_rows.Imj2) 

# Define function to standardize filenames
def standardized(filename):
    filename = filename.upper()
    if ".JPG" not in filename:
        filename += ".JPG"
    return filename

x = []
y = []

# concatenate images
for i, row in temp_rows[0:len(temp_rows)].iterrows():
    try: 
        merged = np.concatenate([skimage.io.imread(standardized(filename)) for filename in row])
        merged = tf.image.resize(merged, (450, 200))
        x.append(merged)
        y.append(species[i])

    except IOError as ioe:
        print(ioe)

# break into train/val/test sets
from sklearn.model_selection import train_test_split
x_train_, x_test, y_train_, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=species)
x_train, x_val, y_train, y_val = train_test_split(x_train_, y_train_, test_size=0.1, random_state=42, stratify=y_train_)

# create a directory for each animal in train
for i in range(len(unique)):
    os.makedirs("sets/multi/train/" + str(unique[i]))

# create a directory for each animal in val
for i in range(len(unique)):
    os.makedirs("sets/multi/val/" + str(unique[i]))

# create a directory for each animal in test
for i in range(len(unique)):
    os.makedirs("sets/multi/test/" + str(unique[i]))

# print lengths of xtrain, train
print(len(x_train), len(y_train))
print(len(x_val), len(y_val))
print(len(x_test), len(y_test))

count = 0

# add images to their correct directory for train
for i in range(len(x_train)):
    animal = str(y_train[i])
    new_name = "sets/multi/train/" + animal + "/" + str(count) + ".JPG"
    skimage.io.imsave(new_name, x_train[i])
    count += 1

# add images to their correct directory for val
for j in range(len(x_val)):
    animal = str(y_val[j])
    new_name = "sets/multi/val/" + animal + "/" + str(count) + ".JPG"
    skimage.io.imsave(new_name, x_val[j])
    count += 1

# add images to their correct directory for val
for k in range(len(x_test)):
    animal = str(y_test[k])
    new_name = "sets/multi/test/" + animal + "/" + str(count) + ".JPG"
    skimage.io.imsave(new_name, x_test[k])
    count += 1
