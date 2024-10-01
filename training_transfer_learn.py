# Laura Bolduc
# Summer 2022 Fellowship
# Training the data with transfer learning only
# when used on binary classification, this had the best results

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

train_gen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.vgg16.preprocess_input
).flow_from_directory("sets/train/",
    target_size=(200, 450),
    batch_size = 32,
    color_mode="rgb",
    class_mode="categorical"
)

val_gen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.vgg16.preprocess_input
).flow_from_directory("sets/val/", 
    target_size=(200, 450),
    batch_size = 32,
    color_mode="rgb",
    class_mode="categorical"
)

test_gen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.vgg16.preprocess_input
).flow_from_directory("sets/test/",
    target_size=(200, 450),
    batch_size = 32,
    color_mode="rgb",
    class_mode="categorical"
)

# load pre trained convolutional base
base = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(200, 450, 3))

# add a new dense top 
network = keras.models.Sequential([
    base,
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation = "relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(22, activation = "softmax")
])

# freeze the base
base.trainable = False

# train the top
network.compile(loss = "categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])
history = network.fit(train_gen, epochs=5, validation_data=val_gen)

# unfreeze the base
base.trainable = True

# early stopping
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
network.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = network.fit(train_gen, epochs=100, validation_data=val_gen, callbacks=[early_stopping])

# fine tune the whole network
network.compile(loss = "categorical_crossentropy", optimizer=keras.optimizers.RMSprop(learning_rate=0.00001), metrics = ["accuracy"])
history = network.fit(train_gen, epochs=3, validation_data=val_gen)

# measure accuracy on the test data
network.evaluate(test_gen)
