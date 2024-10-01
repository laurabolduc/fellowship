# Laura Bolduc
# Summer Fellowship 2022
# Training the data without data augmentation or transfer learning

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

train_gen = keras.preprocessing.image.ImageDataGenerator(
	rescale=1/255
).flow_from_directory("sets/train/",
	target_size=(200, 450), 
	batch_size = 32,
	color_mode="rgb",
	class_mode="categorical"
)

val_gen = keras.preprocessing.image.ImageDataGenerator(
	rescale=1/255
).flow_from_directory("sets/val/",
	target_size=(200, 450),
	batch_size = 32, 
	color_mode="rgb",
	class_mode="categorical"
)

test_gen = keras.preprocessing.image.ImageDataGenerator(
	rescale=1/255
).flow_from_directory("sets/test/",
	target_size=(200, 450),
	batch_size = 32,
	color_mode="rgb",
	class_mode="categorical"
)

# configure the network
network = keras.models.Sequential([
	keras.Input(shape=(200, 450, 3)),
	keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
	keras.layers.MaxPooling2D(),
	keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
	keras.layers.MaxPooling2D(),
	keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
	keras.layers.MaxPooling2D(),
	keras.layers.Flatten(),
	keras.layers.Dense(22, activation="softmax") # 22 is the number of directories, would be better to havbe the code count this
])

# train the network
network.compile(loss = "categorical_crossentropy", optimizer="rmsprop", metrics = ["accuracy"])
history = network.fit(train_gen, epochs=5, validation_data=val_gen)

plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("curve.png")

# test the network
network.evaluate(test_gen)


