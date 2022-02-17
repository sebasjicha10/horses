import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd

path_inception = f"{getcwd()}/.../tmp2/inception_v3_weights_tf_kernels_notop.h5"

# Import the inception model
from tensorflow.keras.application.inception_v3 import InceptionV3

# Create an instance of the inception model
local_weights_file = path_inception

pre_trained_model = InceptionV3(
    input_shape = (150, 150, 3),
    include_top = False,
    weights = None
)

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
    layer.trainable = False

# Print the model summary
pre_trained_model.summary()

last_layer = pre_trained_model.get_layer("mixed7")
print("Last layer output shape: ", last_layer.output_shape)
last_output = last_layer.output

# Callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get("val_accuracy")>0.97):
            print("\nReached 97.0% accuracy so cancelling training!")
            self.model.stop_training = True
    
# Model
from tensorflow.keras.optimizers import RMSprop

# Flatten the inception output layer
x = layers.Flatten()(last_output)
# Add fully connected layer with 1,024 hidden units
x = layers.Dense(1024, activation="relu")(x)
# Add Dropout Layer
x = layers.Dropout(0.2)(x)
# Add an output sigmoid layer
x = layers.Dense(1, activation="sigmoid")(x)

model = Model(pre_trained_model.input, x)

model.compile(
    optimizer = RMSprop(learning_rate=0.0001),
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)

model.summary()

# Get the dataset
path_horse_or_human = f"{getcwd()}/../tmp2/horse-or-human.zip"

path_validation_horse_or_human = f"{getcwd()}/../tmp2/validation-horse-or-human.zip"
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os 
import zipfile
import shutil

shutil.rmtree("/tmp")
local_zip = path_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("/tmp/training")
zip_ref.close()

local_zip = path_validation_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("/tmp/validation")
zip_ref.close()

# Difine directores
train_dir = "/tmp/training"
validation_dir = "tmp/validation"

train_horses_dir = "/tmp/training/horses/"
train_humans_dir = "/tmp/training/humans/"
validation_horses_dir = "/tmp/validation/horses"
validation_humans_dir = "/tmp/validation/humans"

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print(len(train_horses_fnames))
print(len(train_humans_fnames))
print(len(validation_horses_fnames))
print(len(validation_humans_fnames))

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale = 1/255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(rescale = 1/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size = 20,
    class_mode = "binary",
    target_size = (150, 150)
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    batch_size = 20,
    class_mode = "binary",
    target_size = (150, 150)
)

callbacks = myCallback()

history = model.fit_generator(
    train_generator,
    validation_data = validation_generator,
    steps_per_epoch = 52,
    epochs = 3,
    validation_steps = 13,
    verbose = 2,
    callbacks=[callbacks]
)

# Plot
%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))

plt.plot(epochs, acc, "r", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend(loc=0)
plt.figure()

plt.show()