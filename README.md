
*Artificial Neural Network * has a input layer, hidden layer and a output layer all these layers compraies a ANN.
image.png

Convectionual Neural Network has a imput layer, conv 2D layer,Max pooling layer,Flatten layer,Hidden layer and a output layer al these layers compraises of a CNN.
image.png

image.png image.png


[ ]
import os

# Define paths to your data directories
train_dir = '/content/drive/MyDrive/carbikes/train'
val_dir = '/content/drive/MyDrive/carbikes/test'

# Check if directories exist
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory {train_dir} does not exist.")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation directory {val_dir} does not exist.")

# List files in each directory
print("Training images:")
print(os.listdir(train_dir))

print("Validation images:")
print(os.listdir(val_dir))

Training images:
['bikes', 'cars']
Validation images:
['bike', 'cars', 'bikes']

[ ]
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Prepare data generators (adjust paths as needed)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,  # Replace with your training data path
    target_size=(200, 200),
    batch_size=32,
    class_mode='sparse'  # Use 'categorical' for one-hot encoded labels
)

# Verify that training images are being loaded
print(f"Found {train_generator.samples} images belonging to {train_generator.num_classes} classes.")

# Ensure validation data is available
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir,  # Replace with your validation data path
    target_size=(200, 200),
    batch_size=32,
    class_mode='sparse'
)

print(f"Found {val_generator.samples} images belonging to {val_generator.num_classes} classes.")

Found 9 images belonging to 2 classes.
Found 9 images belonging to 2 classes.
Found 4 images belonging to 3 classes.
Found 4 images belonging to 3 classes.

[ ]
train_dir = '/content/drive/MyDrive/carbikes/train'
val_dir = '/content/drive/MyDrive/carbikes/test'


[ ]
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to your data directories
train_dir = '/content/drive/MyDrive/carbikes/train'
val_dir = '/content/drive/MyDrive/carbikes/test'

# Create ImageDataGenerator instances
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(200, 200),
    batch_size=32,
    class_mode='sparse'  # Use 'categorical' if labels are one-hot encoded
)

print(f"Found {train_generator.samples} training images belonging to {train_generator.num_classes} classes.")

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(200, 200),
    batch_size=32,
    class_mode='sparse'
)

print(f"Found {val_generator.samples} validation images belonging to {val_generator.num_classes} classes.")

Found 9 images belonging to 2 classes.
Found 9 training images belonging to 2 classes.
Found 4 images belonging to 3 classes.
Found 4 validation images belonging to 3 classes.

[ ]
import os

def count_images(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                count += 1
    return count

train_dir = '/content/drive/MyDrive/carbikes/train'
val_dir = '/content/drive/MyDrive/carbikes/test'

print(f"Training images count: {count_images(train_dir)}")
print(f"Validation images count: {count_images(val_dir)}")

Training images count: 9
Validation images count: 4

[ ]
import os
path = "/content/drive/MyDrive/carbikes"
classes= os.listdir(path)
print(classes)
['train', 'test']

[ ]
from google.colab import drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).

[ ]
from tensorflow.keras.preprocessing import image_dataset_from_directory

base_dir = "/content/drive/MyDrive/carbikes"

# Create datasets
train_datagen = image_dataset_from_directory(base_dir,
image_size=(200,200),
subset='training',
seed = 1,
validation_split=0.1,
       batch_size= 32)

test_datagen = image_dataset_from_directory(base_dir,
image_size=(200,200),
subset='validation',
seed = 1,
validation_split=0.1,
batch_size= 32)
Found 13 files belonging to 2 classes.
Using 12 files for training.
Found 13 files belonging to 2 classes.
Using 1 files for validation.

[ ]
import tensorflow as tf

from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D


model = tf.keras.models.Sequential([
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
layers.MaxPooling2D(2, 2),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D(2, 2),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D(2, 2),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D(2, 2),

layers.Flatten(),
layers.Dense(512, activation='relu'),
layers.BatchNormalization(),
layers.Dense(512, activation='relu'),
layers.Dropout(0.1),
layers.BatchNormalization(),
layers.Dense(512, activation='relu'),
layers.Dropout(0.2),
layers.BatchNormalization(),
layers.Dense(1, activation='sigmoid') #output layer
])

/usr/local/lib/python3.10/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)

[ ]
import tensorflow as tf

from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D


model = tf.keras.models.Sequential([
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
layers.MaxPooling2D(2, 2),


layers.Flatten(),
layers.Dense(512, activation='relu'),

layers.Dense(1, activation='sigmoid') #output layer
])




[ ]
keras.utils.plot_model(
model,
show_shapes=True,
show_dtype=True,
show_layer_activations=True
)



[ ]
model.compile(
loss='binary_crossentropy',
optimizer='adam',
metrics=['accuracy']
)

[ ]
history = model.fit(train_datagen,
epochs=10,
validation_data=test_datagen,
verbose=1)
Epoch 1/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 7s 7s/step - accuracy: 0.5833 - loss: 14.1172 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
Epoch 2/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 14s 14s/step - accuracy: 0.6667 - loss: 27520.1973 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
Epoch 3/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 7s 7s/step - accuracy: 0.6667 - loss: 19819.7812 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
Epoch 4/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 10s 10s/step - accuracy: 0.6667 - loss: 7241.2817 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
Epoch 5/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 7s 7s/step - accuracy: 0.9167 - loss: 49.6250 - val_accuracy: 0.0000e+00 - val_loss: 6695.6733
Epoch 6/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 5s 5s/step - accuracy: 0.3333 - loss: 5105.0957 - val_accuracy: 0.0000e+00 - val_loss: 5529.2915
Epoch 7/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 5s 5s/step - accuracy: 0.3333 - loss: 3664.1824 - val_accuracy: 0.0000e+00 - val_loss: 571.1514
Epoch 8/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 7s 7s/step - accuracy: 0.9167 - loss: 70.0828 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
Epoch 9/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 5s 5s/step - accuracy: 0.8333 - loss: 270.5917 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
Epoch 10/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 6s 6s/step - accuracy: 0.8333 - loss: 650.7632 - val_accuracy: 1.0000 - val_loss: 0.0000e+00

[ ]
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the image
test_image = image.load_img("/content/drive/MyDrive/carbikes/test/cars/rangerover.jpg", target_size=(200, 200))
plt.imshow(test_image)
plt.axis('off')  # Optional: Hide the axis for a cleaner image display
plt.show()

…if result[0][0] > 0.5:  # Assuming the model outputs probabilities
    print("bike")
else:
    print("car")

Colab paid products - Cancel contracts here

