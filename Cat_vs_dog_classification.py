import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense,Flatten
#from tensorflow.keras.optimaization import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import shutil
import random
import glob
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter(action="ignore",category=FutureWarning)

physical_device=tf.config.expermental.list_physical_devices("GPU")
print("Num GPUs Available:",len(physical_device))
tf.config.experimental.set_memory_growth(physical_device[0])

#!curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip

#!unzip -q kagglecatsanddogs_5340.zip
#!ls

import os

#orgnize the data into train, valid , test dirs
os.chdir("data/dogs-vs-cats")
if os.path.isdir("train/dog") is False:
  os.makedirs("train/dog")
  os.makedirs("train/cat")
  os.makedirs("valid/dog")
  os.makedirs("valid/cat")
  os.makedirs("test/dog")
  os.makedirs("test/cat")
  for c in random.sample(glob.glob("cat"),500):
    shutil.move(c,"train/cat")
  for c in random.sample(glob.glob("dog"),500):
    shutil.move(c,"train/dog")
  for c in random.sample(glob.glob("cat"),100):
    shutil.move(c,"valid/cat")
  for c in random.sample(glob.glob("cat"),100):
    shutil.move(c,"valid/dog")
  for c in random.sample(glob.glob("cat"),50):
    shutil.move(c,"test/cat")
  for c in random.sample(glob.glob("cat"),50):
    shutil.move(c,"test/dog")
os.chdir("../../")          


train_path="data/dogs-vs-cats/train"
valid_path="data/dogs-vs-cats/valid"
test_path="data/dogs-vs-cats/test"
train_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
     .flow_from_directory(directory=train_path,target_size=(224,224),classes=["cat","dog"],batch_size=10)
valid_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
     .flow_from_directory(directory=valid_path,target_size=(224,224),classes=["cat","dog"],batch_size=10)
test_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
     .flow_from_directory(directory=test_path,target_size=(224,224),classes=["cat","dog"],batch_size=10,shuffle=False)          


assert train_batches.n==1000
assert valid_batches.n==200
assert test_batches.n==100
assert train_batches.num_calsses==valid_batches.num_calsses==test_batches.num_calsses

num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)


image_size=(180,180)
batch_size=(128)
train_ds, val_ds=tf.keras.utils.image_dataset_from_directory(
    directory="PetImages",
    batch_size=batch_size,
    image_size=image_size,
    seed=1333,
    validation_split=0.2,
    subset="both"
)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")


data_augmentation=tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),]
)



plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")


augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y))


# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)

epochs = 100

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

img = keras.utils.load_img(
    "PetImages/Cat/6779.jpg", target_size=image_size
)
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(predictions[0])
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")


assert test_batches.n==100

cm=confusion_matrix(y_true=tesl_batches.classes,y_pred=np.argmax(predictions,axis=-1))

# Confusion Matrix plotting function
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Observación')
    plt.xlabel('Predicción')

test_batches.class_indices
cm_plot_labels=["cat","dog"]
plot_confusion_matrix(cm=cm,classes=cm_plot_labels,title="Confusion matrix")