########################################################################

# Function to download the CIFAR-10 dataset and load it into memory.

########################################################################

import numpy as np
import pickle
import os
from . import download
from .dataset import one_hot_encoded

########################################################################

# Directory where you want to download and save the dataset.
DATA_PATH = "data/CIFAR-10/"

# URL for the dataset on the internet.
DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

########################################################################
# Constants for image size.
IMG_SIZE = 32  # Width and height of each image.
NUM_CHANNELS = 3  # Number of channels in each image: Red, Green, Blue.
# Length of the image when flattened.
IMG_SIZE_FLAT = IMG_SIZE * IMG_SIZE * NUM_CHANNELS
NUM_CLASSES = 10  # Number of classes.

########################################################################
# Constants to allocate arrays of the correct size.
_NUM_FILES_TRAIN = 5  # Number of files for the training set.
# Number of images per file in the training set.
_IMAGES_PER_FILE = 10000
# Total number of images in the training set.
_NUM_IMAGES_TRAIN = _NUM_FILES_TRAIN * _IMAGES_PER_FILE

########################################################################
# Private functions to download, extract, and load data files.


def _get_file_path(filename=""):
    """
    Returns the full path of a data file for the dataset.

    If filename=="" returns the directory of the files.
    """
    return os.path.join(DATA_PATH, "cifar-10-batches-py/", filename)


def _unpickle(filename):
    """
    Unpickles the given file and returns the data.
    """
    file_path = _get_file_path(filename)
    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    return data


def _convert_images(raw):
    """
    Converts images from the CIFAR-10 format and
    returns a 4D array with the format: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """
    raw_float = np.array(raw, dtype=float) / 255.0
    images = raw_float.reshape([-1, NUM_CHANNELS, IMG_SIZE, IMG_SIZE])
    images = images.transpose([0, 2, 3, 1])
    return images


def _load_data(filename):
    """
    Loads a pickled data file from the CIFAR-10 dataset
    and returns the converted images and class number for each image.
    """
    data = _unpickle(filename)
    raw_images = data[b'data']
    cls = np.array(data[b'labels'])
    images = _convert_images(raw_images)
    return images, cls

########################################################################
# Public functions that you can call to download the dataset from the internet and load the data into memory.


def maybe_download_and_extract():
    """
    Downloads and extracts the CIFAR-10 dataset if it does not already exist
    in DATA_PATH (set this variable first to the desired path).
    """
    download.maybe_download_and_extract(url=DATA_URL, download_dir=DATA_PATH)


def load_class_names():
    """
    Loads the class names in the CIFAR-10 dataset.

    Returns a list with the names. For example: names[3] is the name
    associated with class number 3.
    """
    raw = _unpickle(filename="batches.meta")[b'label_names']
    names = [x.decode('utf-8') for x in raw]
    return names


def load_training_data():
    """
    Loads all training data from the CIFAR-10 dataset.

    The dataset is divided into 5 data files which are merged here.

    Returns the images, class numbers, and one-hot encoded class labels.
    """
    images = np.zeros(shape=[_NUM_IMAGES_TRAIN, IMG_SIZE,
                      IMG_SIZE, NUM_CHANNELS], dtype=float)
    cls = np.zeros(shape=[_NUM_IMAGES_TRAIN], dtype=int)

    begin = 0

    for i in range(_NUM_FILES_TRAIN):
        images_batch, cls_batch = _load_data(
            filename="data_batch_" + str(i + 1))
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        cls[begin:end] = cls_batch
        begin = end

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=NUM_CLASSES)


def load_test_data():
    """
    Loads all test data from the CIFAR-10 dataset.

    Returns the images, class numbers, and one-hot encoded class labels.
    """
    images, cls = _load_data(filename="test_batch")
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=NUM_CLASSES)

########################################################################
