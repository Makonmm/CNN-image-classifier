import numpy as np
import os
from cache import cache


def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generates the One-Hot encoding of classes from an array of integers.

    Example: if class_number=2 and num_classes=4 then
    the one-hot encoding is the array: [0. 0. 1. 0.]

    :param class_numbers: Array of integers with class numbers.
    :param num_classes: Number of classes. If None, uses max(cls) + 1.
    :return: 2D array with shape: [len(cls), num_classes]
    """
    if num_classes is None:

        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]


class DataSet:
    def __init__(self, in_dir, exts='.jpg'):
        """
        Creates a dataset consisting of the filenames in the given directory
        and subdirectories that match the provided file extensions.

        :param in_dir: Root directory for the dataset files.
        :param exts: String or tuple of strings with valid file extensions.
        """
        in_dir = os.path.abspath(in_dir)
        self.in_dir = in_dir
        self.exts = tuple(ext.lower() for ext in exts)
        self.class_names = []
        self.filenames = []
        self.filenames_test = []
        self.class_numbers = []
        self.class_numbers_test = []
        self.num_classes = 0

        for name in os.listdir(in_dir):
            current_dir = os.path.join(in_dir, name)
            if os.path.isdir(current_dir):
                self.class_names.append(name)
                filenames = self._get_filenames(current_dir)
                self.filenames.extend(filenames)

                class_number = self.num_classes
                class_numbers = [class_number] * len(filenames)
                self.class_numbers.extend(class_numbers)

                filenames_test = self._get_filenames(
                    os.path.join(current_dir, 'test'))
                self.filenames_test.extend(filenames_test)
                class_numbers_test = [class_number] * len(filenames_test)
                self.class_numbers_test.extend(class_numbers_test)

                self.num_classes += 1

    def _get_filenames(self, dir):
        """
        Creates and returns a list of filenames with matching extensions in the given directory.

        :param dir: Directory to scan for files.
        :return: List of filenames. Only names, does not include the directory.
        """
        filenames = []
        if os.path.exists(dir):
            for filename in os.listdir(dir):
                if filename.lower().endswith(self.exts):
                    filenames.append(filename)
        return filenames

    def get_paths(self, test=False):
        """
        Gets the full paths of the files in the dataset.

        :param test: Boolean. Returns paths for the test set (True) or for the training set (False).
        :return: Iterator with strings for the path names.
        """
        filenames = self.filenames_test if test else self.filenames
        class_numbers = self.class_numbers_test if test else self.class_numbers
        test_dir = "test/" if test else ""

        for filename, cls in zip(filenames, class_numbers):
            path = os.path.join(
                self.in_dir, self.class_names[cls], test_dir, filename)
            yield path

    def get_training_set(self):
        """
        Returns the list of paths for the files in the training set,
        the list of class numbers as integers, and the classes as one-hot encoded arrays.
        """
        return (list(self.get_paths()),
                np.asarray(self.class_numbers),
                one_hot_encoded(class_numbers=self.class_numbers, num_classes=self.num_classes))

    def get_test_set(self):
        """
        Returns the list of paths for the files in the test set,
        the list of class numbers as integers, and the classes as one-hot encoded arrays.
        """
        return (list(self.get_paths(test=True)),
                np.asarray(self.class_numbers_test),
                one_hot_encoded(class_numbers=self.class_numbers_test, num_classes=self.num_classes))


def load_cached(cache_path, in_dir):
    """
    Wrapper function to create a DataSet object, which will
    be loaded from a cache file if it already exists, otherwise,
    a new object will be created and saved in the cache file.

    :param cache_path: Path to the cache file.
    :param in_dir: Root directory for the dataset files.
    :return: The DataSet object.
    """
    print("Creating dataset from files in: " + in_dir)
    dataset = cache(cache_path=cache_path, fn=DataSet, in_dir=in_dir)
    return dataset
