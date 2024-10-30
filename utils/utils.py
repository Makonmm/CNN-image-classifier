import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import math
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Util:

    def plot_image(self, image: np.ndarray, img_shape: tuple = (32, 32, 3)):
        """
        Plots a single image.

        :param image: The image to plot.
        :param img_shape: The shape of the image.
        """
        plt.imshow(image.reshape(img_shape),
                   interpolation='nearest')
        plt.axis('off')
        plt.show()

    def plot_images(self, images: np.ndarray, cls_true: np.ndarray, cls_pred: np.ndarray = None, img_shape: tuple = (32, 32, 3), smooth: bool = None):
        """
        Plots a grid of images with true and predicted labels.

        :param images: The images to plot.
        :param cls_true: True class labels.
        :param cls_pred: Predicted class labels (optional).
        :param img_shape: Shape of each image.
        :param smooth: Whether to apply smoothing when plotting images.
        """

        fig, axes = plt.subplots(3, 3)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            interpolation_method = 'spline16' if smooth else 'nearest'
            ax.imshow(images[i].reshape(img_shape),
                      interpolation=interpolation_method)

            xlabel = f"True: {cls_true[i]}" if cls_pred is None else f"True: {
                cls_true[i]}, Pred: {cls_pred[i]}"
            ax.set_xlabel(xlabel)
            ax.axis('off')

        plt.show()

    def plot_history(self, history: dict, metric: str = 'accuracy', loc: str = 'lower right'):
        """
        Plots the training history for a given metric.

        :param history: Training history object.
        :param metric: Metric to plot (e.g., 'accuracy').
        :param loc: Location of the legend.
        """
        plt.plot(history.history[metric], label='train')
        plt.plot(history.history.get('val_' + metric, []), label='test')
        plt.title(f'Model {metric}')
        plt.ylabel(metric)
        plt.xlabel('Epoch')
        plt.legend(loc=loc)
        plt.show()

    def print_test_accuracy(self, model, data, num_classes,
                            show_example_errors: bool = False,
                            show_confusion_matrix: bool = False):
        """
        Calculates and prints the accuracy of the model on the test set.

        :param model: TensorFlow model.
        :param data: Dataset object containing test data.
        :param x: Input data.
        :param y_true: True labels.
        :param num_classes: Number of classes in the dataset.
        :param show_example_errors: Whether to show example errors.
        :param show_confusion_matrix: Whether to show the confusion matrix.
        """
        cls_pred = np.argmax(model.predict(data.test.images), axis=1)
        cls_true = data.test.cls
        correct = (cls_true == cls_pred)
        correct_sum = correct.sum()
        acc = float(correct_sum) / len(data.test.images)

        logging.info(
            f"Accuracy on Test-Set: {acc:.1%} ({correct_sum} / {len(data.test.images)})")

        if show_example_errors:
            self.plot_example_errors(
                data=data, cls_pred=cls_pred, correct=correct)

        if show_confusion_matrix:
            self.plot_confusion_matrix(
                data=data, num_classes=num_classes, cls_pred=cls_pred)

    def plot_confusion_matrix(self, data, num_classes, cls_pred):
        """
        Plots the confusion matrix.

        :param data: Dataset object containing test data.
        :param num_classes: Number of classes in the dataset.
        :param cls_pred: Predicted class labels.
        """
        cls_true = data.test.cls
        cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
        print(cm)
        plt.matshow(cm, cmap='Blues')
        plt.colorbar()
        plt.xticks(range(num_classes))
        plt.yticks(range(num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def plot_example_errors(self, data, cls_pred, correct):
        """
        Plots example errors where predictions were incorrect.

        :param data: Dataset object containing test data.
        :param cls_pred: Predicted class labels.
        :param correct: Array indicating whether predictions were correct.
        """
        incorrect = (correct == False)
        images = data.test.images[incorrect]
        cls_pred = cls_pred[incorrect]
        cls_true = data.test.cls[incorrect]
        self.plot_images(
            images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

    def plot_weights(self, model, layer_index, img_shape: tuple = (32, 32)):
        """
        Plots the weights of a layer.

        :param model: TensorFlow model.
        :param layer_index: Index of the layer to visualize.
        :param img_shape: Shape of each weight image.
        """
        weights, _ = model.layers[layer_index].get_weights()
        w_min, w_max = np.min(weights), np.max(weights)
        fig, axes = plt.subplots(3, 4)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            if i < 10:
                image = weights[:, :, i].reshape(img_shape)
                ax.set_xlabel(f"Weights: {i}")
                ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
            ax.axis('off')

        plt.show()

    def plot_conv_weights(self, weights, input_channel: int = 0):
        """
        Plots the convolutional weights for a specific input channel.

        :param weights: Weights to visualize.
        :param input_channel: Index of the input channel to visualize.
        """
        w_min, w_max = np.min(weights), np.max(weights)
        num_filters = weights.shape[3]
        num_grids = math.ceil(math.sqrt(num_filters))

        fig, axes = plt.subplots(num_grids, num_grids, figsize=(10, 10))

        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        for i, ax in enumerate(axes.flat):
            if i < num_filters:
                img = weights[:, :, input_channel, i]
                ax.imshow(img, vmin=w_min, vmax=w_max,
                          interpolation='nearest', cmap='seismic')
                ax.set_title(f'Kernel {i + 1}')
            else:
                ax.axis('off')

        plt.suptitle(f'Convolutional Weights {
            input_channel}', fontsize=16)
        plt.show()

    def plot_layer_output(self, model, layer_index, image):
        """
        Plots the output of a specified layer after passing an image through the model.

        :param model: TensorFlow model.
        :param layer_index: Index of the layer to visualize.
        :param image: Input image to visualize.
        """
        # Create a new model that will output the specified layer's output
        intermediate_model = tf.keras.Model(inputs=model.inputs[0],
                                            outputs=model.layers[layer_index].output)

        # Predict the output of the intermediate model for the given image
        values = intermediate_model.predict(np.expand_dims(image, axis=0))

        # Get min and max values for normalization
        values_min = np.min(values)
        values_max = np.max(values)

        # Get the number of images (filters) in the output
        num_images = values.shape[3]

        # Calculate the number of grids required for displaying images
        num_grids = math.ceil(math.sqrt(num_images))

        # Create figure with a grid of sub-plots
        fig, axes = plt.subplots(num_grids, num_grids)

        # Iterate through the axes to plot each output image
        for i, ax in enumerate(axes.flat):
            if i < num_images:
                # Get the output image from the values array
                img = values[0, :, :, i]
                # Display the image using binary colormap and normalized values
                ax.imshow(img, vmin=values_min, vmax=values_max,
                          interpolation='nearest', cmap='binary')
                ax.set_title(f"Channel {i + 1}")
            # Hide x and y ticks for better visualization
            ax.set_xticks([])
            ax.set_yticks([])

        # Show the plot
        plt.show()

    def plot_layer_output_torch(self, model, layer_index, image):
        """
        Plots the output of a specified layer after passing an image through the model.

        :param model: PyTorch model.
        :param layer_index: Index of the layer to visualize.
        :param image: Input image to visualize.
        """
        # Create a new model that will output the specified layer's output
        layers = list(model.children())  # Get all layers in the model
        intermediate_model = nn.Sequential(
            *layers[:layer_index + 1])  # Create a sub-model

        # Prepare the image for the model
        input_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(
            0).permute(0, 3, 1, 2).to(device)

        # Get the output of the intermediate model
        with torch.no_grad():
            values = intermediate_model(input_tensor)

        # Convert to numpy and get min and max values for normalization
        # Remove batch dimension and move to CPU
        values = values.squeeze().cpu().numpy()
        values_min = np.min(values)
        values_max = np.max(values)

        # Get the number of images (filters) in the output
        num_images = values.shape[0]  # Number of output channels

        # Calculate the number of grids required for displaying images
        num_grids = math.ceil(math.sqrt(num_images))

        # Create figure with a grid of sub-plots
        fig, axes = plt.subplots(num_grids, num_grids, figsize=(10, 10))

        # Iterate through the axes to plot each output image
        for i, ax in enumerate(axes.flat):
            if i < num_images:
                # Get the output image from the values array
                img = values[i]  # Access the ith output channel
                # Display the image using binary colormap and normalized values
                ax.imshow(img, vmin=values_min, vmax=values_max,
                          interpolation='nearest', cmap='binary')
                ax.set_title(f"Output [{i + 1}]")
            # Hide x and y ticks for better visualization
            ax.set_xticks([])
            ax.set_yticks([])

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    def plot_conv_layer(self, model, layer_index, image):
        """
        Plots the output of a convolutional layer for a given image.

        :param model: TensorFlow model.
        :param layer_index: Index of the convolutional layer to visualize.
        :param image: Input image to visualize.
        """
        intermediate_model = tf.keras.Model(inputs=model.input,
                                            outputs=model.layers[layer_index].output)
        values = intermediate_model.predict(np.expand_dims(image, axis=0))
        num_filters = values.shape[3]
        num_grids = math.ceil(math.sqrt(num_filters))
        fig, axes = plt.subplots(num_grids, num_grids)

        for i, ax in enumerate(axes.flat):
            if i < num_filters:
                img = values[0, :, :, i]
                ax.imshow(img, interpolation='nearest', cmap='binary')
            ax.axis('off')

        plt.show()

    def plot_transfer_values(self, i, images, transfer_values):
        """
        Plots the input image and its corresponding transfer values.

        :param i: Index of the image to visualize.
        :param images: Array of input images.
        :param transfer_values: Array of transfer values for the images.
        """
        print("Input image:")
        plt.imshow(images[i], interpolation='nearest')
        plt.axis('off')
        plt.show()

        print("Transfer-values for the image using Inception model:")
        img = transfer_values[i].reshape((32, 64))
        plt.imshow(img, interpolation='nearest', cmap='Reds')
        plt.axis('off')
        plt.show()

    def plot_scatter(self, values, cls, num_classes):
        """
        Plots a scatter plot of values with colors corresponding to class labels.

        :param values: 2D array of values to plot.
        :param cls: Array of class labels for each value.
        :param num_classes: Number of classes in the dataset.
        """

        cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))
        colors = cmap[cls]
        x = values[:, 0]
        y = values[:, 1]
        plt.scatter(x, y, color=colors)
        plt.show()

    def plot_distorted_image(self, model, distorted_images, image, cls_true):
        """
        Plots distorted images generated from the input image.

        :param model: TensorFlow model.
        :param distorted_images: TensorFlow operation that generates distorted images.
        :param image: Input image to be distorted.
        :param cls_true: True class label for the input image.
        """
        image_duplicates = np.repeat(image[np.newaxis, :, :, :], 9, axis=0)
        result = model.predict(image_duplicates)
        self.plot_images(images=result, cls_true=np.repeat(cls_true, 9))

    def get_test_image(self, images_test, cls_test, i):
        """
        Retrieves a test image and its corresponding label.

        :param images_test: Array of test images.
        :param cls_test: Array of true class labels for the test images.
        :param i: Index of the image to retrieve.
        :return: Tuple of the image and its true label.
        """
        return images_test[i, :, :, :], cls_test[i]
