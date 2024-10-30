from dataset.cifar10 import IMG_SIZE, NUM_CHANNELS, NUM_CLASSES
import dataset.cifar10 as dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import timedelta
import tensorflow as tf
import time
import random

from utils.utils import Util

# Create a utility instance
u = Util()

# Set the path for the CIFAR-10 dataset
dataset.data_path = "data/CIFAR-10/"
# Download and extract the dataset if necessary
dataset.maybe_download_and_extract()

# Load class names, training and test data
class_names = dataset.load_class_names()
images_train, cls_train, labels_train = dataset.load_training_data()
images_test, cls_test, labels_test = dataset.load_test_data()

# Print the sizes of the training and test datasets
print("Size of")
print(f"1. Training-set:\t{len(images_train)}")
print(f"2. Test-set:\t{len(images_test)}")


# Define input layers for the model
x = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS), name='x')
y_true = tf.keras.Input(shape=(NUM_CLASSES,), name='y_true')

# Function to build the neural network model


def main_network():
    model = tf.keras.Sequential([
        # First convolutional layer
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                               padding='same', input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        # Second convolutional layer
        tf.keras.layers.Conv2D(
            128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        # Third convolutional layer
        tf.keras.layers.Conv2D(
            256, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        # Fourth convolutional layer
        tf.keras.layers.Conv2D(
            512, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        # Flatten the output and add dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Add Dropout for regularization
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Add Dropout for regularization
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Add Dropout for regularization
        tf.keras.layers.Dense(
            NUM_CLASSES, activation='softmax')  # Output layer
    ])

    # Compile the model with optimizer, loss function, and metrics
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Function to plot images with true and predicted classes
def plot_images(images, cls_true, cls_pred=None, smooth=None):
    u.plot_images(images=images, cls_true=cls_true,
                  cls_pred=cls_pred, smooth=smooth)


# Randomly select an offset for testing images
offset = random.randint(0, 1000)
images = images_test[offset:offset+9]  # Select 9 test images
cls_true = cls_test[offset:offset+9]  # True classes for the selected images

# Plot images with and without smoothing
plot_images(images=images, cls_true=cls_true, smooth=False)
plot_images(images=images, cls_true=cls_true, smooth=True)

# Create the model
model = main_network()

# Set batch size for training
TRAIN_BATCH_SIZE = 128

# Function to generate a random batch of training data


def random_batch():
    num_images = len(images_train)
    idx = np.random.choice(num_images, size=TRAIN_BATCH_SIZE, replace=False)
    return images_train[idx, :, :, :], labels_train[idx, :]

# Function for optimizing the model


def optimize(num_iterations):
    start_time = time.time()  # Start timer

    for i in range(num_iterations):
        x_batch, y_true_batch = random_batch()  # Get a random batch
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)  # Forward pass
            loss_value = tf.keras.losses.categorical_crossentropy(
                y_true_batch, predictions)  # Calculate loss
        # Compute gradients
        grads = tape.gradient(loss_value, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(grads, model.trainable_variables))  # Update model weights

        # Print training metrics every 100 iterations
        if i % 100 == 0 or i == num_iterations - 1:
            batch_acc = tf.reduce_mean(tf.cast(tf.equal(
                tf.argmax(predictions, axis=1), tf.argmax(y_true_batch, axis=1)), tf.float32))
            msg = "--> Iteration: {0:>6}, Training Batch Accuracy: {1:>6.1%}, Loss: {2:>6.4f}"
            print(msg.format(i, batch_acc.numpy(),
                             tf.reduce_mean(loss_value).numpy()))

    end_time = time.time()  # End timer
    time_dif = end_time - start_time
    print("Execution time: " + str(timedelta(seconds=int(round(time_dif)))))

# Function to plot an image with raw and smooth versions


def plot_image(image):
    fig, axes = plt.subplots(1, 2)
    ax0 = axes.flat[0]
    ax1 = axes.flat[1]

    ax0.imshow(image, interpolation='nearest')
    ax1.imshow(image, interpolation='spline16')

    ax0.set_xlabel('Raw')  # Label for raw image
    ax1.set_xlabel('Smooth')  # Label for smooth image
    plt.show()

    return image

# Function to plot example errors


def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)  # Find incorrect predictions
    images = images_test[incorrect]  # Select incorrect images
    cls_pred = cls_pred[incorrect]  # Select predicted classes
    cls_true = cls_test[incorrect]  # Select true classes
    plot_images(images=images[0:9],
                # Plot the images
                cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

# Function to plot confusion matrix


def plot_confusion_matrix(cls_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_true=cls_test, y_pred=cls_pred)

    # Print confusion matrix for each class
    for i in range(NUM_CLASSES):
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    class_numbers = [" ({0})".format(i) for i in range(NUM_CLASSES)]
    print("".join(class_numbers))

    plt.matshow(cm)  # Display confusion matrix
    plt.colorbar()  # Add color bar
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, range(NUM_CLASSES))
    plt.yticks(tick_marks, range(NUM_CLASSES))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Function to predict class labels for images


def predict_cls(images, labels, cls_true):
    num_images = len(images)
    # Initialize predictions
    cls_pred = np.zeros(shape=num_images, dtype=np.int32)

    i = 0
    while i < num_images:
        j = min(i + 256, num_images)  # Process images in batches
        predictions = model.predict(images[i:j, :])  # Get predictions
        # Store predicted classes
        cls_pred[i:j] = np.argmax(predictions, axis=1)
        i = j  # Move to next batch

    correct = (cls_true == cls_pred)  # Check if predictions are correct
    return correct, cls_pred

# Function to print the accuracy on the test set


def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False):
    correct, cls_pred = predict_cls(
        images_test, labels_test, cls_test)  # Get predictions
    acc, num_correct = correct.mean(), correct.sum()  # Calculate accuracy
    num_images = len(correct)

    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Show example errors and confusion matrix if specified
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

# Function to visualize the outputs of specific layers


def visualize_layer_outputs(model, u, images_test, cls_test):
    img, cls = u.get_test_image(images_test, cls_test, 190)  # Get a test image
    original = plot_image(img)  # Show the original image

    plt.imshow(original)
    # Title for the original image
    plt.title(f'Original Image (Class: [{cls}])')
    plt.axis('off')
    plt.show()

    conv_layer1 = model.layers[0]  # First convolutional layer
    conv_layer2 = model.layers[3]  # Second convolutional layer

    # Get outputs from the convolutional layers
    output_conv1 = conv_layer1(np.expand_dims(img, axis=0))
    output_conv2 = conv_layer2(output_conv1)

    # Plot weights and outputs of the layers
    u.plot_conv_weights(
        weights=conv_layer1.weights[0].numpy(), input_channel=0)
    u.plot_conv_weights(
        weights=conv_layer2.weights[0].numpy(), input_channel=1)

    # Visualize layer outputs for the first and second layers
    u.plot_layer_output(model, 0, img)  # First layer output
    u.plot_layer_output(model, 3, img)  # Third layer output

    # Process and visualize more images
    for index in [105, 305]:
        img, cls = u.get_test_image(
            images_test, cls_test, index)  # Get another test image
        original = plot_image(img)  # Show the original image

        plt.imshow(original)
        # Title for the original image
        plt.title(f'Original Image (Class: [{cls}])')
        plt.axis('off')
        plt.show()

        output_conv1 = conv_layer1(np.expand_dims(
            img, axis=0))  # Get output of first layer
        output_conv2 = conv_layer2(output_conv1)  # Get output of second layer
        u.plot_layer_output(model, 0, img)  # Visualize first layer output
        u.plot_layer_output(model, 2, img)  # Visualize second layer output


# Optimize the model with a specified number of iterations
optimize(num_iterations=100000)
# Print the test accuracy and show example errors and confusion matrix
print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)

# Call the function to visualize outputs of layers
visualize_layer_outputs(model, u, images_test, cls_test)
