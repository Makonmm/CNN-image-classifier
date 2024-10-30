# Imports

from dataset.cifar10 import IMG_SIZE, NUM_CHANNELS, NUM_CLASSES
import dataset.cifar10 as dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random

from utils.utils import Util

# Configure the device to use GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the class Util
u = Util()

# Download and load dataset
dataset.data_path = "data/CIFAR-10/"
dataset.maybe_download_and_extract()

# Load class names and training/test data
class_names = dataset.load_class_names()  # Load class names from the dataset
# Load training images and labels
images_train, cls_train, labels_train = dataset.load_training_data()
# Load test images and labels
images_test, cls_test, labels_test = dataset.load_test_data()


# Print dataset sizes
print("Data size:\n")
print(f"1. Training-set: \t{len(images_train)}")
print(f"2. Test-set: \t{len(images_test)}")

# Define the neural network


class MainNetwork(nn.Module):
    def __init__(self):
        super(MainNetwork, self).__init__()
        # Define the layers of the network
        # First convolutional layer
        self.conv1 = nn.Conv2d(NUM_CHANNELS, 32, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 128, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm2d(128)  # Batch normalization

        # Third convolutional layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(256)  # Batch normalization

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(512)  # Batch normalization

        # First fully connected layer
        self.fc1 = nn.Linear(512 * (IMG_SIZE // 16) * (IMG_SIZE // 16), 1024)
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer for regularization

        self.fc2 = nn.Linear(1024, 512)  # Second fully connected layer
        self.dropout2 = nn.Dropout(0.5)  # Dropout layer for regularization

        self.fc3 = nn.Linear(512, 256)  # Third fully connected layer
        self.dropout3 = nn.Dropout(0.5)  # Dropout layer for regularization

        # Output layer to predict class scores
        self.fc4 = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        # Define the forward pass through the network
        # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # Flatten the tensor for the fully connected layers
        x = x.reshape(x.size(0), -1)
        # Fully connected layer with ReLU and dropout
        x = self.dropout1(F.relu(self.fc1(x)))
        # Fully connected layer with ReLU and dropout
        x = self.dropout2(F.relu(self.fc2(x)))
        # Fully connected layer with ReLU and dropout
        x = self.dropout3(F.relu(self.fc3(x)))
        x = self.fc4(x)  # Output layer (logits)
        return x  # Return the output logits


# Function to plot images

def plot_images(images, cls_true, cls_pred=None, smooth=None):
    u.plot_images(images=images, cls_true=cls_true,
                  cls_pred=cls_pred, smooth=smooth)


# Display random images from the test set
offset = random.randint(0, 1000)  # Generate a random offset
images = images_test[offset:offset + 9]  # Select 9 random images
# Get true class labels for these images
cls_true = cls_test[offset:offset + 9]

# Plot the selected images with and without smoothing
plot_images(images=images, cls_true=cls_true, smooth=False)
plot_images(images=images, cls_true=cls_true, smooth=True)


# Instantiate the model and move it to gpu
model = MainNetwork().to(device)
# Define loss function
criterion = nn.CrossEntropyLoss()

WEIGHT_DECAY = 1e-3  # Set weight decay for regularization
TRAIN_BATCH_SIZE = 64  # Define the batch size for training

# Set up the optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)


# Function to create random batch of training data


def random_batch():
    num_images = len(images_train)  # Get the number of training images
    # Randomly select indices
    idx = np.random.choice(num_images, size=TRAIN_BATCH_SIZE, replace=False)
    return (torch.tensor(images_train[idx], dtype=torch.float32).permute(0, 3, 1, 2).to(device),  # Convert to tensor and permute dimensions
            # Convert labels to tensor
            torch.tensor(labels_train[idx], dtype=torch.float32).to(device))


# Lists to hold losses and accuracy
train_losses = []
train_accuracy = []
test_losses = []
test_accuracy = []

# Optimization function


def optimize(num_iterations):
    start_time = time.time()  # Record the start time
    model.train()  # Set the model to training mode

    for i in range(num_iterations):
        x_batch, y_true_batch = random_batch()  # Get a random batch of data
        optimizer.zero_grad()  # Reset gradients
        predictions = model(x_batch)  # Forward pass through the model
        loss_value = criterion(predictions, torch.max(y_true_batch, 1)[
                               1])  # Here we calculate the loss
        loss_value.backward()  # Backpropagation to compute gradients
        optimizer.step()  # Update model parameters

        train_losses.append(loss_value.item())  # Append loss value to the list
        _, predicted = torch.max(predictions, 1)  # Get predicted classes
        correct_predictions = (
            # Check for correct predictions
            predicted == torch.max(y_true_batch, 1)[1]).float()
        batch_acc = correct_predictions.mean()  # Calculate batch accuracy
        # Append batch accuracy to the list
        train_accuracy.append(batch_acc.item())

        if i % 100 == 0 or i == num_iterations - 1:
            # evaluate_test()  # Call evaluate_test() function every iteration
            _, predicted = torch.max(predictions, 1)  # Get predicted classes
            batch_acc = (predicted == torch.max(
                y_true_batch, 1)[1]).float().mean()  # Recalculate accuracy
            msg = "--> Iteration: {0:>6}, Training Batch Accuracy: {1:>6.1%}, Loss: {2:>6.4f}"
            # Print training progress
            print(msg.format(i, batch_acc.item(), loss_value.item()))

    end_time = time.time()  # Record end time
    time_dif = end_time - start_time  # Calculate execution time
    # Print execution time
    print("Execution time: " + str(timedelta(seconds=int(round(time_dif)))))


# Function to evaluate the test data

def evaluate_test():
    model.eval()  # Set the model to eval mode

    with torch.no_grad():  # Disable gradient calculation for efficiency
        x_batch, y_true_batch = random_batch()  # Set a random batch of test data
        # Forward pass through the model to get predictions
        predictions = model(x_batch)
        loss_value = criterion(predictions, torch.max(
            y_true_batch, 1)[1])  # Calculate the loss

        # Append test losses to the list (test)
        test_losses.append(loss_value.item())
        _, predicted = torch.max(predictions, 1)  # Get predicted classes
        correct_predictions = (
            # Check for correct predictions
            predicted == torch.max(y_true_batch, 1)[1]).float()
        batch_acc = correct_predictions.mean()  # Calculate batch accuracy
        # Append batch accuracy to the list (test)
        test_accuracy.append(batch_acc.item())


# Function to plot the learning curve for training
def plot_learning_curve():
    plt.figure(figsize=(12, 5))  # Create a new figure for the learning curve

    plt.subplot(1, 2, 1)  # Create the first subplot for training loss
    plt.plot(train_losses, label="Train loss",
             color="red")  # Plot training loss
    plt.title("Learning curve: LOSS")  # Set the title for the loss plot
    plt.xlabel("Iterations")  # Label for x-axis
    plt.ylabel("Loss")  # Label for y-axis
    plt.legend()  # Show legend

    plt.subplot(1, 2, 1)  # Create the second subplot for training accuracy
    plt.plot(train_accuracy, label="Train accuracy",
             color="blue")  # Plot training accuracy
    # Set the title for the accuracy plot
    plt.title("Learning curve: ACCURACY")
    plt.xlabel("Iterations")  # Label for x-axis
    plt.ylabel("Accuracy")  # Label for y-axis
    plt.legend()  # Show legend

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display the plots

# Function to plot the test curve for evaluation


def plot_test_curve():
    plt.figure(figsize=(12, 5))  # Create a new figure for the test curve

    plt.subplot(1, 2, 1)  # Create the first subplot for test loss
    plt.plot(test_losses, label="Test loss", color="red")  # Plot test loss
    plt.title("Test curve: LOSS")  # Set the title for the test loss plot
    plt.xlabel("Iterations")  # Label for x-axis
    plt.ylabel("Loss")  # Label for y-axis
    plt.legend()  # Show legend

    plt.subplot(1, 2, 1)  # Create the second subplot for test accuracy
    plt.plot(test_accuracy, label="Test accuracy",
             color="blue")  # Plot test accuracy
    # Set the title for the test accuracy plot
    plt.title("Test curve: ACCURACY")
    plt.xlabel("Iterations")  # Label for x-axis
    plt.ylabel("Accuracy")  # Label for y-axis
    plt.legend()  # Show legend

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display the plots

# Function to plot example errors


def plot_example_errors(cls_pred, correct, cls_true):
    # Get indices of incorrect predictions
    incorrect_indices = np.where(correct == False)[0]
    # Select the first 9 incorrect indices
    selected_indices = incorrect_indices[:9]

    # Get images corresponding to the incorrect predictions
    images_ = images_test[selected_indices]
    # Get predicted classes for these images
    cls_pred = cls_pred[selected_indices]

    cls_true = cls_test[selected_indices]  # Get true classes for these images

    plot_images(images=images_, cls_true=cls_true,
                cls_pred=cls_pred)  # Plot the example errors

# Function to plot a single image


def plot_image(image):
    fig, axes = plt.subplots(1, 2)  # Create a subplot with 2 columns
    ax0 = axes.flat[0]  # First subplot for raw image
    ax1 = axes.flat[1]  # Second subplot for smoothed image

    ax0.imshow(image, interpolation='nearest')  # Display raw image
    ax1.imshow(image, interpolation='spline16')  # Display smoothed image

    ax0.set_xlabel('Raw')  # Label for raw image
    ax1.set_xlabel('Smooth')  # Label
    ax1.set_xlabel('Smooth')  # Label for smoothed image
    plt.show()  # Show the plot

    return image  # Return the image

# Função para plotar matriz de confusão


# Function to plot the confusion matrix
def plot_confusion_matrix(cls_pred):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true=cls_test, y_pred=cls_pred)

    for i in range(NUM_CLASSES):  # Iterate over each class
        # Create a label for the class
        class_name = "({}) {}".format(i, class_names[i])
        # Print the confusion matrix row for the class
        print(cm[i, :], class_name)

    class_numbers = [" ({0})".format(i) for i in range(
        NUM_CLASSES)]  # Create class number labels
    print("".join(class_numbers))  # Print class numbers

    plt.matshow(cm)  # Display the confusion matrix as a heatmap
    plt.colorbar()  # Show color bar for the heatmap
    tick_marks = np.arange(NUM_CLASSES)  # Create tick marks for classes
    plt.xticks(tick_marks, range(NUM_CLASSES))  # Set x-ticks
    plt.yticks(tick_marks, range(NUM_CLASSES))  # Set y-ticks

    # Set title for the confusion matrix plot
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted')  # Label for x-axis
    plt.ylabel('True')  # Label for y-axis
    plt.show()  # Show the plot

# Function to make predictions


def predict_cls(images, cls_true):
    model.eval()  # Set the model to evaluation mode
    num_images = len(images)  # Get the number of images
    # Initialize predictions array
    cls_pred = np.zeros(shape=num_images, dtype=np.int32)

    i = 0
    while i < num_images:  # Loop through images in batches
        j = min(i + 256, num_images)  # Determine the end index for the batch
        with torch.no_grad():  # Disable gradient calculation
            predictions = model(torch.tensor(
                # Forward pass
                images[i:j], dtype=torch.float32).permute(0, 3, 1, 2).to(device))
        cls_pred[i:j] = torch.argmax(
            predictions, axis=1).cpu().numpy()  # Get predicted classes
        i = j  # Move to the next batch

    correct = (cls_true == cls_pred)  # Check if predictions are correct
    return correct, cls_pred  # Return correctness and predicted classes

# Function to print test accuracy


def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False):
    # Get correct predictions and predicted classes
    correct_preds, predicted_classes = predict_cls(images_test, cls_test)
    # Calculate accuracy and number of correct predictions
    acc, num_correct = correct_preds.mean(), correct_preds.sum()
    num_images = len(correct_preds)  # Get the total number of images

    # Format accuracy message
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))  # Print accuracy

    if show_example_errors:  # If flag is set, show example errors
        print("Example errors:")
        plot_example_errors(cls_pred=predicted_classes, correct=correct_preds,
                            cls_true=cls_true)  # Plot example errors

    if show_confusion_matrix:  # If flag is set, show confusion matrix
        print("Confusion Matrix:")
        # Plot confusion matrix
        plot_confusion_matrix(cls_pred=predicted_classes)


# Function to visualize outputs from specific layers
def visualize_layer_outputs(model, u, images_test, cls_test):
    # Get a test image and its class
    img, cls = u.get_test_image(images_test, cls_test, 190)
    original = plot_image(img)  # Show the original image

    plt.imshow(original)  # Display the original image
    plt.title(f'Original Image (Class: [{cls}])')  # Title with the true class
    plt.axis('off')  # Hide axes
    plt.show()  # Show the plot

    conv_layer1 = model.conv1  # Get the first convolutional layer
    conv_layer2 = model.conv2  # Get the second convolutional layer
    conv_layer3 = model.conv3  # Get the third convolutional layer
    conv_layer4 = model.conv4  # Get the fourth convolutional layer

    output_conv1 = conv_layer1(torch.tensor(
        # Forward pass through first conv layer
        img, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device))
    # Forward pass through second conv layer
    output_conv2 = conv_layer2(output_conv1)

    # Visualize the weights of the convolutional layers
    u.plot_conv_weights(weights=conv_layer1.weight.data.cpu().numpy(
    ), input_channel=0)  # Plot weights of the first convolutional layer
    u.plot_conv_weights(weights=conv_layer2.weight.data.cpu().numpy(
    ), input_channel=1)  # Plot weights of the second convolutional layer
    u.plot_conv_weights(
        weights=conv_layer3.weight.data.cpu().numpy(), input_channel=2)
    u.plot_conv_weights(
        weights=conv_layer4.weight.data.cpu().numpy(), input_channel=0)

    # Visualize the output from the first and second layers
    u.plot_layer_output_torch(model, 0, img)  # Output from the first layer
    u.plot_layer_output_torch(model, 3, img)  # Output from the fourth layer

    # Process and visualize more images in the same manner
    for index in [105, 305]:  # Example indices for additional images
        # Get the test image and its class
        img, cls = u.get_test_image(images_test, cls_test, index)
        original = plot_image(img)  # Show the original image

        plt.imshow(original)  # Display the original image
        # Title with the true class
        plt.title(f'Original Image (Class: [{cls}])')
        plt.axis('off')  # Hide axes
        plt.show()  # Show the plot

        output_conv1 = conv_layer1(torch.tensor(
            # Forward pass through first conv layer
            img, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device))
        # Forward pass through second conv layer
        output_conv2 = conv_layer2(output_conv1)
        u.plot_layer_output_torch(model, 0, img)  # Output from the first layer
        # Output from the fourth layer
        u.plot_layer_output_torch(model, 3, img)


# Execute training and testing
# Train the model for a specified number of iterations
optimize(num_iterations=150000)

# Make predictions on the test set
correct, cls_pred = predict_cls(images_test, cls_test)

# Generate a random offset for displaying images
offset = random.randint(0, len(images_test) - 9)
images = images_test[offset:offset + 9]  # Select 9 random test images
cls_true = cls_test[offset:offset + 9]  # Get true classes for these images
# Get predicted classes for these images
cls_pred = cls_pred[offset:offset + 9]

plot_learning_curve()  # Plot the learning curves for loss and accuracy
plot_test_curve()  # Plot the test curve for loss and accuracy
# Print test accuracy and show example errors and confusion matrix
print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)
plot_images(images=images, cls_true=cls_true, cls_pred=cls_pred,
            smooth=False)  # Plot images without smoothing
plot_images(images=images, cls_true=cls_true, cls_pred=cls_pred,
            smooth=True)  # Plot images with smoothing

# Call the function to visualize outputs from specific layers
# Visualize layer outputs for the test images
visualize_layer_outputs(model, u, images_test, cls_test)
