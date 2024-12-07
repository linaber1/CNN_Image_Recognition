# CNN_Image_Recognition

## Overview
This project involves data augmentation and training a neural network for image classification. The dataset consists of images labeled with different characters (numbers and letters), and the goal is to augment the data to balance the classes and then train a neural network to classify the images.

## Data Augmentation
The data augmentation process involves generating new images from the existing dataset by applying various transformations. This helps in increasing the size of the dataset and balancing the classes. The steps involved in data augmentation are as follows:

- Visualize Data Distribution: Use matplotlib and seaborn to visualize the distribution of the data. This helps in understanding the imbalance in the dataset.

- Define Augmentation Functions: Define functions to perform random transformations on the images. The transformations include:

Rotation: Randomly rotate the image by a small angle.
Brightness Adjustment: Randomly adjust the brightness of the image.
Contrast Adjustment: Randomly adjust the contrast of the image.
Color Enhancement: Randomly enhance the color of the image.
Flipping: Randomly flip the image horizontally.
Gaussian Blur: Randomly apply Gaussian blur to the image.

- Process Images: Apply the augmentation functions to the images and save the augmented images in a new folder (new_Data). For each sample, multiple augmented images are generated.

- Update DataFrame: Create new rows in the DataFrame for the augmented images and concatenate them with the original data. This results in an updated DataFrame saved under updated_Labels.


## Training
The training process involves loading the augmented data, preprocessing the images, and training a neural network using PyTorch. The steps involved in training are as follows:

- Load and Preprocess Images: Load the images and labels from the updated DataFrame. Convert the images to grayscale and resize them to 28x28 pixels. Normalize the pixel values to be in the range [0, 1].

- Split Data: Split the data into training and test sets. Ensure a balanced distribution of classes by manually selecting a fixed number of samples per class for the test set. (here 2000 for each label)

- Convert to Tensors: Convert the images and labels to PyTorch tensors. This is necessary for training the neural network using PyTorch.

- Define Neural Network: Define the architecture of the neural network. This includes specifying the layers, activation functions, and other hyperparameters.

- Train the Model: Train the neural network using the training data using cross validation with KFold=5. The model is trained for a specified number of epochs, and the loss and accuracy are monitored during training.

- Evaluate the Model: Evaluate the model on the test data. The accuracy and other performance metrics are calculated. (We obtained 0.998)

- Visualize Predictions: Visualize the predictions made by the model on a subset of the test images. This helps in understanding the performance of the model and identifying any potential issues.

## Conclusion
This project demonstrates the process of data augmentation and training a neural network for image classification. By augmenting the data, we can improve the performance of the model and achieve better results. 
