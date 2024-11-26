MNIST Handwritten Digit Classification using TensorFlow
Project Overview

This project demonstrates how to use deep learning techniques to classify handwritten digits from the MNIST dataset using TensorFlow. The dataset contains 60,000 images of handwritten digits (0-9) for training and 10,000 images for testing. The goal is to train a neural network to predict the correct digit for a given image.
Features:

    Preprocessing: Normalization of image pixel values, splitting of data into training, validation, and test sets.
    Model Design:
        A simple neural network with two hidden layers and dropout for regularization.
        A more complex model with multiple hidden layers and increased layer size.
    Optimizer & Loss Function:
        Used Adam optimizer and sparse categorical cross-entropy loss to train the model.
    Results: Achieved an accuracy of 98.01% on the test dataset using the simple model.

Requirements

To run this project, you need the following Python packages:

    tensorflow
    numpy
    matplotlib
    tensorflow_datasets

Install the necessary libraries using pip:

pip install tensorflow numpy matplotlib tensorflow_datasets

Steps to Run:

    Download the MNIST Dataset: The dataset is automatically loaded using tensorflow_datasets.
    Model Creation:
        The model is a simple fully connected neural network with two hidden layers.
        A more complex model is also implemented to compare results.
    Training: The model is trained for 10 epochs using the Adam optimizer and cross-entropy loss function.
    Evaluation: The model is evaluated on the test dataset to assess its performance.

Training and Testing Results:

    Simple Model Test Accuracy: 98.01%
    Complex Model Test Accuracy: 97.85%

File Structure:

/project-folder
  /data                  # MNIST dataset
  /model                 # Neural network architecture
  /notebooks             # Jupyter notebooks (if any)
  README.md              # Project description and instructions
  train_model.py         # Script to train the model
  evaluate_model.py      # Script to evaluate the model

Contributions

Feel free to fork this project, improve the code, or suggest enhancements. Contributions are welcome!
License

This project is licensed under the MIT License - see the LICENSE file for details.
