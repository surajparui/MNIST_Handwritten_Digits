# MNIST Neural Network Classifier

This project is part of the Udacity course and focuses on building a neural network model to evaluate the MNIST dataset. The goal is to achieve high accuracy in digit recognition using the provided dataset.

## Introduction

The MNIST dataset is a well-known benchmark dataset in the field of machine learning and computer vision. It consists of a large number of handwritten digit images, along with their corresponding labels. The task is to train a model that can correctly classify these digits.

## Benchmark Results

Benchmark results on the MNIST dataset have shown impressive accuracy levels. Some notable results include:

- 88% accuracy achieved by Lecun et al., 1998
- 95.3% accuracy achieved by Lecun et al., 1998
- 99.65% accuracy achieved by Ciresan et al., 2011

These benchmark results highlight the advancements in machine learning techniques and the ability to achieve high accuracy on the MNIST dataset.

## Model Architecture and Training

In this project, you will have the opportunity to design and train your own neural network model for digit recognition. The provided code includes essential imports for building the model using PyTorch.

## Usage

To use this code, follow these steps:

1. Import the necessary libraries, including PyTorch and torchvision.
2. Load and preprocess the MNIST dataset using the torchvision.transforms module.
3. Define your neural network model by creating a custom class that inherits from `torch.nn.Module`.
4. Define the loss function and optimizer for training the model.
5. Train the model using the training dataset and adjust the hyperparameters as needed.
6. Evaluate the trained model on the test dataset and measure its accuracy.
7. Visualize the results and analyze the performance of your model.

Please refer to the provided IPython notebook file `MNIST_Handwritten_Digits.ipynb` for the complete code implementation, including step-by-step instructions and code cells.

## Resources

Additional resources and references for working with the MNIST dataset and neural networks:

- Yann LeCun's page: [MNIST benchmarks](http://yann.lecun.com/exdb/mnist/)
- PyTorch documentation: [torch.nn](https://pytorch.org/docs/stable/nn.html)
- torchvision documentation: [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)

Feel free to explore and modify the code to experiment with different network architectures and training techniques. Good luck with your digit recognition project!
