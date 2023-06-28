# Fire_Detection_Using_CNN

## Problem Statement

The Fire Detection Model is designed to identify and detect instances of fire in images. The objective is to develop a robust and accurate system that can aid in the early detection and prevention of fire-related incidents. By utilizing Convolutional Neural Networks (CNNs) and deep learning techniques, the model aims to achieve high accuracy in classifying images as either containing fire or not.

## Dataset

The dataset used for training and evaluating the Fire Detection Model can be found on Kaggle at the following link: https://www.kaggle.com/datasets/phylake1337/fire-dataset

The dataset consists of a large collection of labeled images, with each image labeled as either containing fire or being fire-free. It is crucial to have a balanced dataset with a sufficient number of positive and negative examples to train an effective fire detection model. The dataset should be divided into training, validation, and testing sets to assess the model's performance accurately.
![1](https://github.com/Paragg99/Fire_Detection_Using_CNN/assets/91948118/e7fec990-91e6-431c-ba83-c4ae4cd9c736)
![highway_gre50](https://github.com/Paragg99/Fire_Detection_Using_CNN/assets/91948118/74cf9f1f-1f02-4d02-a302-466e13c310f0)


## What We've Done

**Data Preparation:** We obtained a labeled dataset from Kaggle containing images with fire and fire-free labels. The dataset was divided into training and testing sets.

**Model Development:** We designed a CNN model architecture tailored for fire detection. The model consists of multiple convolutional max pooling and dropout layers, followed by fully connected layers for classification.
Training: The model was trained using the training dataset. The loss function Binary Crossentropy and Adam optimizer were configured to optimize the model's performance.

**Evaluation:** The trained model was evaluated using the validation dataset to measure its accuracy and assess its generalization capabilities.

**Testing:** Finally, we assessed the model's performance on the testing dataset, providing an 80% accurate measure of its ability to detect fire in unseen images.

## Conclusion

The Fire Detection Model has demonstrated a commendable accuracy of 80% on the test dataset, indicating its effectiveness in identifying fire instances in images. However, there is still room for improvement by utilizing hyperparameter tuning techniques like adjusting the learning rate and implementing early stopping.

Hyperparameter tuning can help fine-tune the model's parameters to achieve even higher accuracy levels. By systematically varying the learning rate and monitoring the model's performance, we can identify the optimal value that leads to better convergence and reduced overfitting. Early stopping enables us to stop training when the model's performance on the validation set starts deteriorating, preventing it from overfitting the training data.

By incorporating these hyperparameter tuning techniques, we can enhance the Fire Detection Model's accuracy and robustness, making it an even more valuable tool for early fire detection and prevention.
