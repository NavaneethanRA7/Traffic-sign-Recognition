# Traffic Sign Recognition Using CNN - README

## Introduction

This project involves the implementation of a Traffic Sign Recognition system using a Convolutional Neural Network (CNN). The system is designed to accurately recognize and classify traffic signs from images, an essential task for autonomous driving systems and advanced driver-assistance systems (ADAS). The implementation is carried out using Python and various libraries, including TensorFlow, Keras, and OpenCV, within a Jupyter Notebook.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Model Architecture](#model-architecture)
6. [Training the Model](#training-the-model)
7. [Evaluating the Model](#evaluating-the-model)
8. [Inference](#inference)
9. [Results](#results)
10. [Acknowledgements](#acknowledgements)

## Project Structure

The project consists of the following files:

- traffic_sign_recognition.ipynb: The Jupyter Notebook containing all the code for training, evaluating, and testing the CNN model.
- README.md: This readme file.
- requirements.txt: A file listing the required libraries and their versions.

## Requirements

To run this project, you need to have the following installed:

- Python 3.7 or higher
- Jupyter Notebook
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

## Installation

1. Clone the repository to your local machine:
   bash
   git clone https://github.com/yourusername/traffic_sign_recognition.git
   cd traffic_sign_recognition
   

2. Create a virtual environment (optional but recommended):
   bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   

3. Install the required libraries:
   bash
   pip install -r requirements.txt
   

4. Open the Jupyter Notebook:
   bash
   jupyter notebook traffic_sign_recognition.ipynb
   

## Dataset

The dataset used for this project is the German Traffic Sign Recognition Benchmark (GTSRB). It contains 43 classes of traffic signs with over 50,000 images. You can download the dataset from [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

After downloading, extract the dataset into a folder named data in the project directory.

## Model Architecture

The CNN model is designed with the following architecture:

- Input layer
- Convolutional layers
- Max-Pooling layers
- Fully Connected (Dense) layers
- Output layer with Softmax activation

The exact architecture details can be found in the Jupyter Notebook.

## Training the Model

To train the model, follow the steps in the Jupyter Notebook. The notebook includes data preprocessing, model architecture definition, compilation, and training. Make sure to adjust the hyperparameters as needed.

## Evaluating the Model

The notebook contains code to evaluate the model's performance on the test dataset. Evaluation metrics such as accuracy, precision, recall, and confusion matrix are computed and displayed.

## Inference

For inference, you can use the trained model to predict traffic signs from new images. The notebook provides examples of how to load new images and perform predictions.

## Results

The results of the trained model, including accuracy and sample predictions, are displayed in the notebook. The model achieves high accuracy on the test set, demonstrating its effectiveness in recognizing traffic signs.

## Acknowledgements

This project is based on the GTSRB dataset. Special thanks to the developers and contributors of TensorFlow, Keras, and OpenCV for providing the tools and libraries used in this project.

---

Feel free to contribute to this project by opening issues or submitting pull requests on the [GitHub repository](https://github.com/yourusername/traffic_sign_recognition).

For any questions or feedback, please contact navaneethan.ra7@gmail.com

Happy coding!
