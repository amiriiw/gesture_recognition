# Gesture Recognition Project

Welcome to the **Gesture Recognition Project**! This project is designed to train a model to recognize hand gestures from images, and then use that model to detect gestures in real-time.

## Overview

This project consists of two main components:

1. **gesture_recognition_model_trainer.py**: This script is responsible for training a model to recognize hand gestures using an image dataset.
2. **gesture_recognition_detector.py**: This script uses the trained model to detect gestures from images.

## Libraries Used

The following libraries are used in this project:

- **[json](https://pypi-json.readthedocs.io/en/latest/)**: Used for saving and loading class names in JSON format.
- **[pathlib](https://docs.python.org/3/library/pathlib.html)**: Used for working with file paths.
- **[numpy](https://numpy.org/devdocs/user/absolute_beginners.html)**: Used for numerical operations and data manipulation.
- **[tensorflow](https://www.tensorflow.org/)**: Used for building, training, and using the model.
- **[matplotlib](https://matplotlib.org/stable/index.html)**: Used for plotting training history.

## Detailed Explanation

### `gesture_recognition_model_trainer.py`

This script is the core of the project, responsible for training the gesture recognition model. The key components of the script are:

- **ImageDataLoader Class**: Handles loading and preprocessing the image data. The main methods include:
  - `load_data()`: Loads training and validation datasets from a directory, applies caching, shuffling, and prefetching to optimize performance.

- **ImageClassifierModel Class**: Handles building, training, and saving the model. The main methods include:
  - `build_model()`: Constructs a Sequential model using layers like Conv2D, MaxPooling2D, Dense, and optional data augmentation.
  - `train_model()`: Trains the model using the provided datasets and returns the training history.
  - `save_model()`: Saves the trained model and class names to files.
  - `plot_training_history()`: Plots the training and validation accuracy and loss over epochs.

### `gesture_recognition_detector.py`

This script uses the trained model to detect gestures from images. The key components of the script are:

- **ImageClassifier Class**: Handles loading the model, making predictions, and returning results. The main methods include:
  - `predict_image()`: Loads an image, processes it, and predicts the gesture class with confidence.

### How It Works

1. **Model Training**:
    - The `gesture_recognition_model_trainer.py` script reads images from the dataset directory.
    - The images are resized and prepared for training.
    - A Convolutional Neural Network (CNN) model is trained on the dataset to recognize gestures, and the model is saved for later use.

2. **Gesture Detection**:
    - The `gesture_recognition_detector.py` script loads the trained model and uses it to predict the gesture class of a given image.

### Dataset

The dataset used for training the model can be accessed via this [Dataset](https://drive.google.com/drive/folders/1R08P48cetCRyVQkAMqE-noCEnF6RTF6t?usp=sharing)

### Installation and Setup

To use this project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/amiriiw/gesture_recognition
    cd gesture_recognition
    ```

2. Install the required libraries:

    ```bash
    pip install numpy tensorflow matplotlib
    ```

3. Prepare your dataset directory as described above.

4. Run the model training script:

    ```bash
    python gesture_recognition_model_trainer.py
    ```

5. Use the trained model for gesture detection:

    ```bash
    python gesture_recognition_detector.py
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


