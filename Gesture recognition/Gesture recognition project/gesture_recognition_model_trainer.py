"""--------------------------------------------------------------------------
Welcome, this is amiriiw, this is a simple project about Gesture recognition.
This file is the file where we train the model.
--------------------------------------------"""
import json  # https://pypi-json.readthedocs.io/en/latest/
import pathlib  # https://docs.python.org/3/library/pathlib.html
import numpy as np  # https://numpy.org/devdocs/user/absolute_beginners.html
import tensorflow as tf  # https://www.tensorflow.org/
from tensorflow import keras  # https://www.tensorflow.org/guide/keras
import matplotlib.pyplot as plt  # https://matplotlib.org/stable/index.html
from tensorflow.keras import layers, Sequential  # https://www.tensorflow.org/guide/keras
"""-----------------------------------------------------------------------------------"""


class ImageDataLoader:
    def __init__(self, data_dir, img_height, img_width, batch_size):
        self.data_dir = pathlib.Path(data_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.class_names = None

    def load_data(self):
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)
        validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)
        self.class_names = train_dataset.class_names
        autotune = tf.data.AUTOTUNE
        train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=autotune)
        validation_dataset = validation_dataset.cache().prefetch(buffer_size=autotune)
        return train_dataset, validation_dataset


class ImageClassifierModel:
    def __init__(self, img_height, img_width, num_classes):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.model = None
        self.class_names = None

    def build_model(self, data_augmentation=False):
        layers_list = [
            layers.Rescaling(1. / 255, input_shape=(self.img_height, self.img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes)
        ]
        if data_augmentation:
            augmentation_layers = keras.Sequential([
                layers.RandomFlip("horizontal", input_shape=(self.img_height, self.img_width, 3)),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ])
            layers_list.insert(0, augmentation_layers)
            layers_list.insert(6, layers.Dropout(0.2))
        self.model = Sequential(layers_list)
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.summary()

    def train_model(self, train_dataset, validation_dataset, epochs):
        history = self.model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)
        return history

    def save_model(self, model_path, class_names_path):
        self.model.save(model_path)
        with open(class_names_path, 'w') as f:
            json.dump(self.class_names, f)
        print(f"Model and class names saved to {model_path} and {class_names_path}")

    def load_model(self, model_path, class_names_path):
        self.model = keras.models.load_model(model_path)
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        print(f"Loaded model from {model_path} and class names from {class_names_path}")
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def plot_training_history(self, history, epochs):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(epochs)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def predict_image(self, img_path):
        if self.class_names is None:
            raise ValueError("Class names have not been loaded. Please load the model with class names.")
        img = keras.preprocessing.image.load_img(img_path, target_size=(self.img_height, self.img_width))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_class = np.argmax(score)
        confidence = 100 * np.max(score)
        return self.class_names[predicted_class], confidence


def main():
    data_dir = "dataset"
    img_height, img_width = 180, 180
    batch_size = 10
    num_classes = 5
    epochs = 200
    data_loader = ImageDataLoader(data_dir, img_height, img_width, batch_size)
    train_dataset, validation_dataset = data_loader.load_data()
    classifier = ImageClassifierModel(img_height, img_width, num_classes)
    classifier.class_names = data_loader.class_names
    classifier.build_model(data_augmentation=True)
    history = classifier.train_model(train_dataset, validation_dataset, epochs)
    classifier.save_model('gesture_classifier.keras', 'class_names.json')
    classifier.plot_training_history(history, epochs)


if __name__ == "__main__":
    main()
"""----"""
