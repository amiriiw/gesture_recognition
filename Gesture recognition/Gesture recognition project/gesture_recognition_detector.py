"""--------------------------------------------------------------------------
Welcome, this is amiriiw, this is a simple project about Gesture recognition.
This file is the file where we detect the gesture.
-----------------------------------------------"""
import json  # https://pypi-json.readthedocs.io/en/latest/
import numpy as np  # https://numpy.org/devdocs/user/absolute_beginners.html
import tensorflow as tf  # https://www.tensorflow.org/
from tensorflow import keras  # https://www.tensorflow.org/guide/keras
"""----------------------------------------------------------------"""


class ImageClassifier:
    def __init__(self, model_path: str, class_names_path: str):
        self.model = keras.models.load_model(model_path)
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)

    def predict_image(self, img_path: str, img_height: int, img_width: int) -> tuple[str, float]:
        img = keras.utils.load_img(img_path, target_size=(img_height, img_width))
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_class = np.argmax(score)
        confidence = 100 * np.max(score)
        return self.class_names[predicted_class], confidence


def main():
    model_path = 'gesture_classifier.keras'
    class_names_path = 'class_names.json'
    img_height, img_width = 180, 180
    classifier = ImageClassifier(model_path, class_names_path)
    test_image_path = "dataset/two/img 7.jpg"
    predicted_class, confidence = classifier.predict_image(test_image_path, img_height, img_width)
    print(f"This image most likely belongs to '{predicted_class}' with a {confidence:.2f}% confidence.")


if __name__ == "__main__":
    main()
"""----"""
