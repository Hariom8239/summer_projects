# data_loader.py

import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

def load_data(data_dir, img_size=(32, 32)):
    def load_images_from_folder(folder):
        images = []
        labels = []
        class_names = sorted(os.listdir(folder))
        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(folder, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label)
        return np.array(images), np.array(labels), class_names

    # Load training data
    train_dir = os.path.join(data_dir, 'train')
    x_train, y_train, class_names = load_images_from_folder(train_dir)
    
    # Load testing data
    test_dir = os.path.join(data_dir, 'test')
    x_test, y_test, _ = load_images_from_folder(test_dir)
    
    # Normalize the images
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    # One-hot encode the labels
    num_classes = len(class_names)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    return x_train, y_train, x_test, y_test, num_classes
