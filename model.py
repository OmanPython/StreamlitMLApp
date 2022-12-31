import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext
from tensorflow.keras.models  import model_from_json

## Method to load Keras model weight and structure files
# def car_plate_reader(image):
def load_model(path):
        try:
            path = splitext(path)[0]
            with open('%s.json' % path, 'r') as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json, custom_objects={})
            model.load_weights('%s.h5' % path)
            st.write("Model Loaded successfully...")
            return model
        except Exception as e:
            print(e)


def car_plate_model(test_image_path,wpod_net):
    def preprocess_image(image_path,resize=False):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        if resize:
            img = cv2.resize(img, (224,224))
        return img

    def get_plate(image_path, Dmax=608, Dmin = 608):
        vehicle = preprocess_image(image_path)
        ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
        side = int(ratio * Dmin)
        bound_dim = min(side, Dmax)
        print(bound_dim,ratio)
        _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
        return vehicle, LpImg, cor

    # test_image_path = "dataset/plate5.jpeg"
    vehicle, LpImg, cor = get_plate(test_image_path)


    return vehicle,LpImg,cor

