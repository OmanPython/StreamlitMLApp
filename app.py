import streamlit as st
from PIL import Image 
# from model import car_plate_model,load_model
import numpy as np
import time
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
        _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
        return vehicle, LpImg, cor

    # test_image_path = "dataset/plate5.jpeg"
    vehicle, LpImg, cor = get_plate(test_image_path)


    return vehicle,LpImg,cor




st.write("Car Plate Detector")

### Loading the NN MODEL
wpod_net_path = "models/wpod-net.json"
wpod_net = load_model(wpod_net_path)

st.write(
    ":fire: Try uploading an image :grin:"
)
    
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])





def fix_image(image1,image2):
    image = Image.fromarray(np.uint8(image1*255))
    col1.write("Original Image :camera:")
    col1.image(image)

    col2.write("Plate(s) :camera:")
    for img in image2:
        img = Image.fromarray(np.uint8(img*255))
        col2.image(img)

col1, col2 = st.columns(2)


if my_upload is not None:
    with st.spinner('Wait for it...'):
        img = Image.open(my_upload)
        if my_upload.name.split(".")[-1].lower() == "png":
            img = img.convert('RGB')
        img = img.save("img.jpg")
        image1, image2,cor = car_plate_model("img.jpg",wpod_net)
        fix_image(image1,image2)
    st.success(f'{len(image2)} Car Plate Detected! Try Another One')
else:
    st.info("Upload a car image with a plate   :car:")
