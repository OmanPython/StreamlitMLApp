import streamlit as st
from PIL import Image 
from model import car_plate_model,load_model
import numpy as np
import time


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
