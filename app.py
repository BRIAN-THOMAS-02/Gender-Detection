import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import h5py
import numpy as np
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import os
import time
import random
import subprocess
import PIL
from PIL import Image
import pandas as pd
import plotly.graph_objs as go

tf.config.run_functions_eagerly = True

# Define the class labels for the 4 objects
class_labels = ['Male', 'Female']

# Load the trained CNN model
#model = tf.keras.models.load_model('models/gender_detection_model4.h5')
model = load_model('models/gender_detection_model5.hdf5')


def run_program():
    subprocess.Popen(["python", "detect_gender_webcam.py"])

    
# Define the Streamlit app
def app():
    # Add a title and description
    st.write("<p style='font-size: 75px;'> Gender Detection App </p>", unsafe_allow_html=True)
    st.write('This app classifies the input human face into existing 2 genders using a custom pre-trained CNN model.')
    
    st.write("\n\n")
 
    # Define your block of text
    text_block = '''
    This project has been created with the intention of classifying the 2 genders of humans <b> Male, Female </b>. <br>
    We have used a very basic architecture for Convolutional Neural Networks here for training our model which will predict and classify these objects in it's respective classes. Please refer the CNN architecture given below where you can understand the number of layers and neurons and trainable parameters. <br>
    You can also refer the graph below where the train_acc, train_loss, val_acc and val_loss through the epochs can be understood.
    '''

    # Use st.markdown to style your block
    st.markdown(
            f"""
            <div style='border: 2.5px solid orange; border-radius: 8px; padding: 10px; background-color: #0000'>
            <p style='font-size:30px; font-weight:bold; color: white; margin-bottom: 2.5px'> Explanation : </p>
            <p style='font-size:17px; line-height: 2.5; margin-bottom: 0px'> {text_block} </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write('')
    st.write('')
    st.write('')
    
    with st.expander("Click to see Graph"):
        st.subheader('Graph of Object Classification')
        sample_img = cv2.imread('plot/plot5.png')
        FRAME_WINDOW = st.image(sample_img, channels='BGR')
        st.write('This is the Graph of Performance Metrics throughout the training of our custom Gender Classification Model')


    option = st.selectbox('Choose any input ', ('Upload Image', 'Camera / Webcam'))

    if option=='Upload Image':
        with st.sidebar:
            st.write("<p style='font-size: 35px;'> Upload Image </p>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader('Upload image file in jpeg, png format', type=['jpg', 'jpeg', 'png'])

            if uploaded_file:
                with st.spinner("Uploading Image..."):
                    #time.sleep(3)
                    st.success("Done!")


        flag = 0
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img1 = Image.fromarray(img)
            st.write('Color Mode on Image : ', img1.mode)
            #img1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            FRAME_WINDOW1 = st.image(img, channels='BGR', width=350)

            # Detect faces in image using cvlib
            faces, confidences = cv.detect_face(img)
            st.write('Faces Detected and Coordinates of Bounding Boxes -', faces)
            st.write('Confidence of each Face detected - ', confidences)

            # loop through detected faces
            for idx, f in enumerate(faces):
                # get corner points of face rectangle
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # This below code is just for saving the image without the green border/rectangle used for bounding box so that it wont affect the model while performing retraining
                # draw rectangle over face
                img2 = cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 0), 0)
                # crop the detected face region
                face_crop2 = np.copy(img2[startY:endY, startX:endX])

                # This below code is for drawing the bounding boxes which has the green rectangle/border
                # draw rectangle over face
                img = cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 5)
                # crop the detected face region
                face_crop1 = np.copy(img[startY:endY, startX:endX])


                # Save the image with the specified file name
                random_num = random.randint(1528, 999999)
                cv2.imwrite('more_data/face_{}.jpg'.format(random_num), face_crop2)

                if (face_crop1.shape[0]) < 10 or (face_crop1.shape[1]) < 10:
                    continue

                # preprocessing for gender detection model
                face_crop = cv2.resize(face_crop1, (128, 128))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)

                # apply gender detection on face
                conf = model.predict(face_crop)[0]  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

                # get label with max accuracy
                idx = np.argmax(conf)
                label = class_labels[idx]
                label = "{}: {:.2f}%".format(label, conf[idx] * 100)


                FRAME_WINDOW = st.image(face_crop2, channels='BGR', width=100)
                st.write('Confidence : ', conf)
                st.write('Index : ', idx)
                st.write('Prediction - ', label)
                flag = 1
                st.markdown("""<hr style="height:2px;border:none;background-color:darkgray;" /> """, unsafe_allow_html=True)

                Y = startY - 20 if startY - 20 > 20 else startY + 20

                # write label and confidence above face rectangle
                output = cv2.putText(img, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 0), 8)

            if flag == 1:
                st.write('Final Output')
                st.write("")
                st.image(output, channels="BGR")
            else:
                st.markdown("""<hr style="height:2px;border:none;background-color:darkgray;" /> """, unsafe_allow_html=True)
                st.write('Apologies as no face was detected, kindly try another image.')


    if option == 'Camera / Webcam':
        st.write('Click Run to Start/Stop Camera')
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)

        classes = ['Male', 'Female']
        while run:
            # read frame from webcam
            status, frame = camera.read()

            # apply face detection
            face, confidence = cv.detect_face(frame)

            # loop through detected faces
            for idx, f in enumerate(face):

                # get corner points of face rectangle
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # draw rectangle over face
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # crop the detected face region
                face_crop = np.copy(frame[startY:endY, startX:endX])

                if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                    continue

                # preprocessing for gender detection model
                face_crop = cv2.resize(face_crop, (128, 128))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)

                # apply gender detection on face
                conf = model.predict(face_crop)[0]  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

                # get label with max accuracy
                idx = np.argmax(conf)
                label = classes[idx]

                label = "{}: {:.2f}%".format(label, conf[idx] * 100)

                #st.write(startY)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                #st.write(Y)

                # write label and confidence above face rectangle
                cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)


# Run the app
app()