import streamlit as st
from PIL import Image
import tensorflow as tf
import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model_train import *



def load_model():
  image_size = [256, 256] # For resizing images. 256px X 256px is same as the size of the images in the datset.
  val_test_split = 0.3 # Validation and test sets together take 30% of the data
  batch_size=64 # No. of samples in each batch
  seed=2000 # Seed value is a arbitary non-negative integer.
  class_names=['1_rupee', '2_rupee', '5_rupee', 'invalid', 'reverse']
  model = keras.Sequential([
      layers.Conv2D(filters=32,
                  kernel_size=5,
                  activation="relu",
                  padding='same',
                  input_shape=[256, 256, 1]),
      layers.MaxPool2D(),
      layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same'),
      layers.MaxPool2D(),
      layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),
      layers.MaxPool2D(),
      layers.Flatten(),
      layers.Dense(units=128, activation="relu"),
      layers.Dense(units=5, activation="softmax"),
  ])
  recall = keras.metrics.Recall(name='recall')
  f1_score = keras.metrics.F1Score(name='f1_score')  # remplace tensorflow_addons.metrics.F1Score (2026)

  METRICS = keras.metrics.Precision(name='precision')
  model.compile(
      optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
      loss='categorical_crossentropy',
      metrics=['accuracy',recall,f1_score,METRICS]
  )
  model.load_weights(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.h5'))  # à côté du script (2026)
  return model

class Predict:    
    def __init__(self):
        self.learn_inference =  load_model()            
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.get_prediction()  
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Image téléchargée", use_container_width=True)
            if st.button('Classify'):
                predictions=model_predict_image(uploaded_file)
                st.write(f'**Prediction**: {predictions}')
            else:
                st.write(f'Click the button to classify')
                #st.write(f'**Probability**: 2%')
        return None  
     
      
if __name__=='__main__':
    file_name='model.h5'
    predictor = Predict.get_image_from_upload()
