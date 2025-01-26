import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator




Dermatologist = load_model('best_model1.keras')
path_dir = 'uploads'
image_height, image_width = (150,150)
disease_names = np.array(['Eczema',
'Warts Molluscum and other Viral Infections',
'Melanoma',
'Atopic Dermatitis',
'Basal Cell Carcinoma (BCC)',
'Melanocytic Nevi (NV)',
'Benign Keratosis-like Lesions (BKL)',
'Psoriasis pictures Lichen Planus and related diseases',
'Seborrheic Keratoses and other Benign Tumors',
'Tinea Ringworm Candidiasis and other Fungal Infections'])

def prepare_input(image_input):
    img = image_input.resize((150,150))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def diagnose(input_img):
    prediction = Dermatologist.predict(input_img)
    report = np.argmax(prediction)
    return disease_names[report]