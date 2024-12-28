from django.http import JsonResponse
from django.shortcuts import render
from django.http import JsonResponse
from django.shortcuts import render
from django.views import View
import os
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
import warnings
import matplotlib.pyplot as plt

from gan_app.gan.generate import generate_data
from gan_app.gan.train import train_gan

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load and preprocess MNIST dataset
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.astype('float32') / 127.5 - 1.
x_train = np.expand_dims(x_train, axis=-1)

def train(request):
    result = train_gan(x_train)
    return JsonResponse({"message": "Training complete."})

def generate(request):
    result = generate_data()
    return JsonResponse({"message": "Images generated and displayed."})

