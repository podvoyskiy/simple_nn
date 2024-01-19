import numpy as np
import os.path
import requests
from Helper import Helper

class Dataset:
    __URI_DATASET = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    
    @staticmethod
    def load():
        if (os.path.isfile("mnist.npz") == False): Dataset.download()

        with np.load("mnist.npz") as f:
            #convert from RGB to Unit RGB
            x_train = f['x_train'].astype("float32") / 255

            #reshape from (60000, 28, 28) to (60000, 784)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

            #labels
            y_train = f['y_train']

            #convert to output layer format
            y_train = np.eye(10)[y_train]

            return x_train, y_train
        
    @classmethod
    def download(cls):
        response = requests.get(cls.__URI_DATASET)
        if (response.status_code != 200): raise Exception(Helper.log('error download dataset', Helper.COLOR_ERROR))
        with open('mnist.npz', 'wb') as f: f.write(response.content)