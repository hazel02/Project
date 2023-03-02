# -*- coding: utf-8 -*-
"""

@author: USER PC
"""


#loading base packages
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
import glob
import cv2
import csv
import os, os.path
from os import path
import pywt
import sys
from PIL import Image



#loading keras packages
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, BatchNormalization, Dropout, Flatten, Dense, Activation, MaxPool2D, Conv2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.vgg19 import preprocess_input as vgg19_preprocessor
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.metrics import Recall,Precision
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Flatten
from tensorflow.python.keras.preprocessing import image as image_preprocessor

from image_feature_extractor.extractors.extractor import Extractor

#loading sklearn packages
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report

#import images for preprocessing

def get_data_from_image(image_path):
    cv_img = cv2.imread(image_path)
    (means, stds) = cv2.meanStdDev(cv_img)
    stats = np.concatenate([means, stds]).flatten()
    image_features_list = [stats.tolist()]
    return image_features_list

images_dir = 'C:\\Users\\USER PC\\Desktop\\MSc Project Work\\Implementation\\Images'

images_names = []
 
with os.scandir(images_dir) as dirs:
    for entry in dirs:
        images_names.append(entry.name)


for image in images_names:

    path = images_dir + image

    image_features_list =  get_data_from_image(path)

    print(image_features_list)

#Data Augmentation
''' # Resize the image to (512, 512)
  # Default for  using hog as feature extractor
   
resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

# Compute the Hog Features; HOG is a feature descriptor used to extract features from images
h = hog.compute(resized)
# Transpose the result from a vector to an array
hogImage = h.T

# append the category of the
# image to a category container
if folder == "Glaucoma":
                category.append(0) #0 --> Glaucoma
elif folder == "Cataract":
                category.append(1) #1 --> Cataract
elif folder == "AMD":
                category.append(2) #2 --> AMD
else :
                category.append(3) #3 --> DR
            
# append the extracted features of
# the image to a category container
            hogArray.append(hogImage)


# convert the extracted features
# from array to vector
hogArray_np = np.array(hogArray)


# Reshaped the Features to the acurrate size
reshaped_hog_Array = np.reshape(
    hogArray_np, (hogArray_np.shape[0], hogArray_np.shape[1]))


# setup PCA for dimensionality reduction
pca = PCA(n_components=NUM_FEATURES)
reduced_features = pca.fit_transform(reshaped_hog_Array)
features = reduced_features.tolist()

# Create a container to hold data to be saved into csv
csvData = []
for id, line in enumerate(features):
    newImg = line

    # Prepend the category of each image to
    # the begining of the features
    newImg.insert(0, category[id])
    csvData.append(newImg)

# Save the csv file
filename = str(NUM_FEATURES) + '_extracted_features.csv'
with open(filename, 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()

print("Done Extracting Features")'''

from skimage import io
from skimage.transform import resize


def create_directory(directory):
    '''
    Creates a new folder in the specified directory if the folder doesn't exist.
    INPUT
        directory: Folder to be created, called as "folder/".
    OUTPUT
        New folder in the current directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
def crop_and_resize_images(path, new_path, cropx, cropy, img_size=512):
    '''
    Crops, resizes, and stores all images from a directory in a new directory.
    INPUT
        path: Path where the current, unscaled images are contained.
        new_path: Path to save the resized images.
        img_size: New size for the rescaled images.
    OUTPUT
        All images cropped, resized, and saved from the old folder to the new folder.
    '''
    create_directory(new_path)
    dirs = [l for l in os.listdir(path) if l != '.DS_Store']
    total = 0

    for item in dirs:
        img = io.imread(path+item)
        y,x,channel = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        img = img[starty:starty+cropy,startx:startx+cropx]
        img = resize(image, (512,512))
        io.imsave(str(new_path + item), img)
        total += 1
        print("Saving: ", item, total)


if __name__ == '__main__':
    crop_and_resize_images(path='../data/train/', new_path='../data/train-resized-256/', cropx=1800, cropy=1800, img_size=256)
    crop_and_resize_images(path='../data/test/', new_path='../data/test-resized-256/', cropx=1800, cropy=1800, img_size=256)

#preprocessing images
def prepocess_images(X):
    image_list=[]
    
    for x in X:
        image_list.append(preprocess_input(x))
        
    return np.array(image_list)






#Vgg 19 for extraction
class DeepExtractor(Extractor):
    
    def __init__(self, base_route, model_name: str, size=224, batch_size=128):
        
        super().__init__(base_route=base_route, size=size, batch_size=batch_size)
        
        self.model = None
        self.file_writer = None
        self.model_preprocess = None
        self.model_name = model_name
        
        self._set_extractor_model(self.model_name)
    
    def _set_extractor_model(self, model_name: str) -> None:
        if model_name == "vgg19":
            self.model = VGG19(include_top=False, weights="imagenet", input_shape=self.image_shape)
            self.model_preprocess = vgg19_preprocessor
            
        else:
            raise Exception("Invalid pre-trained Keras Application")

def extract(self, image_route: str) -> np.ndarray:
        image = image_preprocessor.load_img(image_route, target_size=(self.width, self.height))
        image = np.expand_dims(image_preprocessor.img_to_array(image), axis=0)
        preprocessed_img = self.model_preprocess(image)
        
        return self.model.predict(preprocessed_img).flatten()
    
def _find_features_size(self) -> int:
        example_image_route = os.path.join(self.base_route, self.directory_iterator.filenames[0])
        return len(self.extract(image_route=example_image_route))









