import tensorflow as tf
import os
import random
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import skimage.measure
from model_definition import *
import glob
from keras.preprocessing import image

# load latest jpg file in root directory, load traine UNET model, segment image, and count the cells
def predict_image ():
    
    # load latest file of type .jpg
    def latest_file():
        list_of_files = glob.glob('*.jpg')
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        return latest_file
        
    # specify model architecture
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    kernel_size = 8

    model = unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, kernel_size);
    
    #Load saved UNET model
    unet_model_name = 'checkpoint_unet.h5'
    checkpoint_filepath = unet_model_name
    model.load_weights(checkpoint_filepath);
    #print('\nModel loaded as {}\n'.format(checkpoint_filepath))
    
    image_path = latest_file()
    
    x = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    print(len(x))
    sizes_test = []

    img = imread(image_path)
    #sizes_test.append([img.shape[0], img.shape[1]])
    #img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    #x = ((img))
    imshow(img)
    
    #print(x)
    #y = model.predict(x, verbose=1)
   # z = (y > 0.5).astype(np.uint8)
    
   # a = 1
   # fig, a = plt.subplots(nrows=1, ncols=3, figsize=(10,10))

   # ax[0].set_title("Input")
  #  ax[0].imshow(x[a])

   # ax[1].set_title("Predicted w/o Threshold")
   # ax[1].imshow(np.squeeze(y[a]), cmap='gray')

   # ax[2].set_title("Predicted with Threshold")
   # ax[2].imshow(np.squeeze(z[a]), cmap='gray')

   # for a in number:
   #   a.axis("off")

   # plt.tight_layout()
   # plt.show()

   # limg = skimage.measure.label(z[a], connectivity=2, return_num=True)

   # print("mask w/ threshold cell count: ", np.max(limg[0]))
    
    
# test the segmentation and counting on a random training image
def test_prediction (preds_test, X_test, preds_test_t):
    
    # Perform predictions on a random training sample of ID ix
    ix = random.randint(0, len(preds_train_t))
    print(ix)
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,10))

    ax[0].set_title("Input")
    ax[0].imshow(X_train[ix])

    ax[1].set_title("Ground Truth")
    ax[1].imshow(np.squeeze(Y_train[ix]))

    ax[2].set_title("Predicted w/o Threshold")
    ax[2].imshow(np.squeeze(preds_train[ix]), cmap='gray')

    ax[3].set_title("Predicted with Threshold")
    ax[3].imshow(np.squeeze(preds_train_t[ix]), cmap='gray')

    for a in ax:
      a.axis("off")

    plt.tight_layout()
    plt.show()
    
    # Calculate Intersection Over Union (IOU) score:
    def iou_score(truth, predicted):
        intersection = np.logical_and(truth, predicted)
        union = np.logical_or(truth, predicted)
        iou_score = np.sum(intersection) / np.sum(union)

        return iou_score

    # display IOU score 
    iou = iou_score(Y_train, preds_train_t)
    print("IOU score: ", round(iou, 2))
    
    # Utilise connected component analysis (CCA) to count cells in random (ix) image:
    labeled_image = skimage.measure.label(preds_train_t[ix], connectivity=2, return_num=True)

    print("mask w/ threshold cell count: ", np.max(labeled_image[0]))
   
# test the segmentation and counting on a given image id
def prediction_test_number (number, preds_test, X_test, preds_test_t):
    
    # Perform predictions on some testing samples
    ix = number
    print(number)
    fig, number = plt.subplots(nrows=1, ncols=3, figsize=(10,10))

    ax[0].set_title("Input")
    ax[0].imshow(X_test[number])

    ax[1].set_title("Predicted w/o Threshold")
    ax[1].imshow(np.squeeze(preds_test[number]), cmap='gray')

    ax[2].set_title("Predicted with Threshold")
    ax[2].imshow(np.squeeze(preds_test_t[number]), cmap='gray')

    for a in number:
      a.axis("off")

    plt.tight_layout()
    plt.show()

    limg = skimage.measure.label(preds_test_t[number], connectivity=2, return_num=True)

    print("mask w/ threshold cell count: ", np.max(limg[0]))