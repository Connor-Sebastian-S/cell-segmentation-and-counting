import tensorflow as tf
import os
import random
from datetime import datetime
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import skimage.measure
from keras.preprocessing import image

from model_definition import unet

# load latest jpg file in root directory, load traine UNET model, segment image, and count the cells
def predict_image ():
    
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    input_path = 'images/'

    # load all folder in the image folder
    # the folders are named by date dd-mm-yy
    # if the folder name is not equal to todays ate, remove from list
    input_ids = next (os.walk(input_path))[1]
    for l in input_ids:
        if l != datetime.today().strftime('%Y-%m-%d'):
            input_ids.remove(l)

    # format model input based on folder list - should only be 1
    X_input = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

    sizes_test = []

    # resize input image, assuming image is called 'image.jpg'
    for n, id_ in tqdm(enumerate(input_ids), total=len(input_ids)): 
        path = input_path + id_
        print(path)
        img = imread(path + '/image.jpg')[:,:,:IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_input[n] = img

    end = timer()

    # define model parameters
    kernel_size = 8

    model = unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, kernel_size)

    #Load saved UNET model
    unet_model_name = 'checkpoint_unet.h5'
    checkpoint_filepath = unet_model_name
    model.load_weights(checkpoint_filepath);

    # prediction 
    preds_input = model.predict(X_input, verbose=1)
    preds_input_t = (preds_input > 0.5).astype(np.uint8)

    ix = 0
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,10))

    # show input image
    ax[0].set_title("Input")
    ax[0].imshow(X_input[0])

    # show non-thresholded image
    ax[1].set_title("Predicted without Threshold")
    ax[1].imshow(np.squeeze(preds_input[0]), cmap='gray')

    # show thresholded image
    ax[2].set_title("Predicted with Threshold")
    ax[2].imshow(np.squeeze(preds_input_t[0]), cmap='gray')

    for a in ax:
      a.axis("off")

    plt.tight_layout()
    plt.show()

    # predict cell count and display
    limg = skimage.measure.label(preds_input_t[ix], connectivity=2, return_num=True)
    print("Cell count: ", np.max(limg[0]))

#TODO
# generate report of prediction data and results
    
    
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