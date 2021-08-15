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

def training_model():
    # Define common variables:
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    TRAIN_PATH = 'train/'
    TEST_PATH = 'test/'

    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]

    # Define X train and Y train tensors:
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)

    print('\nReading training images\n')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')

    # Data Cleaning:
    start = timer()

    print('\nResizing training images and masks\n')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img  # Fill empty X_train with values from img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)  

        Y_train[n] = mask   

    # Test images:
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

    # Data Cleaning:
    sizes_test = []
    print('\nResizing test images\n') 
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img

    print('\nDone!\n')
    end = timer()
    print("\nTime taken load data: ", end - start, "seconds\n") 

    # Check tensor shapes:
    print('\nChecking tensor shapes\n')
    X_train = tf.random.shuffle(X_train, seed=101).numpy()
    Y_train = tf.random.shuffle(Y_train, seed=101).numpy()
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)

    kernel_size = 8

    model = unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, kernel_size)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='accuracy'),
        tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)
    ]
    print('\nStarting training of model\n')
    start = timer()

    results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=32, epochs=200, callbacks=callbacks)
    # results = model.fit(X_train[:100], Y_train[:100], epochs=250, callbacks=callbacks)

    end = timer()
    print("\nTime taken for model to run: ", end - start, "seconds\n") 

    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # Plot the training and validation accuracy and loss at each epoch
    loss = results.history['loss']
    val_loss = results.history['val_loss']
    epochs = range(1, len(loss) + 1)

    acc = results.history['accuracy']
    val_acc = results.history['val_accuracy']

    preds_train = model.predict(X_train, verbose=1)
    preds_test = model.predict(X_test, verbose=1)
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    # Calculate Intersection Over Union (IOU) score:
    def iou_score(truth, predicted):
        intersection = np.logical_and(truth, predicted)
        union = np.logical_or(truth, predicted)
        iou_score = np.sum(intersection) / np.sum(union)

        return iou_score

    iou = iou_score(Y_train, preds_train_t)
    print("IOU score: ", round(iou, 2))

    # Save trained model
    unet_model_name = 'checkpoint_unet.h5'
    model.save(unet_model_name)
    print('\nModel saved as {}\n'.format(unet_model_name))