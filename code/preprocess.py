import numpy as np
import pandas as pd
import os

#Global mean and stdev variables
mean = np.zeros(3,)
stdev = np.zeros(3,)


def get_data(normalize):

    """ Get the data and preprocess it """

    global mean
    global stdev
    print("PREPROCESSING...")
    data_dir = '../../../../data/' # Change this so it refers to where you have the data

    main_dir = os.path.abspath(__file__)

    path = os.path.normpath(main_dir + data_dir + 'fer2013.csv')

    data_file = open(path)

    data = pd.read_csv(data_file).values

    train_data = np.array([lst for lst in data if lst[2] == 'Training'])

    test_data = np.array([lst for lst in data if lst[2] == 'PublicTest'
                or lst[2] == 'PrivateTest'])
    
    train_images = np.array([np.array([np.float32(x) for x in img.split(' ')]).reshape(48,48) 
                        for img in train_data[:,1]])
    print("got training images")
    
    test_images = np.array([np.array([np.float32(x) for x in img.split(' ')]).reshape(48,48) 
                        for img in test_data[:,1]])
    print("got testing images")

    train_labels = np.array([np.float32(x) for x in train_data[:,0]])
    print("got training labels")


    test_labels = np.array([np.float32(x) for x in test_data[:,0]])
    print("got testing labels")

    train_images_scaled = train_images / 255.

    mean = np.mean(train_images_scaled, axis=(0,1,2))
    stdev = np.std(train_images_scaled, axis=(0,1,2))

    print("got mean and std")

    print("Dataset mean: ", mean)

    print("Dataset std: ", stdev)

    # Performing data normalization
    if normalize:
        for i in range(len(train_images)):
            train_images[i] = pre_process_fn(train_images[i])

    print("PREPROCESSING DONE!")
    return train_images, train_labels, test_images, test_labels


def standardize(img):

    """ A method to standardize the images """

    global mean
    global stdev

    img = (img - mean) / stdev

    return img

def pre_process_fn(img):

    """A method to preprocess the images """

    img /= 255.
    standardize(img)

    return img

def normalize_test(test_images):
    test_mean = np.mean(test_images / 255., axis=(0,1,2))
    test_std = np.std(test_images / 255., axis=(0,1,2))

    for img in test_images:
        img /= 255.
        img = (img - test_mean) / test_std
    
    return test_images
