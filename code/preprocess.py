import numpy as np
import pandas as pd
import hyperparameters as hp
import os


mean = np.zeros(3,)
stdev = np.zeros(3,)

def get_data(normalize):

    global mean
    global stdev

    # Reading in info from CSV

    data_dir = '../../../../data/' # Change this so it refers to where you have the data


    main_dir = os.path.abspath(__file__)

    path = os.path.normpath(main_dir + data_dir + 'fer2013.csv')

    data_file = open(path)

    data = pd.read_csv(data_file).values

    # Separating into test and training data

    train_data = np.array([lst for lst in data if lst[2] == 'Training'])

    test_data = np.array([lst for lst in data if lst[2] == 'PublicTest'
                or lst[2] == 'PrivateTest'])

    # Converting things to np arrays

    train_images = np.array([np.array([np.float32(x) for x in img.split(' ')]).reshape(48,48)
                        for img in train_data[:,1]])
    print("got training images")
    print(train_images.shape)

    test_images = np.array([np.array([np.float32(x) for x in img.split(' ')]).reshape(48,48)
                        for img in test_data[:,1]])
    print("got testing images")
    print(test_images.shape)

    train_labels = np.array([np.float32(x) for x in train_data[:,0]])
    print("got training labels")
    print(train_labels.shape)

    test_labels = np.array([np.float32(x) for x in test_data[:,0]])
    print("got testing labels")
    print(test_labels.shape)

    # Calculating mean and std of dataset for normalization

    train_images_scaled = train_images / 255.

    mean = np.mean(train_images_scaled, axis=(0,1,2))
    stdev = np.std(train_images_scaled, axis=(0,1,2))

    print("got mean and std")

    print("Dataset mean: ", mean)

    print("Dataset std: ", stdev)

    # Performing data normalization
    if normalize:

        for img in train_images:
            img = pre_process_fn(img)

        test_images = normalize_test(test_images)
            
        # train_images = pre_process_fn(train_images)

    # train_images = np.expand_dims(train_images, -1)
    # test_images = np.expand_dims(test_images, -1)
    print("PREPROCESSING DONE!")
    return train_images, train_labels, test_images, test_labels



# def get_data(normalize):
#     print("PREPROCESSING...")
    
#     global mean
#     global stdev

#     data_dir = '../../../../data/' # Change this so it refers to where you have the data

#     main_dir = os.path.abspath(__file__)


    return train_images, train_labels, test_images, test_labels

def standardize(img):

    global mean
    global stdev

    img = (img - mean) / stdev

    return img

def pre_process_fn(img):
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