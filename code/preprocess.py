import numpy as np
import pandas as pd
import hyperparameters as hp
import os
def get_data():
    
    data_dir = '../../../../data/' # Change this so it refers to where you have the data

    main_dir = os.path.abspath(__file__)

    path = os.path.normpath(main_dir + data_dir + 'fer2013.csv')

    data_file = open(path)

    data = pd.read_csv(data_file).values


    train_data = np.array([lst for lst in data if lst[2] == 'Training'])
    #print(train_data.shape)

    test_data = np.array([lst for lst in data if lst[2] == 'PublicTest'
                or lst[2] == 'PrivateTest'])
    #print(test_data.shape)
    
    train_images = np.array([np.array([np.float32(x) for x in img.split(' ')]).reshape(48,48) 
                        for img in train_data[:,1]])
    print("got training images")
    # print(train_images.shape)
    train_images -= np.mean(train_images, axis=0)
    train_images /= np.std(train_images, axis=0)
    
    test_images = np.array([np.array([np.float32(x) for x in img.split(' ')]).reshape(48,48) 
                        for img in test_data[:,1]])

    print("got testing images")
    # print(test_images.shape)

    train_labels = np.array([np.float32(x) for x in train_data[:,0]])
    print("got training labels")
    # print(train_labels.shape)


    test_labels = np.array([np.float32(x) for x in test_data[:,0]])
    print("got testing labels")
    # print(test_labels.shape)

    return train_images, train_labels, test_images, test_labels



def get_data_for_model2():
    print("PREPROCESSING FOR MODEL 2")
    data_dir = '../../../data/' # Change this so it refers to where you have the data

    main_dir = os.path.abspath(__file__)

    path = os.path.normpath(main_dir + data_dir + 'fer2013.csv')

    data_file = open(path)

    data = pd.read_csv(data_file)


    datapoints = data['pixels'].tolist()


    X=[]

    X = []
    for xseq in datapoints:
        xx = [int(xp) for xp in xseq.split(' ')]
        xx = np.asarray(xx).reshape(48, 48)
        X.append(xx.astype('float32'))

    X = np.asarray(X)
    X = np.expand_dims(X, -1)

    y = pd.get_dummies(data['emotion']).to_numpy()

    # X /=127.0
    # X -= 1

    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
s

    print("PREPROCESSING DONE!")
    return X,y