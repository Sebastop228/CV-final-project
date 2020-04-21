import numpy as np
import pandas as pd
import hyperparameters as hp
import os
def get_data():
    
    data_dir = '../../../../data/' # Change this so it refers to where you have the data

    main_dir = os.path.abspath(__file__)

    train_path = os.path.normpath(main_dir + data_dir + 'train.csv')
    #test_path = os.path.normpath(main_dir + data_dir + 'test.csv')

    train_file = open(train_path)
    #test_file = open(test_path)

    train_data = pd.read_csv(train_file)
    #test_data = pd.read_csv(test_file)
    
    images = np.array([np.array([np.float64(x) for x in img.split(' ')]).reshape(48,48) 
                        for img in train_data['pixels'].values])
    total_num = images.shape[0]
    num_training = int(total_num * hp.percent_training)

    train_images = images[:num_training]
    print("got training images")
    #print(train_images.shape)
    test_images = images[num_training:]
    print("got testing images")
    #print(test_images.shape)

    labels = np.array([np.float64(x) for x in train_data['emotion'].values])

    train_labels = labels[:num_training]
    print("got training labels")
    #print(train_labels.shape)
    test_labels = labels[num_training:]
    print("got testing labels")
    #print(train_labels.shape)

    return train_images, train_labels, test_images, test_labels