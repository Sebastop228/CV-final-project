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

<<<<<<< Updated upstream
    train_data = pd.read_csv(train_file)
    #test_data = pd.read_csv(test_file)
=======
    data = pd.read_csv(data_file).values
    #print(data.shape)

    train_data = np.array([lst for lst in data if lst[2] == 'Training'])
    #print(train_data.shape)

    test_data = np.array([lst for lst in data if lst[2] == 'PublicTest'
                or lst[2] == 'PrivateTest'])
    #print(test_data.shape)
    
    train_images = np.array([np.array([np.float32(x) for x in img.split(' ')]).reshape(48,48) 
                        for img in train_data[:,1]])
    print("got training images")
    # print(train_images.shape)
>>>>>>> Stashed changes
    
    images = np.array([np.array([np.float64(x) for x in img.split(' ')]).reshape(48,48) 
                        for img in train_data['pixels'].values])
    total_num = images.shape[0]
    num_training = int(total_num * hp.percent_training)

    train_images = images[:num_training]
    print("got training images")
    #print(train_images.shape)
    test_images = images[num_training:]
    print("got testing images")
<<<<<<< Updated upstream
    #print(test_images.shape)
=======
    # print(test_images.shape)

    train_labels = np.array([np.float32(x) for x in train_data[:,0]])
    print("got training labels")
    # print(train_labels.shape)
>>>>>>> Stashed changes

    labels = np.array([np.float64(x) for x in train_data['emotion'].values])

    train_labels = labels[:num_training]
    print("got training labels")
    #print(train_labels.shape)
    test_labels = labels[num_training:]
    print("got testing labels")
<<<<<<< Updated upstream
    #print(train_labels.shape)
=======
    # print(test_labels.shape)
>>>>>>> Stashed changes

    return train_images, train_labels, test_images, test_labels