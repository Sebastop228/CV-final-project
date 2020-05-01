import numpy as np
import pandas as pd
import os


mean = np.zeros(3,)
stdev = np.zeros(3,)


def get_data(normalize):

    global mean
    global stdev
    print("PREPROCESSING...")
    data_dir = '../../../../data/' # Change this so it refers to where you have the data

    main_dir = os.path.abspath(__file__)

    path = os.path.normpath(main_dir + data_dir + 'fer2013.csv')

    data_file = open(path)

    data = pd.read_csv(data_file).values
    #print(data.shape)

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

#     path = os.path.normpath(main_dir + data_dir + 'fer2013.csv')

#     data_file = open(path)

#     data = pd.read_csv(data_file)


#     datapoints = data['pixels'].tolist()


#     images = []
#     for xseq in datapoints:
#         xx = [int(xp) for xp in xseq.split(' ')]
#         xx = np.asarray(xx).reshape(48, 48)
#         images.append(xx.astype('float32'))

#     images = np.asarray(images)


#     labels = pd.get_dummies(data['emotion']).to_numpy()

#     images_scaled = images / 255.0

#     mean = np.mean(images_scaled, axis=(0,1,2))
#     stdev = np.std(images_scaled, axis=(0,1,2))

#     print("Got mean and std!")

#     print("Dataset mean: ", mean)

#     print("Dataset std: ", stdev)

#     if normalize:
#         print("Normalizing...")
#         images = pre_process_fn(images)

#     images = np.expand_dims(images, -1)


#     print("PREPROCESSING DONE!")
#     return images, labels

def standardize(img):

    global mean
    global stdev

    img = (img - mean) / stdev

    return img

def pre_process_fn(img):
    img /= 255.
    standardize(img)

    return img