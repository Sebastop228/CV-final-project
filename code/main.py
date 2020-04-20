import tensorflow as tf 
import numpy as np
import hyperparameters as hp
import pandas as pd  # You're going to need to install this
import csv
import os
from model import Model

data_dir = '../../../data/' # Change this so it refers to where you have the data


def train(model, train_labels, train_images):

    amt_to_train = train_images.shape[0] #Amt of images we're training over
    batch_size = hp.batch_size # make batch size a hyperparameter, either in model or hyperparameters.py

    for i in range(0, amt_to_train, batch_size):
        # indexed into the arrays so that, for the last batch, i+batch_size doesn't go over the size of the array
        batch_images = train_images[i:min(i+batch_size, amt_to_train)]
        batch_labels = train_labels[i:min(i+batch_size, amt_to_train)]
        with tf.GradientTape() as tape:
            # Had to expand dimmensions so things would work
            batch_images = np.expand_dims(batch_images, axis=3)
            probs = model.call(batch_images)
            loss = model.loss_fn(batch_labels, probs)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))



def test(model, test_labels, test_images):
    amt_to_test = test_images.shape[0]
    batch_size = hp.batch_size
    amt_correct = 0
    for i in range(0, amt_to_test, batch_size):
        batch_images = test_images[i:min(i+batch_size, amt_to_test)]
        batch_labels = test_labels[i:min(i+batch_size, amt_to_test)]
        # Also had to expand dimmensions here
        batch_labels = np.expand_dims(batch_images, axis=3)
        probs = model.call(batch_images)

        #figure out how we want to calculate accuracy - accuracy function in model?? Would pass it labels and logits

        amt_correct += model.accuracy_fn(batch_labels, probs)
    return amt_correct/amt_to_test


def main():

    main_dir = os.path.dirname(__file__)

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


    model = Model()
    # Putting input shape here instead
    model(tf.keras.Input(shape=(48, 48, 1)))

    for i in range(hp.num_epochs):
        train(model, train_labels, train_images)

        accuracy = test(model, test_labels, test_images)

        print("Epoch ", i, " accuracy is ", accuracy)

if __name__ == '__main__':
    main()