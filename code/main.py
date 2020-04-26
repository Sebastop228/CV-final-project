import tensorflow as tf 
import numpy as np
import hyperparameters as hp
import hyperparameters2 as hp2
import pandas as pd  # You're going to need to install this
import csv
import os
<<<<<<< Updated upstream
from model import Model
from preprocess import get_data
=======
import argparse
from model2 import Model #switch add or remove "2" from lowercase "model"
from preprocess import *
>>>>>>> Stashed changes



def train(model, train_labels, train_images):

    amt_to_train = train_images.shape[0] #Amt of images we're training over
    batch_size = hp2.batch_size # make batch size a hyperparameter, either in model or hyperparameters.py

    for i in range(0, amt_to_train, batch_size):
        # indexed into the arrays so that, for the last batch, i+batch_size doesn't go over the size of the array
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        # if(amt_to_train-i < batch_size):
        #     break
        print("TRAINING: batch ", i, "out of ", amt_to_train)
=======
=======
>>>>>>> Stashed changes

        # print("TRAINING: batch ", i, "out of ", amt_to_train)

>>>>>>> Stashed changes
        batch_images = train_images[i:min(i+batch_size, amt_to_train)]
        batch_labels = train_labels[i:min(i+batch_size, amt_to_train)]
        with tf.GradientTape() as tape:
            # Had to expand dimmensions so things would work
            batch_images = np.expand_dims(batch_images, axis=3)
            # print(batch_images.shape)
            probs = model.call(batch_images)
            loss = model.loss_fn(batch_labels, probs)
            # print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))



def test(model, test_labels, test_images):
    amt_to_test = test_images.shape[0]
    batch_size = hp2.batch_size
    amt_correct = 0
    for i in range(0, amt_to_test, batch_size):
        # if(amt_to_test-i < batch_size):
        #     break
        batch_images = test_images[i:min(i+batch_size, amt_to_test)]
        batch_labels = test_labels[i:min(i+batch_size, amt_to_test)]
        # Also had to expand dimmensions here
        batch_labels = np.expand_dims(batch_images, axis=3)
        probs = model.call(batch_images)

        amt_correct += model.accuracy_fn(batch_labels, probs)
    print("Found ", amt_correct, "correct out of ", amt_to_test)
    return amt_correct/amt_to_test


def main():

    train_images, train_labels, test_images, test_labels = get_data()


    model = Model()
    # Putting input shape here instead
    model(tf.keras.Input(shape=(48, 48, 1)))

<<<<<<< Updated upstream
    for i in range(hp.num_epochs):
        print("Training epoch ", i)
        train(model, train_labels, train_images)

        print("Testing epoch ", i)
=======
    epoch = tf.Variable(0, trainable=False)
    checkpoint = tf.train.Checkpoint(epoch=epoch, model=model)
    manager = tf.train.CheckpointManager(checkpoint, './checkpoints', max_to_keep=3)

    if ARGS.load_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    for i in range(epoch.numpy(), hp2.num_epochs, 1):
        print("Training for epoch ",i, "out of ", hp2.num_epochs)
        train(model, train_labels, train_images)
        print("Testing for epoch ", i, "out of ", hp2.num_epochs)
>>>>>>> Stashed changes
        accuracy = test(model, test_labels, test_images)

        print("Epoch ", i, " accuracy is ", accuracy)

if __name__ == '__main__':
    main()