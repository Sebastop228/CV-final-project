import tensorflow as tf 
import numpy as np
import hyperparameters as hp
import hyperparameters2 as hp2
# import pandas as pd  # You're going to need to install this
import csv
import os
import argparse
from model2 import Model
from preprocess import *
from sklearn.model_selection import train_test_split


data_dir = '../../data/' # Change this so it refers to where you have the data

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!")
    parser.add_argument(
        '--load-checkpoint',
        action='store_true',
        help='''1 to load checkpoint, 0 to train from scratch''')
    return parser.parse_args()

def train(model, train_labels, train_images):

    amt_to_train = train_images.shape[0]
    batch_size = hp2.batch_size 

    for i in range(0, amt_to_train, batch_size):
        # indexed into the arrays so that, for the last batch, i+batch_size doesn't go over the size of the array
        batch_images = train_images[i:min(i+batch_size, amt_to_train)]
        batch_labels = train_labels[i:min(i+batch_size, amt_to_train)]
        with tf.GradientTape() as tape:
            # Had to expand dimmensions so things would work
            # batch_images = np.expand_dims(batch_images, axis=3) # NOT NECESSARY FOR MODEL 2
            probs = model.call(batch_images)
            # print(probs)
            # exit(0)
            loss = model.loss_fn(batch_labels, probs)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))



def test(model, test_labels, test_images):
    amt_to_test = test_images.shape[0]
    batch_size = hp2.batch_size
    amt_correct = 0
    for i in range(0, amt_to_test, batch_size):
        batch_images = test_images[i:min(i+batch_size, amt_to_test)]
        batch_labels = test_labels[i:min(i+batch_size, amt_to_test)]
        # Also had to expand dimmensions here
        # batch_images = np.expand_dims(batch_images, axis=3) # NOT NECESSARY FOR MODEL 2
        probs = model.call(batch_images)

        #figure out how we want to calculate accuracy - accuracy function in model?? Would pass it labels and logits

        amt_correct += model.accuracy_fn(batch_labels, probs)
    return amt_correct/amt_to_test


def main():
    ################# IMPORTANT ############################
    #If you want to run model 1, be sure to comment back the proper get_data (below) and uncomment 
    # the one for model 2. Also, be sure to in the header switch to "from model import Model" instead of
    # "model2 import Model". Lastly, switch all the "hp2"s to "hp"




    ################# FOR MODEL 1 ##########################
    # train_images, train_labels, test_images, test_labels = get_data()

    ################# FOR MODEL 1 ##########################


    ################# FOR MODEL 2 ##########################
    x, y = get_data_for_model2()
    train_images, test_images, train_labels, test_labels = train_test_split(x, y, test_size=0.1, random_state=42)
    # train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=41)
    ################# FOR MODEL 1 ##########################

    model = Model()
    # Putting input shape here instead
    model(tf.keras.Input(shape=(48, 48, 1)))

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
        print("Training for epoch ", i, "out of ", hp2.num_epochs)
        train(model, train_labels, train_images)
        print("Testing for epoch ", i, "out of ", hp2.num_epochs)
        accuracy = test(model, test_labels, test_images)
        if i % 4 == 3:
            epoch.assign(i + 1)
            save_path = manager.save()
            print("Saved checkpoint for epoch {}: {}".format(i, save_path))

        print("Epoch ", i, " accuracy is ", accuracy)

ARGS = parse_args()

if __name__ == '__main__':
    main()