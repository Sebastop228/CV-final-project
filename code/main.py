import tensorflow as tf 
import numpy as np
import hyperparameters as hp
import pandas as pd  # You're going to need to install this
import csv
import os
import argparse
from model import Model
from preprocess import *

data_dir = '../../../data/' # Change this so it refers to where you have the data

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!")
    parser.add_argument(
        '--load-checkpoint',
        action='store_true',
        help='''1 to load checkpoint, 0 to train from scratch''')
    parser.add_argument(
        '--normalize-data',
        action='store_true',
        help=''' add this flag if you want to normalize the input data ''')
    parser.add_argument(
        '--augment-data',
        action='store_true',
        help=''' add this flag if you want to augment the input data ''')
    return parser.parse_args()

def train(augment, model, train_labels, train_images):

    if augment:

        train_images = np.expand_dims(train_images, axis=3)

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    featurewise_center=True,
                    preprocessing_function = pre_process_fn,
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True)

        batches = 0
        for x_batch, y_batch in datagen.flow(train_images, train_labels, batch_size=hp.batch_size):
            y_batch = tf.keras.utils.to_categorical(y_batch, num_classes=7)
            model.fit(x_batch, y_batch)
            batches += 1
            if batches >= len(train_images) / hp.batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
                break

    else:

        amt_to_train = train_images.shape[0] #Amt of images we're training over
        batch_size = hp.batch_size # make batch size a hyperparameter, either in model or hyperparameters.py

        for i in range(0, amt_to_train, batch_size):
        # indexed into the arrays so that, for the last batch, i+batch_size doesn't go over the size of the array

            print("TRAINING: batch ", i, "out of ", amt_to_train)

            batch_images = train_images[i:min(i+batch_size, amt_to_train)]
            batch_labels = train_labels[i:min(i+batch_size, amt_to_train)]
            with tf.GradientTape() as tape:
            # Had to expand dimmensions so things would work
                batch_images = np.expand_dims(batch_images, axis=3)
                probs = model.call(batch_images)
                loss = model.loss_fn(batch_labels, probs)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))



def test(normalize, model, test_labels, test_images):

    if normalize:
        mean = np.mean(test_images, axis=(0,1,2))
        stdev = np.std(test_images, axis=(0,1,2))

        test_images = (test_images - mean) / stdev

    amt_to_test = test_images.shape[0]
    batch_size = hp.batch_size
    amt_correct = 0
    for i in range(0, amt_to_test, batch_size):
        batch_images = test_images[i:min(i+batch_size, amt_to_test)]
        batch_labels = test_labels[i:min(i+batch_size, amt_to_test)]
        # Also had to expand dimmensions here
        batch_images = np.expand_dims(batch_images, axis=3)
        probs = model.call(batch_images)

        #figure out how we want to calculate accuracy - accuracy function in model?? Would pass it labels and logits

        amt_correct += model.accuracy_fn(batch_labels, probs)
    return amt_correct/amt_to_test


def main():

    normalize = False

    if ARGS.normalize_data:
        normalize = True
    
    train_images, train_labels, test_images, test_labels = get_data(normalize)

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

    augment = False

    if ARGS.augment_data:
        augment = True
        normalize = True
        model.compile(loss='categorical_crossentropy', metrics= ['categorical_accuracy'])

    for i in range(epoch.numpy(), hp.num_epochs, 1):
        train(augment, model, train_labels, train_images)
        accuracy = test(normalize, model, test_labels, test_images)
        if i % 4 == 3:
            epoch.assign(i + 1)
            save_path = manager.save()
            print("Saved checkpoint for epoch {}: {}".format(i, save_path))

        print("Epoch ", i, " accuracy is ", accuracy)

ARGS = parse_args()

if __name__ == '__main__':
    main()
