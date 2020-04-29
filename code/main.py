import tensorflow as tf 
import numpy as np
import hyperparameters as hp
import hyperparameters2 as hp2
import cv2
# import pandas as pd  # You're going to need to install this
import csv
import os
import argparse
from model import Model
from model2 import Model
from preprocess import *

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

    amt_to_train = train_images.shape[0] #Amt of images we're training over
    batch_size = hp2.batch_size # make batch size a hyperparameter, either in model or hyperparameters.py

    for i in range(0, amt_to_train, batch_size):
        # indexed into the arrays so that, for the last batch, i+batch_size doesn't go over the size of the array

        # print("TRAINING: batch ", i, "out of ", amt_to_train)

        batch_images = train_images[i:min(i+batch_size, amt_to_train)]
        batch_labels = train_labels[i:min(i+batch_size, amt_to_train)]
        with tf.GradientTape() as tape:
<<<<<<< Updated upstream
            # Had to expand dimmensions so things would work
            batch_images = np.expand_dims(batch_images, axis=3)
=======
            # Had to expand dimensions so things would work
            # batch_images = np.expand_dims(batch_images, axis=3) # NOT NECESSARY FOR MODEL 2
>>>>>>> Stashed changes
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
        batch_images = np.expand_dims(batch_images, axis=3)
        probs = model.call(batch_images)

        #figure out how we want to calculate accuracy - accuracy function in model?? Would pass it labels and logits

        amt_correct += model.accuracy_fn(batch_labels, probs)
    return amt_correct/amt_to_test


def main():

    train_images, train_labels, test_images, test_labels = get_data()

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

#based off of https://github.com/atulapra/Emotion-detection/blob/master/src/emotions.py
#and https://realpython.com/face-detection-in-python-using-a-webcam/
#put in proper spot
#put in correct filepath
model.load_weights()

 # prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotions = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

feed = cv2.VideoCapture(0)

while True:
    #read frame-by-frame
    ret, frame = video_capture.read()

    #convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect face
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # Draw a rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


if __name__ == '__main__':
    main()