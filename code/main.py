import tensorflow as tf 
import numpy as np
import pandas as pd  # You're going to need to install this
import csv
import os
import argparse
import cv2
from model2 import Model
from model import first_Model
from preprocess import *


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!")
    parser.add_argument(
        '--load-checkpoint',
        action='store_true',
        help='''1 to load checkpoint, 0 to train from scratch''')
    parser.add_argument(
        '--live-feed',
        action = 'store_true',
        help = '''Pass this argument to do live emotion recognition '''
    )
    parser.add_argument(
        '--normalize-data',
        action='store_true',
        help=''' add this flag if you want to normalize the input data '''
    )
    parser.add_argument(
        '--augment-data',
        action='store_true',
        help=''' add this flag if you want to augment the input data '''
    )
    parser.add_argument(
        '--first-model',
        action='store_true',
        help=''' Use the first model (default is second model) '''
    )
    return parser.parse_args()

def train(augment, model, train_labels, train_images, validation_data):

    if augment:

        train_images = np.expand_dims(train_images, axis=3)

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    preprocessing_function = pre_process_fn,
                    rotation_range=20,
                    #width_shift_range=0.2,
                    #height_shift_range=0.2,
                    #horizontal_flip=True
                    )

        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=7)
        model.fit(datagen.flow(train_images, train_labels, batch_size=model.batch_size),
                    steps_per_epoch=len(train_images) / model.batch_size, 
                    epochs=model.num_epochs,
                    validation_data= validation_data)

    else:

        #amt_to_train = train_images.shape[0]
        #batch_size = model.batch_size 

        #for i in range(0, amt_to_train, batch_size):
            # indexed into the arrays so that, for the last batch, i+batch_size doesn't go over the size of the array
            #batch_images = train_images[i:min(i+batch_size, amt_to_train)]
            #batch_labels = train_labels[i:min(i+batch_size, amt_to_train)]
            #with tf.GradientTape() as tape:
                #batch_images = np.expand_dims(batch_images, axis=3) # NOT NECESSARY FOR MODEL 2
                #probs = model.call(batch_images)
                #batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=7)
                #loss = model.loss_fn(batch_labels, probs)
            #gradients = tape.gradient(loss, model.trainable_variables)
            #model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_images = np.expand_dims(train_images, axis=3)
        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=7)

        model.fit(train_images, train_labels, batch_size=model.batch_size,
                  steps_per_epoch=len(train_images) / model.batch_size, 
                  epochs=model.num_epochs,
                  validation_data= validation_data)



def test(model, test_labels, test_images):
    
    #if normalize:
        #test_images /= 255.
        #mean = np.mean(test_images, axis=(0,1,2))
        #stdev = np.std(test_images, axis=(0,1,2))

        #test_images = (test_images - mean) / stdev
    
    #amt_to_test = test_images.shape[0]
    #batch_size = model.batch_size
    #amt_correct = 0
    #for i in range(0, amt_to_test, batch_size):
        #batch_images = test_images[i:min(i+batch_size, amt_to_test)]
        #batch_labels = test_labels[i:min(i+batch_size, amt_to_test)]
        #batch_images = np.expand_dims(batch_images, axis=3) # NOT NECESSARY FOR MODEL 2
        #probs = model.call(batch_images)
        #batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=7)
        #amt_correct += model.accuracy_fn(batch_labels, probs)
    #return amt_correct/amt_to_test

    model.evaluate(test_images, test_labels, batch_size=model.batch_size)


def main():

    if ARGS.first_model:
        print("Using model 1!")
        model = first_Model()
    else:
        print("Using model 2!")
        model = Model()
    model(tf.keras.Input(shape=(48, 48, 1)))
    if not ARGS.live_feed:

        normalize = False
        augment = False

        if ARGS.normalize_data:
            normalize = True

        train_images, train_labels, test_images, test_labels = get_data(normalize)
        # train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

        epoch = tf.Variable(0, trainable=False)
        checkpoint = tf.train.Checkpoint(epoch=epoch, model=model)
        manager = tf.train.CheckpointManager(checkpoint, './checkpoints', max_to_keep=3)

        if ARGS.load_checkpoint:
            checkpoint.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                print("Restored from {}".format(manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

        if ARGS.augment_data:
            augment = True
            model.compile(loss='categorical_crossentropy', metrics= ['categorical_accuracy'])

            test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=7)

            test_images = normalize_test(test_images)
            test_images = np.expand_dims(test_images, axis=3)


            validation_data = (test_images, test_labels)
            train(augment, model, train_labels, train_images, validation_data)

            #results = model.evaluate(test_images, test_labels, batch_size=model.batch_size)
            results = test(model, test_labels, test_images)
            print('test loss, test acc:', results)

        else:


            #for i in range(epoch.numpy(), model.num_epochs, 1):
                #print("Training for epoch ", i, "out of ", model.num_epochs)
                #train(augment, model, train_labels, train_images, [])
                #print("Testing for epoch ", i, "out of ", model.num_epochs)
                #accuracy = test(normalize, model, test_labels, test_images)
                #if i % 4 == 3:
                    #epoch.assign(i + 1)
                    #save_path = manager.save()
                    #print("Saved checkpoint for epoch {}: {}".format(i, save_path))

                #print("Epoch ", i, " accuracy is ", accuracy)
            model.compile(loss='categorical_crossentropy', metrics= ['categorical_accuracy'])

            test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=7)
            test_images = np.expand_dims(test_images, axis=3)

            validation_data = (test_images, test_labels)
            train(augment, model, train_labels, train_images, validation_data)

            #results = model.evaluate(test_images, test_labels, batch_size=model.batch_size)
            results = test(model, test_labels, test_images)
            print('test loss, test acc:', results)
            

        model.save_weights('model.h5')
    
    elif ARGS.live_feed:
        print("Doing emotion recognition from live-feed!")

        #based off of https://github.com/atulapra/Emotion-detection/blob/master/src/emotions.py
        #and https://realpython.com/face-detection-in-python-using-a-webcam/
        model.load_weights('model.h5')

        model.compile(model.optimizer, loss = model.loss_fn)

        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)

        # dictionary which assigns each label an emotion (alphabetical order)
        emotions = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        path = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(path)
        feed = cv2.VideoCapture(0)

        while True:
            #read frame-by-frame
            ret, frame = feed.read()

            #convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #detect face
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                # Draw a rectangle around the faces
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, emotions[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        feed.release()
        cv2.destroyAllWindows()


ARGS = parse_args()

if __name__ == '__main__':
    main()