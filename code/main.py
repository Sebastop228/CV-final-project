import tensorflow as tf 
import numpy as np
import pandas as pd 
import csv
import os
import argparse
import datetime
import shutil
import cv2
from cm import ConfusionMatrixLogger
from model2 import Model
from model import first_Model
from preprocess import *

def parse_args():
    
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!"
    )
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Pass this argument to load latest checkpoint'''
    )
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights. In
        the case of task 2, passing a checkpoint path will disable
        the loading of VGG weights.'''
    )
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

def train(augment, model, train_labels, train_images, validation_data, checkpoint_path):

    """ Train the model on the training set of images """

    # Tensorboard:
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/" + curr_time
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch',
            profile_batch=0)


    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + \
                    "weights.e{epoch:02d}-" + \
                    "acc{val_categorical_accuracy:.4f}.h5",
            monitor='val_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True,
            period=5),
        tensorboard_callback
    ]

    # Include confusion logger in callbacks if flag set
    if ARGS.confusion:
        cm_dir = "logs/confusion_matrix_" + curr_time
        callback_list.append(ConfusionMatrixLogger(validation_data, cm_dir))

    #if one of the command lines includes data augmentation
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
        
        model.fit(datagen.flow(train_images, train_labels, batch_size=model.batch_size, shuffle=False),
                    steps_per_epoch=len(train_images) / model.batch_size, 
                    epochs=model.num_epochs,
                    validation_data= validation_data,
                    callbacks=callback_list)
    else:
        train_images = np.expand_dims(train_images, axis=3)
        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=7)

        model.fit(train_images, train_labels, batch_size=model.batch_size,
                  steps_per_epoch=len(train_images) / model.batch_size, 
                  epochs=model.num_epochs,
                  validation_data= validation_data,
                  callbacks=callback_list)



def test(model, test_labels, test_images):
    
    """" Test the trained model on a set of test images """

    model.evaluate(test_images, test_labels, batch_size=model.batch_size)


def main():
    """ Run the program give command-line arguments """

    checkpoint_path = "./checkpoints/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if ARGS.first_model:
        print("Using model 1!")
        model = first_Model()
    else:
        print("Using model 2!")
        model = Model()

    model(tf.keras.Input(shape=(48, 48, 1)))

    # Load weights indicated by --load-checkpoint flag
    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint)

    #if not running the live feed
    if not ARGS.live_feed:
        normalize = False
        augment = False
        if ARGS.normalize_data:
            normalize = True

        train_images, train_labels, test_images, test_labels = get_data(normalize)
        # train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

        if ARGS.augment_data:
            augment = True

            model.compile(loss='categorical_crossentropy', metrics= ['categorical_accuracy'])

            test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=7)
            test_images = normalize_test(test_images)
            test_images = np.expand_dims(test_images, axis=3)

            validation_data = (test_images, test_labels)

            train(augment, model, train_labels, train_images, validation_data, checkpoint_path)

            #results = model.evaluate(test_images, test_labels, batch_size=model.batch_size)
            test(model, test_labels, test_images)
        else:

            model.compile(loss='categorical_crossentropy', metrics= ['categorical_accuracy'])

            test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=7)
            test_images = np.expand_dims(test_images, axis=3)

            validation_data = (test_images, test_labels)
            train(augment, model, train_labels, train_images, validation_data, checkpoint_path)

            #results = model.evaluate(test_images, test_labels, batch_size=model.batch_size)
            test(model, test_labels, test_images)
            
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
        emotions = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Sad", 5: "Surprised", 6: "Neutral"}

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
                prediction = model.predict(cropped_img)[0]
                sorted_prediction = np.argsort(-prediction)
                maxindex = int(sorted_prediction[0])
                second_index = int(sorted_prediction[1])
                third_index = int(sorted_prediction[2])
                emotion1 = emotions[maxindex] + ": " + str(prediction[maxindex])
                emotion2 = emotions[second_index] + ": " + str(prediction[second_index])
                emotion3 = emotions[third_index] + ": " + str(prediction[third_index])
                cv2.putText(frame, emotion1, (x+20, y-60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, emotion2, (x+20, y-40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, emotion3, (x+20, y-20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)

            #display the results of emotion recognition
            cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        feed.release()
        cv2.destroyAllWindows()


ARGS = parse_args()

if __name__ == '__main__':
    main()