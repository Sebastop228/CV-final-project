import tensorflow as tf 
import numpy as np
#architecture sourced from http://cs231n.stanford.edu/reports/2016/pdfs/005_Report.pdf

class Model(tf.keras.Model):
    
    """ Our second emotion recognition architecture model """

    def __init__(self):
        super(Model, self).__init__()

        tf.keras.backend.set_floatx('float32')

        #################### HYPERPARAMETERS #############
        self.batch_size = 128

        self.learning_rate = 0.001

        self.num_epochs = 100

        self.stddev = 0.1

        self.decay = 1e-7

        self.percent_training = 0.8
        #################### HYPERPARAMETERS #############

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate, beta_1 = 0.9, beta_2 = 0.999) #, decay = self.decay)
        
        #No initializer was used in the code in the above github repo
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=self.stddev)

        self.architecture = tf.keras.Sequential()

        #We tried three different architectures, the only one that was somewhat
        #successful was the one below (modified from the version in the paper)

        ################################## ARCHITECTURE BLOCK 2 #######################################################
        #From https://medium.com/themlblog/how-to-do-facial-emotion-recognition-using-a-cnn-b7bbae79cd8f
        #Github Repo: https://github.com/gitshanks/fer2013

        # Says accuracy of 66%; was able to obtain 65%


        self.architecture.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.architecture.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        self.architecture.add(tf.keras.layers.BatchNormalization())
        self.architecture.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.architecture.add(tf.keras.layers.Dropout(0.5))

        self.architecture.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        self.architecture.add(tf.keras.layers.BatchNormalization())
        self.architecture.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        self.architecture.add(tf.keras.layers.BatchNormalization())
        self.architecture.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.architecture.add(tf.keras.layers.Dropout(0.5))

        self.architecture.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
        self.architecture.add(tf.keras.layers.BatchNormalization())
        self.architecture.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
        self.architecture.add(tf.keras.layers.BatchNormalization())
        self.architecture.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.architecture.add(tf.keras.layers.Dropout(0.5))

        self.architecture.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
        self.architecture.add(tf.keras.layers.BatchNormalization())
        self.architecture.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
        self.architecture.add(tf.keras.layers.BatchNormalization())
        self.architecture.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.architecture.add(tf.keras.layers.Dropout(0.5))

        self.architecture.add(tf.keras.layers.Flatten())

        self.architecture.add(tf.keras.layers.Dense(512, activation='relu'))
        self.architecture.add(tf.keras.layers.Dropout(0.4))
        self.architecture.add(tf.keras.layers.Dense(256, activation='relu'))
        self.architecture.add(tf.keras.layers.Dropout(0.4))
        self.architecture.add(tf.keras.layers.Dense(128, activation='relu'))
        self.architecture.add(tf.keras.layers.Dropout(0.5))

        self.architecture.add(tf.keras.layers.Dense(7, activation='softmax'))


        ################################## ARCHITECTURE BLOCK 2 #######################################################

    def call(self, inputs):

        """ A method to return the architecture """

        return self.architecture(inputs)