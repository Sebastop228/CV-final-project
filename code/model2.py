import tensorflow as tf 
import numpy as np
import hyperparameters2 as hp2
# from tf.keras.layers import Dense, Dropout, Flatten, MaxPool2D, BatchNormalization, Activation

#architecture sourced from http://cs231n.stanford.edu/reports/2016/pdfs/005_Report.pdf

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        # Do we want to use float32 or float64??
        tf.keras.backend.set_floatx('float32')


        self.optimizer = tf.keras.optimizers.Adam(learning_rate = hp2.learning_rate, beta_1 = 0.9, beta_2 = 0.999) #, decay = hp2.decay)
        
        #They don't use a kernel initializer in their github code, but we could play around with that.
        #Also for some reason some of the conv layers have same padding and some have valid?
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=hp2.stddev)

        self.architecture = tf.keras.Sequential()
        
        
        ################################## ARCHITECTURE BLOCK 1 #######################################################
        #Could not get to work due to only returning one-hots. Paper says 65% accuracy
        
        # self.architecture = [

        #     # "https://github.com/jrishabh96/Facial-Expression-Recognition/blob/master/cnn_major.py""


        #     tf.keras.layers.Conv2D(64, (3,3), padding = 'same'),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Activation('relu'),
        #     tf.keras.layers.MaxPool2D(pool_size = (2,2)),
        #     tf.keras.layers.Dropout(0.25),

        #     tf.keras.layers.Conv2D(128, (5,5), padding = 'same'),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Activation('relu'),
        #     tf.keras.layers.MaxPool2D(pool_size = (2,2)),
        #     tf.keras.layers.Dropout(0.25),

        #     tf.keras.layers.Conv2D(512, (3,3), padding = 'same'),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Activation('relu'),
        #     tf.keras.layers.MaxPool2D(pool_size = (2,2)),
        #     tf.keras.layers.Dropout(0.25),

        #     tf.keras.layers.Conv2D(512, (3,3), padding = 'same'),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Activation('relu'),
        #     tf.keras.layers.MaxPool2D(pool_size = (2,2)),
        #     tf.keras.layers.Dropout(0.25),

        #     tf.keras.layers.Flatten(),

        #     tf.keras.layers.Dense(512, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.001)),
        #     tf.keras.layers.BatchNormalization(),
        #     # tf.keras.layers.Activation('relu'),
        #     tf.keras.layers.Dropout(0.25),

        #     tf.keras.layers.Dense(7, activation = 'sigmoid') #potentially try softmax?

        # ]
        ################################## ARCHITECTURE BLOCK 1 #######################################################


        ################################## ARCHITECTURE BLOCK 2 #######################################################
        #From https://medium.com/themlblog/how-to-do-facial-emotion-recognition-using-a-cnn-b7bbae79cd8f
        #Github Repo: https://github.com/gitshanks/fer2013

        # Says accuracy of 66%; was able to obtain 62%

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


        ################################## ARCHITECTURE BLOCK 3 #######################################################
        # From http://cs231n.stanford.edu/reports/2016/pdfs/005_Report.pdf
        # Possibly interesting resource? https://arxiv.org/pdf/1612.02903.pdf
    
        # self.architecture.append(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = (1,1), activation = "relu"))
        # #figure out parameters for batch normalization (I think this is what they mean by
        # # batch normalization)
        # self.architecture.append(tf.keras.layers.BatchNormalization())
        # #figure out dropout rate
        # self.architecture.append(tf.keras.layers.Dropout(rate = 0.25))
        # self.architecture.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        
        # self.architecture.append(tf.keras.layers.Conv2D(filters = 128, kernel_size = 5, strides = (1,1), activation = "relu"))
        # #figure out parameters for batch normalization
        # self.architecture.append(tf.keras.layers.BatchNormalization())
        # #figure out dropout rate
        # self.architecture.append(tf.keras.layers.Dropout(rate = 0.25))
        # self.architecture.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        # self.architecture.append(tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, strides = (1,1), activation = "relu"))
        # #figure out parameters for batch normalization
        # self.architecture.append(tf.keras.layers.BatchNormalization())
        # #figure out dropout rate
        # self.architecture.append(tf.keras.layers.Dropout(rate = 0.25))
        # self.architecture.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        # self.architecture.append(tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, strides = (1,1), activation = "relu"))
        # #figure out parameters for batch normalization
        # self.architecture.append(tf.keras.layers.BatchNormalization())
        # #figure out dropout rate
        # self.architecture.append(tf.keras.layers.Dropout(rate = 0.25))
        # self.architecture.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        # self.architecture.append(tf.keras.layers.Flatten())

        # self.architecture.append(tf.keras.layers.Dense(256, activation='relu'))
        # #figure out parameters for batch normalization
        # self.architecture.append(tf.keras.layers.BatchNormalization())
        # #figure out dropout rate
        # self.architecture.append(tf.keras.layers.Dropout(rate = 0.25))

        # self.architecture.append(tf.keras.layers.Flatten())

        # self.architecture.append(tf.keras.layers.Dense(512, activation='relu'))
        # #figure out parameters for batch normalization
        # self.architecture.append(tf.keras.layers.BatchNormalization())
        # #figure out dropout rate
        # self.architecture.append(tf.keras.layers.Dropout(rate = 0.25))

        # self.architecture.append(tf.keras.layers.Flatten())

        # self.architecture.append(tf.keras.layers.Dense(7, activation = "softmax"))

        ################################## ARCHITECTURE BLOCK 3 #######################################################

    def call(self, inputs):
        #Currently using Sequential(); switch to layer if necessary

        # for layer in self.architecture:
        #     inputs = layer(inputs)
        # return inputs

        return self.architecture(inputs)



    def loss_fn(self, labels, predictions):
        #Current preprocessing makes it so labels do not need to be made into a onehot; change if necessary

        # one_hot = tf.keras.utils.to_categorical(labels)
        return tf.keras.losses.categorical_crossentropy(labels, predictions, from_logits = False) #Binary crossentropy referenced by some papers


    def accuracy_fn(self, labels, probs):
        #Current preprocessing makes it so labels also needs to be "argmaxed"; change if necessary

        highest_prediction_index = np.argmax(probs, axis = 1)
        highest_label_index = np.argmax(labels, axis=1)
        amt_correct = np.count_nonzero(highest_label_index == highest_prediction_index)
        
        return amt_correct