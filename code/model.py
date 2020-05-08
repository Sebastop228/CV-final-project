import tensorflow as tf 
import numpy as np


#Found the paper github repo: https://github.com/akashsara/emotion-recognition

class first_Model(tf.keras.Model):
    
    """ The first version of our architecture model """

    def __init__(self):
        super(first_Model, self).__init__()

        tf.keras.backend.set_floatx('float32')

        #################### HYPERPARAMETERS #############

        self.batch_size = 128

        self.learning_rate = 0.0001

        self.num_epochs = 24

        self.stddev = 0.1

        self.decay = 10e-6

        self.percent_training = 0.8

        #################### HYPERPARAMETERS #############

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate, decay = self.decay)

        self.architecture = []

        self.architecture.append(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = (1,1), activation = "relu", padding = "same"))
        self.architecture.append(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = (1,1), activation = "relu"))
        
        self.architecture.append(tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2)))
        self.architecture.append(tf.keras.layers.Dropout(rate = 0.25))
        
    
        self.architecture.append(tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, strides = (1,1), activation = "relu", padding = "same"))
        self.architecture.append(tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, strides = (1,1), activation = "relu"))
       
        self.architecture.append(tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, strides = (1,1), activation = "relu", padding = "same"))
        self.architecture.append(tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, strides = (1,1), activation = "relu"))

        self.architecture.append(tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2))) 
        self.architecture.append(tf.keras.layers.Dropout(rate = 0.3))

        self.architecture.append(tf.keras.layers.Flatten())
        
        #Fully connected layer
        self.architecture.append(tf.keras.layers.Dense(1024, activation  = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.001)))
        self.architecture.append(tf.keras.layers.Dropout(rate = 0.25))

        #Final fully connected layer
        self.architecture.append(tf.keras.layers.Dense(7, activation = "softmax"))


    def call(self, inputs):

        """ output the result of calling the model on the inputs """

        for layer in self.architecture:
            inputs = layer(inputs)

        return inputs



    def loss_fn(self, labels, predictions):

        """ Compute the loss for this model """

        return tf.keras.losses.categorical_crossentropy(labels, predictions, from_logits = False)


    def accuracy_fn(self, labels, probs):

        """Compute the accuracy for this model """
        
        highest_prediction_index = np.argmax(probs, axis = 1)
        highest_label_index = np.argmax(labels, axis=1)
        amt_correct = np.count_nonzero(highest_label_index == highest_prediction_index)
        
        return amt_correct