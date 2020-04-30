import tensorflow as tf 
import numpy as np
import hyperparameters as hp


#Found the paper github repo: https://github.com/akashsara/emotion-recognition

class first_Model(tf.keras.Model):
    def __init__(self):
        super(first_Model, self).__init__()

        # Do we want to use float32 or float64??
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
        
        #They don't use a kernel initializer in their github code, but we could play around with that.
        #Also for some reason some of the conv layers have same padding and some have valid?
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=self.stddev)

        #self.architecture = tf.keras.Sequential()
        self.architecture = []
    
        #padding and kernel initializer? Stddev for kernel initializer?
        self.architecture.append(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = (1,1), activation = "relu", padding = "same"))

        self.architecture.append(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = (1,1), activation = "relu"))
        self.architecture.append(tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2)))
        self.architecture.append(tf.keras.layers.Dropout(rate = 0.25))
        #padding
        self.architecture.append(tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, strides = (1,1), activation = "relu", padding = "same"))
        self.architecture.append(tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, strides = (1,1), activation = "relu"))
        #padding
        self.architecture.append(tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, strides = (1,1), activation = "relu", padding = "same"))
        self.architecture.append(tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, strides = (1,1), activation = "relu"))

        self.architecture.append(tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2))) 
        self.architecture.append(tf.keras.layers.Dropout(rate = 0.3))

        self.architecture.append(tf.keras.layers.Flatten())
        #The paper mentions using an "l2 regularizer with penalty 0.001" for this next Dense layer. The docs for Dense layers has 3 types
        #of regularizers, so I wasn't sure with one, but after some further research and input from my brother we think it's
        #the kernel regularizer (confirmed by paper's github repo)
        self.architecture.append(tf.keras.layers.Dense(1024, activation  = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.001)))
        self.architecture.append(tf.keras.layers.Dropout(rate = 0.25))

        #Also not sure about output shape. In the paper they make it be 7, Cause that's the amount of emotions they're working with.
        #We might want to change that if we work with a different number of them
        self.architecture.append(tf.keras.layers.Dense(7, activation = "softmax"))







    def call(self, inputs):
        for layer in self.architecture:
            inputs = layer(inputs)

        return inputs



    def loss_fn(self, labels, predictions):
        # Which loss function to use? Paper uses Categorical crossentropy (from their github repo)

        # Added one-hot encoding in order to compute loss
        # one_hot = tf.keras.utils.to_categorical(labels)
        return tf.keras.losses.categorical_crossentropy(labels, predictions, from_logits = False)


    def accuracy_fn(self, labels, probs):
        
        
        highest_prediction_index = np.argmax(probs, axis = 1)
        highest_label_index = np.argmax(labels, axis=1)
        amt_correct = np.count_nonzero(highest_label_index == highest_prediction_index)
        
        return amt_correct