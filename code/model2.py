import tensorflow as tf 
import numpy as np
import hyperparameters2 as hp2

#architecture sourced from http://cs231n.stanford.edu/reports/2016/pdfs/005_Report.pdf

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        # Do we want to use float32 or float64??
        tf.keras.backend.set_floatx('float32')


        self.optimizer = tf.keras.optimizers.Adam(learning_rate = hp.learning_rate, decay = hp.decay)
        
        #They don't use a kernel initializer in their github code, but we could play around with that.
        #Also for some reason some of the conv layers have same padding and some have valid?
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=hp.stddev)

        #self.architecture = tf.keras.Sequential()
        self.architecture = []
    
        self.architecture.append(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = (1,1), activation = "relu"))
        #figure out parameters for batch normalization (I think this is what they mean by
        # batch normalization)
        self.architecture.append(tf.keras.layers.BatchNormalization())
        #figure out dropout rate
        self.architecture.append(tf.keras.layers.Dropout(rate = 0.25))
        self.architecture.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        
        self.architecture.append(tf.keras.layers.Conv2D(filters = 128, kernel_size = 5, strides = (1,1), activation = "relu"))
        #figure out parameters for batch normalization
        self.architecture.append(tf.keras.layers.BatchNormalization())
        #figure out dropout rate
        self.architecture.append(tf.keras.layers.Dropout(rate = 0.25))
        self.architecture.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        self.architecture.append(tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, strides = (1,1), activation = "relu"))
        #figure out parameters for batch normalization
        self.architecture.append(tf.keras.layers.BatchNormalization())
        #figure out dropout rate
        self.architecture.append(tf.keras.layers.Dropout(rate = 0.25))
        self.architecture.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        self.architecture.append(tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, strides = (1,1), activation = "relu"))
        #figure out parameters for batch normalization
        self.architecture.append(tf.keras.layers.BatchNormalization())
        #figure out dropout rate
        self.architecture.append(tf.keras.layers.Dropout(rate = 0.25))
        self.architecture.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        self.architecture.append(tf.keras.layers.Flatten())

        self.architecture.append(tf.keras.layers.Dense(256, activation='relu'))
        #figure out parameters for batch normalization
        self.architecture.append(tf.keras.layers.BatchNormalization())
        #figure out dropout rate
        self.architecture.append(tf.keras.layers.Dropout(rate = 0.25))

        self.architecture.append(tf.keras.layers.Flatten())

        self.architecture.append(tf.keras.layers.Dense(512, activation='relu'))
        #figure out parameters for batch normalization
        self.architecture.append(tf.keras.layers.BatchNormalization())
        #figure out dropout rate
        self.architecture.append(tf.keras.layers.Dropout(rate = 0.25))

        self.architecture.append(tf.keras.layers.Flatten())

        self.architecture.append(tf.keras.layers.Dense(7, activation = "softmax"))


    def call(self, inputs):
        
        for layer in self.architecture:
            inputs = layer(inputs)

        return inputs



    def loss_fn(self, labels, predictions):
        # Which loss function to use? Paper uses Categorical crossentropy (from their github repo)

        # Added one-hot encoding in order to compute loss
        one_hot = tf.keras.utils.to_categorical(labels)
        return tf.keras.losses.categorical_crossentropy(one_hot, predictions, from_logits = False)


    def accuracy_fn(self, labels, probs):
        
        #So if if I'm not mistaken, "probs" should be tensor of shape [7, batch_size], where each element [i,j] is the probability that
        #image j portrays emotion i (the dimensions could be flipped, Im not too sure. Basically all we would have to do is count
        #the number of "correct" predictions; that is, the amount of images (of which we have batch_size) for which the highest probability
        #is the correct emotion (which we would get from looking at the corresponding label)
        
        print("LABEL IS ", labels)
        print("LABEL SHAPE IS ", labels.shape)
        highest_prediction_index = np.argmax(probs, axis = 1)
        amt_correct = np.count_nonzero(labels == highest_prediction_index)
        print("AMT CORRECT IS ", amt_correct)
        #exit(0)
        return amt_correct