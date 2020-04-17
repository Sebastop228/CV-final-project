import tensorflow as tf 
import numpy as np
import hyperparameters as hp

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        #DEFINE HYPERPARAMETERS
        #Seperate hyperparameters.py file? Or define here?

        self.optimizer = tf.keras.optimizers.Adam()

        # use tf.Sequential?? Might have better runtime
        self.architecture = [


            #DEFINE ARCHITECTURE


        ]





    def call(self, inputs):

        for layer in self.architecture:
            inputs = layer(inputs)

        return inputs



    def loss_fn(labels, predictions):
        # Which loss function to use?
        # Comment out
        return 0