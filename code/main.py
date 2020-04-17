import tensorflow as tf 
import numpy as np
import hyperparameters as hp


def train(model, train_labels, train_images):

    amt_to_train = train_images.shape[0] #Amt of images we're training over
    batch_size = hp.batch_size # make batch size a hyperparameter, either in model or hyperparameters.py

    for i in range(0, amt_to_train, batch_size):
        # indexed into the arrays so that, for the last batch, i+batch_size doesn't go over the size of the array
        batch_images = train_images[i:min(i+batch_size, amt_to_train)]
        batch_labels = train_labels[i:min(i+batch_size, amt_to_train)]
        with tf.GradientTape() as tape:
            logits = model.call(batch_images)
            loss = model.loss(batch_labels, logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))



def test(model, test_labels, test_images):
    amt_to_test = test_images.shape[0]
    batch_size = hp.batch_size
    for i in range(0, amt_to_test, batch_size):
        batch_images = test_images[i:min(i+batch_size, amt_to_test)]
        batch_labels = test_labels[i:min(i+batch_size, amt_to_test)]
        logits = model.call(batch_images)

        #figure out how we want to calculate accuracy - accuracy function in model?? Would pass it labels and logits

        accuracy = 0
    return accuracy


def main():
    # Find how to get files
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    model = Model()

    for i in range(hp.num_epochs):
        train(model, train_labels, train_images)

        accuracy = test(model, test_labels, test_images)

        print("Epoch ", i, " accuracy is ", accuracy)

if __name__ == '__main__':
    main()