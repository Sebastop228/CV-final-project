import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import io
import itertools
import tensorflow as tf
from tensorflow import keras

def plot_to_image(figure):
    """ Converts a pyplot figure to an image tensor. """

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image


class ConfusionMatrixLogger(tf.keras.callbacks.Callback):
    """ Keras callback for logging a confusion matrix for viewing
    in Tensorboard. """

    def __init__(self, validation_data):
        super(ConfusionMatrixLogger, self).__init__()

        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        self.log_confusion_matrix(epoch, logs)

    def plot_confusion_matrix(self, cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
            cm (array, shape = [n, n]): a confusion matrix of integer classes
            class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

    def log_confusion_matrix(self, epoch, logs):
        # # Use the model to predict the values from the validation dataset.
        # test_pred_raw = self.model.predict(self.test_images)
        # test_pred = np.argmax(test_pred_raw, axis=1)

        # # Calculate the confusion matrix.
        # cm = sklearn.metrics.confusion_matrix(self.test_labels, test_pred)
        # # Log the confusion matrix as an image summary.
        # class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        # figure = self.plot_confusion_matrix(cm, class_names=class_names)
        # cm_image = plot_to_image(figure)

        # # Log the confusion matrix as an image summary.
        # file_writer_cm = tf.summary.create_file_writer('logs/confusion_matrix')
        # with file_writer_cm.as_default():
        #     tf.summary.image("Confusion Matrix", cm_image, step=epoch)

        """ Writes a confusion matrix plot to disk. """

        test_pred = []
        test_true = []
        count = 0
        for i in self.validation_data:
            test_pred.append(self.model.predict(i[0]))
            test_true.append(i[1])
            count += 1
            if count >= 1500 / self.model.batch_size:
                break

        test_pred = np.array(test_pred)
        test_pred = np.argmax(test_pred, axis=-1).flatten()
        test_true = np.array(test_true).flatten()

        # Source: https://www.tensorflow.org/tensorboard/image_summaries
        cm = sklearn.metrics.confusion_matrix(test_true, test_pred)
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        figure = self.plot_confusion_matrix(cm, class_names=class_names)
        cm_image = plot_to_image(figure)

        file_writer_cm = tf.summary.create_file_writer('logs/confusion_matrix')

        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix (on validation set)", cm_image, step=epoch)