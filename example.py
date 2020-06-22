import tensorflow as tf
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorboard.plugins.hparams import api as hp
from random import choice, shuffle
from pprint import pprint
from datetime import datetime
import io
import itertools

from six.moves import range
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


class CMCallback(tf.keras.callbacks.Callback):
    def __init__(self, fw, model, test_data, test_labels):
        self.fw = fw
        self.model = model
        self.test_data = test_data
        self.test_labels = np.argmax(test_labels, axis=1)

    def on_epoch_end(self, epoch, logs=None):
        # Use the model to predict the values from the validation dataset.
        test_pred = self.model.predict_classes(self.test_data)
        # test_pred = np.argmax(test_pred_raw, axis=1)

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(self.test_labels, test_pred)
        # Log the confusion matrix as an image summary.
        figure   = plot_confusion_matrix(cm, class_names=[str(i) for i in range(1, 11)])
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with self.fw.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)


def get_model(input_shape, num_classes=10, hu_1=32, hu_2=64, hu_3=128, dp=0.25):
    """Get the neural network model

    Args:
        num_classes : Number of output classes
        hu_1        : Number of neurons of hidden layer 1
        hu_2        : Number of neurons of hidden layer 2
        hu_3        : Number of neurons of hidden layer 3
        dp          : Dropout rate

    Return:
        model : Built model ready to fit
    """
    model = Sequential()
    model.add(
        Conv2D(hu_1, kernel_size=(3, 3), activation="relu", input_shape=input_shape)
    )
    model.add(Conv2D(hu_2, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(hu_3, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    return model


def rand_params(n_iter, *params):
    """Combine params

    Combine params and gerenates a list of HParams

    Args:
        n_iter  : number of randomizations
        *params : Params list

    Return:
        hp_list : List of HParams dict
    """

    def choice_param(params, hp_list, n_tries=0, max_tries=5):
        hparams = {}
        for param in params:
            # Randomize the choice of each param
            param_values = param.domain.values
            shuffle(param_values)
            hparams[param.name] = choice(param_values)

        if n_tries < max_tries:
            if hparams in hp_list:
                choice_param(params, hp_list, n_tries=n_tries + 1)
            else:
                return hparams
        else:
            return None

    hp_list = list()
    for i in range(n_iter):
        hparams = choice_param(params, hp_list)
        if hparams is not None:
            hp_list.append(hparams)

    return hp_list


def fit_model(log_dir, n_randomize=5, epochs=10, num_classes=10):
    """Fits model to the MNIST dataset

    This function calls a randomizes of hyperparameters and for each combination
    fits a model to the MNIST dataset. Afther fiting the model, this function logs
    its results for further analysis in Tensorboard

    Args:
        log_dir     : Logs master directory
        n_randomize : Number of randomizations to find the best HParams
    """
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # save class labels to disk to color data points in TensorBoard accordingly
    with open(os.path.join(log_dir, "metadata.tsv"), "w") as f:
        np.savetxt(f, np.abs(y_test))

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # HParams
    HP_HU_1            = hp.HParam("hu_1", hp.Discrete([32, 64]))
    HP_HU_2            = hp.HParam("hu_2", hp.Discrete([64, 128]))
    HP_HU_3            = hp.HParam("hu_3", hp.Discrete([128, 256]))
    HP_BATCH           = hp.HParam("batch", hp.Discrete([100, 150]))
    HP_DROPOUT_PERCENT = hp.HParam("dp", hp.Discrete([25, 50]))
    METRIC_ACCURACY    = "accuracy"

    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams_config(
            hparams=[HP_HU_1, HP_HU_2, HP_HU_3, HP_BATCH, HP_DROPOUT_PERCENT],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name="Accuracy")],
        )
    # Tensorboard callback
    tf_board = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq="epoch"
    )

    # Try each combination and log the results
    hp_list = rand_params(
        n_randomize, HP_HU_1, HP_HU_2, HP_HU_3, HP_BATCH, HP_DROPOUT_PERCENT
    )

    for i, hparams in enumerate(hp_list):
        run_dir = log_dir + f"/run_{i}"
        print("--- Starting trial: %s" % i)
        pprint(f"Running with params {hparams}")

        # Tensorboard callback
        tf_board = tf.keras.callbacks.TensorBoard(
            log_dir=run_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
            embeddings_freq=1,
        )

        model = get_model(
            input_shape,
            hu_1=hparams["hu_1"],
            hu_2=hparams["hu_2"],
            hu_3=hparams["hu_3"],
            dp=hparams["dp"] / 100,
        )

        # Confusion matrix callback
        file_writer_cm = tf.summary.create_file_writer(run_dir + "/cm")
        tf_cm_callback = CMCallback(file_writer_cm, model, x_test, y_test)

        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adadelta(),
            metrics=["accuracy"],
        )

        model.fit(
            x_train,
            y_train,
            batch_size=hparams["batch"],
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=[tf_board, tf_cm_callback],
        )
        score = model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        with tf.summary.create_file_writer(run_dir).as_default():
            # record the values used in this trial
            hp.hparams(hparams, trial_id=str(i))
            accuracy = score[1]
            tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

            # Record some images from the training set
            images = np.reshape(x_train[0:10], (-1, 28, 28, 1))
            tf.summary.image(
                "25 training data examples", images, max_outputs=10, step=0
            )


def call_tensorboard(file):
    query = input("Do you wish to run tensorboard (y/n) ? ")
    Fl = query[0].lower()
    if query == "" or not Fl in ["y", "n"]:
        print("Please answer with yes or no!")
        exit()
    if Fl == "y":
        os.system("tensorboard --logdir=" + file)
        exit()
    else:
        exit()


def main():
    # If there is a GPU limit its memory growth
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Path to Tensorboard Logs
    base_path = os.getcwd()
    logs_path = base_path + "/logs"
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    fit_model(logs_path, n_randomize=1, epochs=5)
    call_tensorboard(logs_path)


if __name__ == "__main__":
    main()
