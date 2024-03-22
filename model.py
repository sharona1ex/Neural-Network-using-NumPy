from activations_functions import sigmoid, relu, tanh
from neural_network import MyNN
import pandas as pd
import pickle
from datetime import datetime


def createNN(layer_sizes, activations, train, model=None):
    """
    A function to create neural net from the dataset's training part and report accuracy.
    :param layer_sizes: layer sizes
    :param activations: activations for each layer
    :param train: if True then a fresh model is trained else OLD model will be used to report accuracy
    :param model: name of old model.
    :return: MyNN object
    """
    # create NN
    nn = MyNN(layer_sizes, activations)

    # load titanic data
    dataset = pd.read_csv("data/train.csv")

    # clean the dataset from any outliers
    dataset = nn.clean(dataset)

    # pre-process data
    X_train, y_train, X_validation, y_validation = nn.pre_process(dataset)

    # # Train the neural network
    if train:
        # train model
        nn.train(X_train, y_train, epochs=1000, batch_size=2, learning_rate=0.01)
        # save the model
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        with open('nn_{}.pkl'.format(timestamp_str), 'wb') as file:
            pickle.dump(nn, file)
    else:
        # load model
        with open(model, 'rb') as file:
            model = pickle.load(file)
            if nn.layer_sizes == model.layer_sizes:
                nn.weights = model.weights
                nn.biases = model.biases
            else:
                raise Exception("Incompatible model loaded. (Train a fresh model, set TRAIN=True in model.py)")

    print("Training accuracy:")
    predictions = nn.predict(X_train)
    print(str(nn.accuracy(predictions[0], y_train[0]) * 100) + " %")

    print("Validation accuracy:")
    predictions_val = nn.predict(X_validation)
    print(str(nn.accuracy(predictions_val[0], y_validation[0]) * 100) + " %")

    print("Testing accuracy:")
    test_dataset = pd.read_csv("data/test.csv")
    X_test, y_test = nn.pre_process(test_dataset, training=False)
    predictions_test = nn.predict(X_test)
    print(str(nn.accuracy(predictions_test[0], y_test[0]) * 100) + " %")

    return nn


if __name__ == '__main__':
    # CONFIGURATIONS
    # # if this is false then trained model will be used else a model will be trained afresh.
    TRAIN = False
    MODEL = 'trained_models/nn_2024-03-21_22-21-23.pkl'
    # # Neural Network configuration (2 hidden layer with 2 neurons each)
    LAYER_SIZES = [4, 2, 2, 1]
    # # Activation functions to be used after each layer
    ACTIVATION_FUNCTIONS = [sigmoid, sigmoid, sigmoid]

    # MODELLING
    nn = createNN(layer_sizes=LAYER_SIZES,
                  activations=ACTIVATION_FUNCTIONS,
                  train=TRAIN,
                  model=MODEL)