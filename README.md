# Neural Network using NumPy

## Objective
To build a neural network class from scratch using NumPy and use it for Titanic dataset from Kaggle.

## How to run this code?
There are three python files created for this project:
1. `activations_functions.py`: This file has three activation functions and their derivatives.
2. `neural_network.py`: This file has a class called MyNN (short for My Neural Network) that allows user to initialize a NN, train it, test it and predict on test data.
3. `model.py`: This file actually imports the above two files, loads Titanic dataset, creates neural network, trains its, validates its and runs the model on test data to report test accuracy.

To run it on your machine:
1. Clone this repo: `git clone https://github.com/StarRider/Neural-Network-using-NumPy.git`
2. Install libraries from `requirements.txt`
3. Go to the directory `Neural-Network-using-NumPy`
4. Run `model.py` using Python 3.10

Following these steps will help you to run a pretrained model. To train a fresh model open `model.py` and set `TRAIN=True`. Also feel free to change other configurations. Your trained model will be stored in the `trained_models` folder for your future use.

## How configure this model?
The configurations are placed in `model.py`. The following configurations will allow you to try this model for variety of cases.
1. TRAIN: If this is True then your model.py will train a fresh model, if it is False, it will use a pretrained model and will output the testing accuracy along with training and validation accuracy.
2. MODEL: This is where you can specify which pretrained model to use if TRAIN is False. It is the path to a pretrained model.
3. LAYER_SIZES: This is the parameter that allows you to define the number of neurons in each layer and the number of layers. For eg, [4, 2, 1] means, there is an input layer with 4 neurons, a hidden layer with 2 neurons and an output layer with 1 neuron.
4. ACTIVATION_FUNCTIONS: This is list activation functions to be used after each hidden layer and output layer. 

## Dataset
The dataset is obtained from Kaggle and the dataset has also been placed here in the `data` folder.
1. `data/train.csv` : This is the training data, that is furhter split into 80% percent of training and 20% of validation data (This split is done within `neural_network.py` in pre_process function).
2. `data/test.csv`  : This is the test data, which the model touches the first time after training and validation is done.


