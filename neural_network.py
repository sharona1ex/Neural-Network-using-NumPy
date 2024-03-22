import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MyNN:
    def __init__(self, layer_sizes, activation_functions):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i - 1]) for i in range(1, self.num_layers)]
        self.biases = [np.random.randn(layer_sizes[i], 1) for i in range(1, self.num_layers)]
        self.activation_functions = activation_functions

    def forward_propagation(self, X):
        activations = [X]
        z_values = []
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            a = self.activation_functions[i](z)
            z_values.append(z)
            activations.append(a)
        return activations, z_values

    def backward_propagation(self, X, y, activations, z_values, learning_rate):
        num_samples = X.shape[1]
        d_weights = [np.zeros(w.shape) for w in self.weights]
        d_biases = [np.zeros(b.shape) for b in self.biases]

        # Backpropagation
        delta = (activations[-1] - y) * self.activation_functions[-1](z_values[-1], derivative=True)
        d_weights[-1] = np.dot(delta, activations[-2].T) / num_samples
        d_biases[-1] = np.sum(delta, axis=1, keepdims=True) / num_samples

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].T, delta) * self.activation_functions[-l](z_values[-l], derivative=True)
            d_weights[-l] = np.dot(delta, activations[-l - 1].T) / num_samples
            d_biases[-l] = np.sum(delta, axis=1, keepdims=True) / num_samples

        # Update weights and biases
        self.weights = [w - learning_rate * dw for w, dw in zip(self.weights, d_weights)]
        self.biases = [b - learning_rate * db for b, db in zip(self.biases, d_biases)]

    def train(self, X, y, epochs, batch_size, learning_rate):
        num_samples = X.shape[1]
        for epoch in range(epochs):
            for i in range(0, num_samples, batch_size):
                X_batch = X[:, i:i + batch_size]
                y_batch = y[:, i:i + batch_size]
                activations, z_values = self.forward_propagation(X_batch)
                self.backward_propagation(X_batch, y_batch, activations, z_values, learning_rate)
            print(f"Epoch {epoch + 1}/{epochs} completed")

    def mean_squared_error(self, y_hat, y):
        return np.mean((y_hat - y) ** 2)

    def test(self, X_test, y_test):
        # Forward propagation to get predictions
        activations, _ = self.forward_propagation(X_test)
        y_pred = activations[-1]

        # Calculate mean squared error
        mse = self.mean_squared_error(y_pred, y_test)
        print("Mean Squared Error on Test Data:", mse)

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        return activations[-1]

    def accuracy(self, y_pred, y):
        y_hat = np.where(y_pred >= 0.7, 1, 0)
        return accuracy_score(y_true=y, y_pred=y_hat)

    def pre_process(self, data, training=True):
        """
        A custom data preprocessing function for titanic dataset.
        :param data: the dataframe of titanic data set
        :return: classification labels and scaled features for training and validation
        """
        # split features and label
        X, y = data[["Pclass", "Sex", "Age", "Embarked"]], data['Survived']
        X = X.to_numpy()
        y = y.to_numpy()
        if training:
            # split training and validation data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77, shuffle=True)
            self.scaler = StandardScaler().fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled.T, y_train.reshape(1, -1), X_test_scaled.T, y_test.reshape(1, -1)
        else:
            # this sections scales the real test data for predictions
            return self.scaler.transform(X).T, y.reshape(1, -1)

    def clean(self, dataset):
        """
        This function removes outliers from the dataset provided for each column if the total
        outliers equal to or below 5% of the total data.
        :param dataset: Input dataset as numpy matrix.
        :return: Cleaned dataset as numpy matrix.
        """
        cleaned_dataset = dataset.copy()

        # Calculate the number of outliers for each column
        outliers_count = np.sum((dataset < np.percentile(dataset, 25)) | (dataset > np.percentile(dataset, 75)), axis=0)

        # Calculate the total number of outliers
        total_outliers = np.sum(outliers_count)

        # Check if the total number of outliers is equal to or below 5% of the total data
        if total_outliers <= 0.05 * dataset.size:
            # Remove outliers from columns where the total number of outliers is below the threshold
            cleaned_dataset = cleaned_dataset[~((dataset < np.percentile(dataset, 25)) | (dataset > np.percentile(dataset, 75))).any(axis=1)]

        return cleaned_dataset