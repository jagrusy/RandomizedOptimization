from mlrose import mlrose
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import time

from util import getCreditCardData

# NN Params Optimized From Supervised Learning Project
# {'alpha': 0.18373673469387755, 'hidden_layer_sizes': (10,), 'momentum': 0.9}
# 0.81375
HIDDEN_NODES = [10]

def ANN(X_train, X_test, y_train, y_test, optimization_alg):
    print('using {}'.format(optimization_alg))
    # Initialize neural network object and fit object
    nn_model = mlrose.NeuralNetwork(hidden_nodes = HIDDEN_NODES, activation = 'relu', \
                                    algorithm = optimization_alg, max_iters = 100, \
                                    bias = True, is_classifier = True, learning_rate = 0.0001, \
                                    early_stopping = True, clip_max = 5, max_attempts = 100, \
                                    random_state = 3)
    time1 = time.time()
    nn_model.fit(X_train, y_train)
    time2 = time.time()
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model.predict(X_train)
    time3 = time.time()

    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    print('training stats')
    print(time2 - time1)
    print(y_train_accuracy)

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model.predict(X_test)

    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    print('testing stats')
    print(time3 - time2)
    print(y_test_accuracy)

if __name__ == "__main__":
    np.random.seed(0)
    test_size = 0.2
    # Preprocess for mlrose
    X_train, X_test, y_train, y_test = getCreditCardData(path='./Data/ccdefault.xls', test_size=test_size)

    # Normalize feature data
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # One hot encode target values
    one_hot = OneHotEncoder()

    y_train = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    y_test = one_hot.transform(y_test.reshape(-1, 1)).todense()

    ANN(X_train, X_test, y_train, y_test, 'gradient_descent')
    ANN(X_train, X_test, y_train, y_test, 'random_hill_climb')
    ANN(X_train, X_test, y_train, y_test, 'simulated_annealing')
    ANN(X_train, X_test, y_train, y_test, 'genetic_alg')

# Benchmark with Gradient Descent


# Check performance with RHC, SA, and Genetic

