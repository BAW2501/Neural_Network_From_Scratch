import numpy as np
import matplotlib.pyplot as plt


def sigmoid(Z):
    return 1/(1+np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)


def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def init_layers(nn_architecture, seed=99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1

        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values[f'W{str(layer_idx)}'] = (
            np.random.randn(layer_output_size, layer_input_size) * 0.1
        )
        params_values[f'b{str(layer_idx)}'] = (
            np.random.randn(layer_output_size, 1) * 0.1
        )

    return params_values


def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid
    else:
        raise NotImplemented('Non-implemented activation function')

    return activation_func(Z_curr), Z_curr


def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr

        activ_function_curr = layer["activation"]
        W_curr = params_values[f"W{str(layer_idx)}"]
        b_curr = params_values[f"b{str(layer_idx)}"]
        A_curr, Z_curr = single_layer_forward_propagation(
            A_prev, W_curr, b_curr, activ_function_curr)

        memory[f"A{str(idx)}"] = A_prev
        memory[f"Z{str(layer_idx)}"] = Z_curr

    return A_curr, memory


def get_cost_value(Y_hat, Y, eps=0.001):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat + eps).T) +
                     np.dot(1 - Y, np.log(1 - Y_hat + eps).T))
    return np.squeeze(cost)


def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1]

    if activation == "relu":
        backward_activation_func = relu_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise NotImplemented('Non-Implemented activation function')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture, eps=0.000000000001):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)

    dA_prev = - (np.divide(Y, Y_hat + eps) - np.divide(1 - Y, 1 - Y_hat + eps))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]

        dA_curr = dA_prev

        A_prev = memory[f"A{str(layer_idx_prev)}"]
        Z_curr = memory[f"Z{str(layer_idx_curr)}"]

        W_curr = params_values[f"W{str(layer_idx_curr)}"]
        b_curr = params_values[f"b{str(layer_idx_curr)}"]

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

        grads_values[f"dW{str(layer_idx_curr)}"] = dW_curr
        grads_values[f"db{str(layer_idx_curr)}"] = db_curr

    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate):

    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values[f"W{str(layer_idx)}"] -= (
            learning_rate * grads_values[f"dW{str(layer_idx)}"]
        )
        params_values[f"b{str(layer_idx)}"] -= (
            learning_rate * grads_values[f"db{str(layer_idx)}"]
        )

    return params_values


def train(X, Y, nn_architecture, epochs, learning_rate, verbose=False, callback=None):
    params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        Y_hat, cashe = full_forward_propagation(
            X, params_values, nn_architecture)

        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)

        grads_values = full_backward_propagation(
            Y_hat, Y, cashe, params_values, nn_architecture)
        params_values = update(params_values, grads_values,
                               nn_architecture, learning_rate)

        if (i % 50 == 0):
            if (verbose):
                print(
                    "Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
            if (callback is not None):
                callback(i, params_values)

    return params_values, cost_history, accuracy_history


def load_data():
    # create dataset of 1000 samples where each sample is a vector of 2 values
    X = np.random.rand(2, 1000) - 0.5
    # if the sum of the values in the vector is greater than 1, then the label is 1, otherwise 0
    Y = (np.sum(X, axis=0, keepdims=True)  > 0).astype(np.uint8)
    return X, Y


if __name__ == '__main__':
    X, Y = load_data()
    nn_architecture = [
        {"input_dim": 2, "output_dim": 4, "activation": "relu"},
        {"input_dim": 4, "output_dim": 6, "activation": "relu"},
        {"input_dim": 6, "output_dim": 6, "activation": "relu"},
        {"input_dim": 6, "output_dim": 4, "activation": "relu"},
        {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
    ]
    params_values, cost_history, accuracy_history = train(X, Y, nn_architecture, 1000, 0.1)
    Y_hat, _ = full_forward_propagation(X, params_values, nn_architecture)
    Y_hat = convert_prob_into_class(Y_hat).astype(int)
    plt.hist(Y_hat)
    plt.show()
    plt.hist(Y)
    plt.show()
    # print(cost_history)
    # print(accuracy_history)
    plt.title("Cost")
    plt.plot(cost_history)
    plt.show()
    plt.title("Accuracy")
    plt.plot(accuracy_history)
    plt.show()
