"""Imports==========================================================================================================="""
# Third-party libraries
import numpy as np  # Matrix and vector calculations
import pickle as pck  # Object storing
import random  # Initialization


class NeuralNetwork(object):
    """Neural Network class."""

    def __init__(self, sizes: list, mean: int = 0, variance: int = 1) -> None:
        self.sizes = sizes
        """List with number of neurons in each layer."""

        self.n_layers = len(sizes)
        """Number of layers."""

        # Initializes biases for each neuron of the 2nd layer forward with gaussian distribution mean=0, variance=1
        self.biases = [np.random.normal(mean, variance, [y, 1]) for y in self.sizes[1:]]
        """Matrix of biases."""

        """ Initializes weights for each neuron connection
            Index for a weight from a K neuron in a layer L to a J neuron in an L+1 layer is weights[L][J][K]
            Example: weight from the 3rd neuron in the 2nd layer to the 5th in the 3rd layer is weights[2][5][3]"""
        self.weights = [np.random.normal(mean, variance, [y, x]) for x, y in zip(sizes[:-1], sizes[1:])]
        """Matrix of weights."""

        # Graphing purposes
        self.accuracy = [0.0]
        """Accuracy values for every epoch of training."""

    @staticmethod
    def sigmoid(x):
        """Return the sigmoid function value of x."""
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        """Return the sigmoid derivative function value of x."""
        sigmoid_val = 1.0 / (1.0 + np.exp(-x))
        return sigmoid_val * (1 - sigmoid_val)

    @staticmethod
    def cost_prime(output_activations, y):
        """Return the vector of partial derivative value of the quadratic cost."""
        return output_activations - y

    def predict(self, a):
        """Make a prediction on some input a."""
        # For each layer of weights "w" and biases "b" calculate activations and move forward
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self, train_data, epochs, mini_batch_size, learning_rate, test_data=None):
        """Train the Network with SGD using training data, a list of tuples with (input, output)"""
        n_train = len(train_data)

        for epoch in range(epochs):
            # Shuffle (reorganize randomly) training data
            random.shuffle(train_data)

            # Separate training data into mini batches stored in a list mini_batches
            mini_batches = [train_data[k:(k + mini_batch_size)] for k in range(0, n_train, mini_batch_size)]

            # Update weights and biases for each mini batch using SGD and backpropagation
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if test_data:   # If test_data is provided
                n_test = len(test_data)
                if epoch == 0:
                    print("Accuracy\t↓↓↓")
                print(f"Epoch {epoch + 1}:\t{self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {epoch + 1} complete")

    def step_sgd(self, mini_batches, learning_rate, test_data):
        """Apply SGD while keeping track of accuracy within each epoch (useful for graphing)."""
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, learning_rate)
        n_test = len(test_data[:100])
        self.accuracy.append(self.evaluate(test_data) / n_test)

    def update_mini_batch(self, mini_batch, learning_rate):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single mini
        batch.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(learning_rate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backpropagation(self, x, y):
        """Return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x.  nabla_b and
        nabla_w are layer-by-layer lists of numpy arrays, similar to self.biases and self.weights.
        """

        # Create empty arrays for the gradients of the cost (split into biases and weights)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Calculate the activations for the input and store them
        activation = x
        activations = [x]   # List to store all the activations, layer by layer
        zs = []             # List to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_prime(activations[-1], y) * \
            self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for layer in range(2, self.n_layers):
            z = zs[-layer]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural network outputs the correct result. Note that the
        neural network's output is assumed to be the index of whichever neuron in the final layer has the highest
        activation.
        """
        test_results = [(np.argmax(self.predict(x)), y) for (x, y) in test_data[:100]]
        return sum(int(x == y) for (x, y) in test_results)

    def save(self, filename):
        """Save Neural Networks Parameters to .pkl file"""
        with open(filename, "wb") as file:
            pck.dump(self, file, pck.HIGHEST_PROTOCOL)


def load_net(filename):
    """Return NeuralNetwork object from .pkl file path"""
    with open(filename, "rb") as file:
        net = pck.load(file)
    assert type(net.sizes) == list
    return net
