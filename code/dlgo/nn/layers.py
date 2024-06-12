from __future__ import print_function
import numpy as np


def sigmoid_double(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid(z):
    return np.vectorize(sigmoid_double)(z)


def sigmoid_prime_double(x):
    return sigmoid_double(x) * (1 - sigmoid_double(x))


def sigmoid_prime(z):
    return np.vectorize(sigmoid_prime_double)(z)


class Layer(object):  # Layers are stacked to build a sequential neural network.
    def __init__(self):
        self.params = []

        self.previous = None  # A layer knows its predecessor ('previous')...
        self.next = None  # ... and its successor ('next').

        self.input_data = None  # Each layer can persist data flowing into and out of it in the forward pass.
        self.output_data = None

        self.input_delta = None  # Analogously, a layer holds input and output data for the backward pass.
        self.output_delta = None
    
    def connect(self, layer):  # This method connects a layer to its direct neighbours in the sequential network.
        self.previous = layer
        layer.next = self
    
    def forward(self):  # Each layer implementation has to provid a function to feed input data forward.
        raise NotImplementedError
    
    def get_forward_input(self):  # input_data is reserved for the first layer, all others get their input from the previous output.
        if self.previous is not None:
            return self.previous.output_data
        else:
            return self.input_data
    
    def backward(self):  # Layers have to implement backpropagation of error terms, that is a way to feed input errors backward through the network.
        raise NotImplementedError

    def get_backward_input(self):  # Input delta is reserved for the last layer, all other layers get their error terms from their successor.
        if self.next is not None:
            return self.next.output_delta
        else:
            return self.input_delta
    
    def clear_deltas(self):  # We compute and accumulate deltas per mini-batch, after which we need to reset these deltas.
        pass

    def update_params(self, learning_rate):  # Update layer parameters according to current deltas, using the specified learning_rate.
        pass

    def describe(self):  # Layer implementations can print their properties.
        raise NotImplementedError


class ActivationLayer(Layer):  # This activation layer uses the sigmoid function to activate neurons.
    def __init__(self, input_dim):
        super(ActivationLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim
    
    def forward(self):
        data = self.get_forward_input()
        self.output_data = sigmoid(data)  # The forward pass is simply applying the sigmoid to the input data.
    
    def backward(self):
        delta = self.get_backward_input()
        data = self.get_forward_input()
        self.output_delta = delta * sigmoid_prime(data)  # The backward pass is element-wise multiplication of the error term with the sigmoid derivative evaluated at the input to this layer.
    
    def describe(self):
        print("|-- " + self.__class__.__name__)
        print("  |-- dimensions: ({},{})".format(self.input_dim, self.output_dim))


class DenseLayer(Layer):

    def __init__(self, input_dim, output_dim):  # Dense layers have input and output dimensions.

        super(DenseLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = np.random.randn(output_dim, input_dim)  # We randomly initialize weight matrix and bias vector.
        self.bias = np.random.randn(output_dim, 1)

        self.params = [self.weight, self.bias]  # The layer parameters consist of weights and bias terms.

        self.delta_w = np.zeros(self.weight.shape)  # Deltas for weights and biases are set to zero.
        self.delta_b = np.zeros(self.bias.shape)

    def forward(self):
        data = self.get_forward_input()
        self.output_data = np.dot(self.weight, data) + self.bias  # The forward pass of the dense layer is the affine linear transformation on input data defined by weights and biases.
    
    def backward(self):
        data = self.get_forward_input()
        delta = self.get_backward_input()  # For the backward pass we first get input data and delta.

        self.delta_b = self.delta_b + delta  # The current delta is added to the bias delta.

        self.delta_w += np.dot(delta, data.transpose())  # Then we add this term to the weight delta.

        self.output_delta = np.dot(self.weight.transpose(), delta)  # The backward pass is completed by passing an output delta to the previous layer.
    
    def update_params(self, rate):  # Using weight and bias deltas we can update model parameters with gradient descent.
        self.weight -= rate * self.delta_w
        self.bias -= rate * self.delta_b

    def clear_deltas(self):  # After updating parameters we should reset all deltas.
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def describe(self):  # A dense layer can be described by its input and output dimensions.
        print("|--- " + self.__class__.__name__)
        print("  |-- dimensions: ({},{})".format(self.input_dim, self.output_dim))