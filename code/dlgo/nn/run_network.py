import load_mnist
import network
from layers import DenseLayer, ActivationLayer

training_data, test_data = load_mnist.load_data()  # First, load training and test data.

net = network.SequentialNetwork()  # Next, initialize a sequential neural network.

net.add(DenseLayer(784, 392))  # You can then add dense and activation layers one by one.
net.add(ActivationLayer(392))
net.add(DenseLayer(392, 196))
net.add(ActivationLayer(196))
net.add(DenseLayer(196, 10))
net.add(ActivationLayer(10))  # The final layer has size 10, the number of classes to predict.

# You can now easily train the model by specifying train and test data, the number of epochs, the mini-batch size and the learning rate.
net.train(training_data, epochs=10, mini_batch_size=10, learning_rate=3.0, test_data=test_data)  