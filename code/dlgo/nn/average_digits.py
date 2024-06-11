import numpy as np
from matplotlib import pyplot as plt
from dlgo.nn.load_mnist import load_data
from dlgo.nn.layers import sigmoid_double


def average_digit(data, digit):  # We compute the average over all samples in our data representing a given digit.
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)


train, test = load_data()
avg_eight = average_digit(train, 8)  # We use the average eight as parameters for a simple model to detect eights.


img = (np.reshape(avg_eight, (28, 28)))
plt.imshow(img)
plt.show()