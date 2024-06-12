import numpy as np
from matplotlib import pyplot as plt
from load_mnist import load_data
from layers import sigmoid_double


def average_digit(data, digit):  # We compute the average over all samples in our data representing a given digit.
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)


train, test = load_data()
avg_eight = average_digit(train, 8) / 255  # We use the average eight as parameters for a simple model to detect eights.


img = (np.reshape(avg_eight, (28, 28)))
plt.imshow(img)
plt.show()

x_3 = train[2][0] / 255  # Training sample at index 2 is a "4".
x_18 = train[17][0] / 255  # Training sample at index 17 is an "8"

W = np.transpose(avg_eight)
print(np.dot(W, x_3))  # This evaluates to about 20.1.
print(np.dot(W, x_18))  # This term is much bigger, about 54.2.


def predict(x, W, b):  #   A simple prediction is defined by applying sigmoid to the output of np.doc(W, x) + b.
    return sigmoid_double(np.dot(W, x) + b)


b = -45  # Based on the examples computed so far we set the bias term to -45.

print(predict(x_3, W, b))  # The prediction for the example with a "4" is close to zero.
print(predict(x_18, W, b))  # 0.96


def evaluate(data, digit, threshold, W, b):  # As evaluation metric we choose accuracy, the ratio of correct predictions among all.
    total_samples = 1.0 * len(data)
    correct_predictions = 0
    for x in data:
        if predict(x[0] / 255, W, b) > threshold and np.argmax(x[1]) == digit:  # Predicting an instance of an eight as "8" is a correct prediction.
            correct_predictions += 1
        if predict(x[0] / 255, W, b) <= threshold and np.argmax(x[1]) != digit:  # If the prediction is below our threshold and the sample is not an "8", we also predicted correctly.
            correct_predictions += 1
    return correct_predictions / total_samples


print(evaluate(data=train, digit=8, threshold=0.5, W=W, b=b))  # Accuracy on training data of our simple model is 78% (0.7814)

print(evaluate(data=test, digit=8, threshold=0.5, W=W, b=b))  # Accuracy on test data is slightly lower, at 77% (0.7749)

eight_test = [x for x in test if np.argmax(x[1]) == 8]
print(evaluate(data=eight_test, digit=8, threshold=0.5, W=W, b=b))