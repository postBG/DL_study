import numpy as np
from data_prep import features, targets, features_test, targets_test

"""
- Set the weight step to zero: Δw_i = 0
- For each record in the training data:
    - Make a forward pass through the network, calculating the output y^ = f(∑w_i * x_i)
    - Calculate the error term for the output unit, δ = (y − y^) * f'(∑w_i * x_i)
    - Update the weight step Δw_i = Δw_i + δ * x_i 
- Update the weights w_i = w_i + (η * Δw_i)/m  where η is the learning rate and m is the number of records. 
Here we're averaging the weight steps to help reduce any large variations in the training data.
- Repeat for e epochs.
"""


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# TODO: We haven't provided the sigmoid_prime function like we did in
#       the previous lesson to encourage you to come up with a more
#       efficient solution. If you need a hint, check out the comments
#       in solution.py from the previous lecture.

# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# Initialize weights
weights = np.random.normal(scale=1 / n_features ** .5, size=n_features)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        # Note: We haven't included the h variable from the previous
        #       lesson. You can add it if you want, or you can calculate
        #       the h together with the output

        # TODO: Calculate the output
        h = np.dot(x, weights)
        output = sigmoid(h)

        # TODO: Calculate the error
        error = y - output

        # TODO: Calculate the error term
        sigmoid_prime = output * (1 - output)
        error_term = error * sigmoid_prime

        # TODO: Calculate the change in weights for this sample
        #       and add it to the total weight change
        del_w += learnrate * error_term * x

    # TODO: Update weights using the learning rate and the average change in weights
    weights += del_w

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
