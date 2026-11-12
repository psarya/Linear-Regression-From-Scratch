import numpy as np


# Hypothesis function: calculate predictions
def hypothesis(X, theta):
    return np.dot(X, theta)


# Cost function: calculate Mean Squared Error
def compute_cost(X, y, theta):

    m = len(y)

    predictions = hypothesis(X, theta)

    errors = predictions - y

    cost = np.sum(errors ** 2) / m

    return cost


# Gradient Descent: optimize model parameters
def gradient_descent(X, y, theta, learning_rate, iterations):

    m = len(y)

    cost_history = []

    for i in range(iterations):

        # Calculate predictions
        predictions = hypothesis(X, theta)

        # Calculate errors
        errors = predictions - y

        # Calculate gradients
        gradients = (2 / m) * np.dot(X.T, errors)

        # Update parameters
        theta = theta - learning_rate * gradients

        # Calculate updated cost
        cost = compute_cost(X, y, theta)

        # Store cost
        cost_history.append(cost)

        # Display training progress
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.6f}")

    return theta, cost_history