import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from linear_regression import hypothesis, compute_cost, gradient_descent

# Step 1: Download the California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Step 2: Convert the dataset into a Pandas DataFrame
df = housing.frame

# Step 3: Save the dataset as a CSV file
df.to_csv("data/california_housing.csv", index=False)

# Step 4: Display the first five rows
print("First five rows:")
print(df.head())

# Step 5: Display dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Step 6: Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Step 7: Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset downloaded and saved successfully!")


#create the hypothesis function

def hypothesis(X, theta):
    return np.dot(X, theta)



#implement the cost function
def compute_cost(X, y, theta):

    m = len(y)

    predictions = hypothesis(X, theta)

    errors = predictions - y

    cost = np.sum(errors ** 2) / m

    return cost

    return theta, cost_history



#test the functions with a small example
X = np.array([
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5]
], dtype=float)

y = np.array([3, 5, 7, 9, 11], dtype=float)

# Initialize theta
theta = np.zeros(X.shape[1])

# Train model
theta, cost_history = gradient_descent(
    X,
    y,
    theta,
    learning_rate=0.01,
    iterations=5000
)

print("Learned parameters:", theta)

print("Final training MSE:", cost_history[-1])

print("Predictions:", hypothesis(X, theta))


#plot the cost function

plt.figure(figsize=(8, 5))

plt.plot(cost_history)

plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Gradient Descent Cost Convergence")

plt.grid(True)

plt.show()


#load the dataset
housing = fetch_california_housing(as_frame=True)

df = housing.frame

print("\nCalifornia Housing Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


X = df[["MedInc"]].values
y = df["MedHouseVal"].values

print("\nInput Shape:", X.shape)
print("Output Shape:", y.shape)


# split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


#standardize the input features
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

std[std == 0] = 1

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

print("\nStandardized Training Data:")
print(X_train_scaled[:5])


#prepare the data for our model
X_train_final = np.c_[
    np.ones(X_train_scaled.shape[0]),
    X_train_scaled
]

X_test_final = np.c_[
    np.ones(X_test_scaled.shape[0]),
    X_test_scaled
]

print("\nFinal Training Shape:", X_train_final.shape)
print("Final Testing Shape:", X_test_final.shape)



#train using gradient descent
theta = np.zeros(X_train_final.shape[1])

learning_rate = 0.01
iterations = 2000

theta, cost_history = gradient_descent(
    X_train_final,
    y_train,
    theta,
    learning_rate,
    iterations
)

print("\nLearned Parameters:")
print(theta)

print("\nFinal Training Cost:")
print(cost_history[-1])


#plot the training cost
plt.figure(figsize=(8, 5))

plt.plot(cost_history)

plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Gradient Descent Cost Convergence")

plt.grid(True)

plt.savefig("results/cost_convergence.png", dpi=300)

plt.show()



#load and explore the housing dataset
import matplotlib.pyplot as plt

df = pd.read_csv("data/california_housing.csv")
print(df.head())
print(df.shape)
print(df.isnull().sum())
df = df.dropna()
print(df.columns.tolist())


# Select features and target

X = df.drop(columns=["MedHouseVal"]).values
y = df["MedHouseVal"].values

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Step 3: Split the dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training targets:", y_train.shape)
print("Testing targets:", y_test.shape)


# Calculate mean and standard deviation from training data
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

# Prevent division by zero
std[std == 0] = 1

# Standardize training data
X_train_scaled = (X_train - mean) / std

# Standardize testing data using training statistics
X_test_scaled = (X_test - mean) / std

# Display results
print("\nFeature Standardization Completed!")

print("\nOriginal Training Data:")
print(X_train[:3])

print("\nStandardized Training Data:")
print(X_train_scaled[:3])

print("\nMean of Standardized Features:")
print(np.mean(X_train_scaled, axis=0))

print("\nStandard Deviation of Standardized Features:")
print(np.std(X_train_scaled, axis=0))

# Add a bias column

X_train_bias = np.c_[
    np.ones(X_train_scaled.shape[0]),
    X_train_scaled
]

X_test_bias = np.c_[
    np.ones(X_test_scaled.shape[0]),
    X_test_scaled
]

print("\nTraining data with bias:", X_train_bias.shape)
print("Testing data with bias:", X_test_bias.shape)

print("\nFirst training row with bias:")
print(X_train_bias[0])

# Initialize weights and bias to zero

theta = np.zeros(X_train_bias.shape[1])

print("\nInitial model parameters:")
print(theta)



# Test the hypothesis function

initial_predictions = hypothesis(X_train_bias, theta)

print("\nFirst 5 predictions:")
print(initial_predictions[:5])

print("\nFirst 5 actual house values:")
print(y_train[:5])





# Calculate initial training cost
initial_cost = compute_cost(X_train_bias, y_train, theta)

print("\nInitial Training Cost (MSE):")
print(initial_cost)




# Train the linear regression model

learning_rate = 0.01
iterations = 10000

print("\nStarting Gradient Descent Training...\n")

final_theta, cost_history = gradient_descent(
    X_train_bias,
    y_train,
    theta,
    learning_rate,
    iterations
)

print("\nTraining Completed!")

print("\nFinal Model Parameters:")
print(final_theta)

print("\nInitial Training Cost:")
print(initial_cost)

print("\nFinal Training Cost:")
print(cost_history[-1])


# Plot Cost Convergence

import os

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# Create the graph
plt.figure(figsize=(10, 6))

# Plot training cost against iterations
plt.plot(cost_history, label="Training Cost (MSE)", color="blue")

# Add graph title and axis labels
plt.title("Gradient Descent Cost Convergence")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error (MSE)")

# Add grid and legend
plt.grid(True)
plt.legend()

# Save the graph
plt.savefig("results/cost_convergence.png", dpi=300)

# Display the graph
plt.show()

print("\nCost convergence graph saved successfully!")


# Evaluate the model on testing data

# Predict house values using the trained parameters
y_pred = hypothesis(X_test_bias, final_theta)

# Display first 5 predictions
print("\nFirst 5 Predicted House Values:")
print(y_pred[:5])

# Display first 5 actual values
print("\nFirst 5 Actual House Values:")
print(y_test[:5])


# Calculate Test Mean Squared Error manually

test_mse = np.mean((y_test - y_pred) ** 2)

print("\nTest Mean Squared Error (MSE):")
print(test_mse)


# Calculate R² Score manually

# Sum of squared prediction errors
ss_res = np.sum((y_test - y_pred) ** 2)

# Total variation in actual house values
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

# Calculate R²
r2 = 1 - (ss_res / ss_tot)

print("\nTest R² Score:")
print(r2)

# Display final evaluation summary

print("\n========== MODEL EVALUATION ==========")

print(f"Training MSE: {cost_history[-1]:.6f}")
print(f"Testing MSE:  {test_mse:.6f}")
print(f"Testing R²:   {r2:.6f}")

print("======================================")


# Compare with Scikit-Learn

from sklearn.linear_model import LinearRegression

# Create scikit-learn model
sklearn_model = LinearRegression()

# Train using the same standardized training data
sklearn_model.fit(X_train_scaled, y_train)

# Predict using scikit-learn
sklearn_predictions = sklearn_model.predict(X_test_scaled)

# Calculate scikit-learn MSE manually
sklearn_mse = np.mean((y_test - sklearn_predictions) ** 2)

# Calculate scikit-learn R² manually
sklearn_ss_res = np.sum((y_test - sklearn_predictions) ** 2)

sklearn_ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

sklearn_r2 = 1 - (sklearn_ss_res / sklearn_ss_tot)

# Display model comparison

print("\n========== MODEL COMPARISON ==========")

print(f"{'Metric':<20} {'NumPy Model':<20} {'Scikit-Learn':<20}")

print(f"{'MSE':<20} {test_mse:<20.6f} {sklearn_mse:<20.6f}")

print(f"{'R² Score':<20} {r2:<20.6f} {sklearn_r2:<20.6f}")

print("======================================")

# Compare model parameters

print("\n========== PARAMETER COMPARISON ==========")

print(f"{'Parameter':<15} {'NumPy':<20} {'Scikit-Learn':<20}")

# Compare intercept
print(
    f"{'Intercept':<15} "
    f"{final_theta[0]:<20.6f} "
    f"{sklearn_model.intercept_:<20.6f}"
)

# Compare feature coefficients
feature_names = df.drop(columns=["MedHouseVal"]).columns

for i, feature in enumerate(feature_names):
    print(
        f"{feature:<15} "
        f"{final_theta[i + 1]:<20.6f} "
        f"{sklearn_model.coef_[i]:<20.6f}"
    )

print("==========================================")

# Calculate differences between predictions

prediction_difference = np.abs(y_pred - sklearn_predictions)

print("\n========== PREDICTION DIFFERENCE ==========")

print("Mean Absolute Difference:", np.mean(prediction_difference))

print("Maximum Absolute Difference:", np.max(prediction_difference))

print("==========================================")


# Simple Linear Regression

# Select Median Income as the only input feature
X_simple = df[["MedInc"]].values

# Select house value as the target
y_simple = df["MedHouseVal"].values

print("\n========== SIMPLE LINEAR REGRESSION ==========")

print("Feature shape:", X_simple.shape)
print("Target shape:", y_simple.shape)


# Split simple regression data

X_simple_train, X_simple_test, y_simple_train, y_simple_test = train_test_split(
    X_simple,
    y_simple,
    test_size=0.2,
    random_state=42
)

print("Training shape:", X_simple_train.shape)
print("Testing shape:", X_simple_test.shape)


# Standardize Median Income

simple_mean = np.mean(X_simple_train, axis=0)
simple_std = np.std(X_simple_train, axis=0)

# Prevent division by zero
simple_std[simple_std == 0] = 1

X_simple_train_scaled = (X_simple_train - simple_mean) / simple_std
X_simple_test_scaled = (X_simple_test - simple_mean) / simple_std


# Add bias column

X_simple_train_bias = np.c_[
    np.ones(X_simple_train_scaled.shape[0]),
    X_simple_train_scaled
]

X_simple_test_bias = np.c_[
    np.ones(X_simple_test_scaled.shape[0]),
    X_simple_test_scaled
]

print("Training matrix with bias:", X_simple_train_bias.shape)


# Initialize parameters

theta_simple = np.zeros(X_simple_train_bias.shape[1])

print("Initial parameters:", theta_simple)


# Train simple linear regression

simple_theta, simple_cost_history = gradient_descent(
    X_simple_train_bias,
    y_simple_train,
    theta_simple,
    learning_rate=0.01,
    iterations=2000
)

print("\nSimple Regression Final Parameters:")
print(simple_theta)

print("\nSimple Regression Final Training Cost:")
print(simple_cost_history[-1])


# Make predictions

simple_predictions = hypothesis(
    X_simple_test_bias,
    simple_theta
)

# Calculate MSE manually
simple_mse = np.mean(
    (y_simple_test - simple_predictions) ** 2
)

# Calculate R² manually
simple_ss_res = np.sum(
    (y_simple_test - simple_predictions) ** 2
)

simple_ss_tot = np.sum(
    (y_simple_test - np.mean(y_simple_test)) ** 2
)

simple_r2 = 1 - (simple_ss_res / simple_ss_tot)

print("\n========== SIMPLE REGRESSION RESULTS ==========")

print(f"Testing MSE: {simple_mse:.6f}")
print(f"Testing R²: {simple_r2:.6f}")

print("===============================================")


# Step 14: Save experimental results

results_df = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression",
        "Scikit-Learn Linear Regression"
    ],
    "Features": [
        1,
        8,
        8
    ],
    "Test_MSE": [
        simple_mse,
        test_mse,
        sklearn_mse
    ],
    "Test_R2": [
        simple_r2,
        r2,
        sklearn_r2
    ]
})

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nModel comparison results saved successfully!")


