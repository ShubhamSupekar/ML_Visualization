import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data based on y = 2x + 3
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Features (independent variable)
y = 2 * X + 3  # Target (dependent variable)

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the data
model.fit(X, y)

# Predict values based on the trained model for the same training data
y_pred = model.predict(X)

# Predict new values for X = 6, 7, 8
new_X = np.array([6, 7, 8]).reshape(-1, 1)
predicted_y = model.predict(new_X)

# Actual values for X = 6, 7, 8 based on y = 2x + 3
actual_y = 2 * new_X + 3

# Visualize the results
plt.figure(figsize=(8, 5))

# Plot the actual training data points
plt.scatter(X, y, color='blue', label='Training Data (y = 2x + 3)')

# Plot the regression line for the training data
plt.plot(X, y_pred, color='red', label='Fitted Regression Line')

# Plot the new predictions
plt.scatter(new_X, predicted_y, color='green', label='Predicted Values for X = 6, 7, 8')

# Add labels and a title
plt.title('Linear Regression Example (y = 2x + 3)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Show the plot
plt.show()

# Print performance metrics for the training data
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R²): {r2:.2f}')

# Compare predicted values with actual values for X = 6, 7, 8
for i in range(len(new_X)):
    print(f"X = {new_X[i][0]}: Predicted y = {predicted_y[i][0]:.2f}, Actual y = {actual_y[i][0]:.2f}")
    if np.abs(predicted_y[i] - actual_y[i]) > 1e-5:  # Check for significant difference
        print(f"--> Difference detected for X = {new_X[i][0]}: Predicted y = {predicted_y[i][0]:.2f}, Actual y = {actual_y[i][0]:.2f}")
