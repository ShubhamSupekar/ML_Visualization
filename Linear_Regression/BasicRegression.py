import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define the dataset
X = np.array([1, 2, 3]).reshape(-1, 1)  # Features (x values)
y = np.array([1, 4, 9])  # Target values (y = x^2)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict y for x = 4
x_pred = np.array([[4]])  # Input value for prediction
y_pred = model.predict(x_pred)

# Print the predicted value
print(f'Predicted y value for x = 4: {y_pred[0]:.2f}')

# Visualize the data and the prediction
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='green', label='Regression Line')
plt.scatter(x_pred, y_pred, color='red', label=f'Predicted (x=4, y={y_pred[0]:.2f})')

# Labels and legend
plt.title('Basic Regression Example')
plt.xlabel('x')
plt.ylabel('y (x^2)')
plt.legend()
plt.show()
