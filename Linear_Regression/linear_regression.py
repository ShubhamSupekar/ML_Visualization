import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset from Excel (replace with actual file path)
file_path = 'yahoo_data.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Select one feature (e.g., 'Open') and the target ('Close*')
X = data[['Open']].values  # Independent variable
y = data['Close*'].values  # Dependent variable

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model and train it on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# ----- Graph 1: Data Classified Around the Regression Line -----

plt.figure(figsize=(14, 7))

# Plot the training data points with different colors for Open and Close
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Training Data (Close Price)')
plt.plot(X_train, model.predict(X_train), color='green', label='Regression Line')

# Labels and title
plt.title('Data Classified Around the Regression Line')
plt.xlabel('Open Price')
plt.ylabel('Close Price')
plt.legend()

# ----- Graph 2: Predictions vs Actual -----

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='orange', label='Actual Close Prices')
plt.scatter(X_test, y_pred, color='purple', marker='x', label='Predicted Close Prices')

# Plot lines connecting actual and predicted points for clarity
for i in range(len(X_test)):
    plt.plot([X_test[i], X_test[i]], [y_test[i], y_pred[i]], 'r--')

# Labels and title
plt.title('Predicted vs Actual Close Prices')
plt.xlabel('Open Price')
plt.ylabel('Close Price')
plt.legend()

# Show the combined plot
plt.tight_layout()
plt.show()

# ----- Performance Metrics -----
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print performance metrics
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R²): {r2:.2f}')
