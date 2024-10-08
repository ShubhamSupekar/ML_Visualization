import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# User-Defined Parameters
train_test_ratio = float(input("Enter train-test split ratio (e.g., 0.8 for 80% training): "))
use_regularization = input("Do you want to use regularization (yes/no)? ").strip().lower() == 'yes'
regularization_strength = float(input("Enter regularization strength (alpha) [if applicable]: ") or 0.1) if use_regularization else 0
use_bias = input("Include intercept (bias)? (yes/no): ").strip().lower() == 'yes'
reg_type = input("Which type of regularization (Ridge/Lasso)? [if applicable]: ").strip().lower() if use_regularization else None

# Load dataset from Excel (replace with actual file path)
file_path = 'yahoo_data.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Use only 10% of the total data
# Comment below line if you want to train and test on whole dataset in this im only using 10%v of the dataset to train and test to get better view in graph  
data = data.sample(frac=0.05, random_state=42)

# Select one feature (e.g., 'Open') and the target ('Close*')
X = data[['Open']].values  # Independent variable
y = data['Close*'].values  # Dependent variable

# Split the data based on the user input
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - train_test_ratio), random_state=42)

# Initialize the Linear Regression model or a regularized model based on user input
if use_regularization:
    if reg_type == 'ridge':
        model = Ridge(alpha=regularization_strength, fit_intercept=use_bias)
    elif reg_type == 'lasso':
        model = Lasso(alpha=regularization_strength, fit_intercept=use_bias)
    else:
        print("Invalid regularization type selected. Using Ridge as default.")
        model = Ridge(alpha=regularization_strength, fit_intercept=use_bias)
else:
    model = LinearRegression(fit_intercept=use_bias)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# ----- Graph 1: Data Classified Around the Regression Line -----

plt.figure(figsize=(14, 7))

# Plot the training data points
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

# ----- Regression Coefficient and Intercept -----
print(f"Regression Coefficient (Slope): {model.coef_[0]:.4f}")
print(f"Intercept (Bias): {model.intercept_:.4f}")
