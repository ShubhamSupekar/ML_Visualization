import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('housing.csv')

# Preprocess the data
data = data.drop(columns=['ocean_proximity'])  # Drop categorical column
data = data.dropna()  # Remove any rows with missing values

# Features and target variable
X = data.drop(columns=['median_house_value'])
y = data['median_house_value']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to compare
models = {
    "Linear": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Support Vector": SVR()
}

# Initialize lists for storing metrics
model_names = []
mse_values = []
r2_values = []

# Evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    model_names.append(name)
    mse_values.append(mse)
    r2_values.append(r2)

    print(f"{name} - Mean Squared Error (MSE): {mse:.2f}")
    print(f"{name} - R-squared (R²): {r2:.2f}\n")

# Plotting the results
fig, ax1 = plt.subplots(figsize=(12, 6))

# Create a bar plot for Mean Squared Error
ax1.bar(model_names, mse_values, color='b', alpha=0.6, label='MSE')
ax1.set_ylabel('Mean Squared Error (MSE)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis for R-squared
ax2 = ax1.twinx()
ax2.plot(model_names, r2_values, color='r', marker='o', label='R²')
ax2.set_ylabel('R-squared (R²)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Set title and show legend
plt.title('Model Comparison: MSE and R-squared')
fig.tight_layout()
plt.show()
