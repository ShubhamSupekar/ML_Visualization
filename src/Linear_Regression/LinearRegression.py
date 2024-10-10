import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv('housing.csv')

# Step 1: Ask the user to select the target column
print("Available columns in the dataset:", df.columns)
target_column = input("Enter the target column (the column to predict): ")

# Step 2: Automatically select numerical columns as potential features
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove the target column from the list of features
numerical_columns.remove(target_column)

# Step 3: Calculate the correlation matrix and apply the threshold
correlation_matrix = df[numerical_columns].corr().abs()

# Step 4: Set a threshold for correlation (for example, 0.8)
threshold = 0.8
filtered_columns = numerical_columns.copy()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if correlation_matrix.iloc[i, j] > threshold:
            colname = correlation_matrix.columns[i]
            if colname in filtered_columns:
                print(f"Removing highly correlated feature: {colname}")
                filtered_columns.remove(colname)

print(f"\nFiltered Features after applying correlation threshold: {filtered_columns}")

# Step 5: Prepare train-test split
X_train, X_test, y_train, y_test = train_test_split(df[filtered_columns], df[target_column], test_size=0.2, random_state=42)

# Step 6: Train the linear regression model on the filtered features
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions and evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Step 8: Display the result of the model
print("\nModel Evaluation")
print(f"Features used: {filtered_columns}")
print(f"R² Score: {r2}")
