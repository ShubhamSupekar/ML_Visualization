import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load your dataset
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

# Step 5: Define a function to evaluate combinations of features
def evaluate_combinations(features_list, target):
    best_combination = None
    best_r2 = -float('inf')
    results = []

    # Test combinations of 2, 3, and so on up to the length of filtered_columns
    for r in range(2, len(features_list) + 1):
        for combo in combinations(features_list, r):
            X_train, X_test, y_train, y_test = train_test_split(df[list(combo)], df[target], test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            results.append((combo, r2))

            if r2 > best_r2:
                best_r2 = r2
                best_combination = combo

    return best_combination, best_r2, results

# Step 6: Evaluate the feature combinations
best_features, best_r2, all_results = evaluate_combinations(filtered_columns, target_column)

# Step 7: Display results for all combinations and the best one
print("\nAll combinations evaluated (sorted by R² score):")
sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
for combo, r2 in sorted_results:
    print(f"Features: {combo}, R² Score: {r2}")

print(f"\nBest combination of features: {best_features}, with an R² score of: {best_r2}")
