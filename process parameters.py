import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats.mstats import winsorize

# Load the Excel file
file_path = "Book2.xlsx"
df = pd.read_excel(file_path)

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Define Independent and Dependent Variables
independent_vars = [
    "PIPE OD (INCH)", "PIPE WT (MM)", "C(%)", "Mn(%)", "Si(%)", "Cr(%)", "Ni(%)",
    "Nb(%)", "Ti(%)", "V(%)", "Mo(%)", "Cu(%)", "Al(%)", "N(%)", "B(%)", "CE(PCM)",
    "BEND ANGLE (°)", "BEND RADIUS"
]

dependent_vars = [
    "OD TEMPERATURE INTRADOS (°C)", "WATER FLOW RATE (M3 PER HR)", "SPEED (MM PER MIN)",
    "BEND FREQUENCY (HZ)", "INDUCTION POWER (KW)",
    "WATER PRESSURE (KG PER CM2)", "AIR PRESSURE (KG PER CM2)"
]

# Custom hyperparameters for each dependent variable
hyperparameters = {
    "OD TEMPERATURE INTRADOS (°C)": {"n_estimators": 700, "max_depth": 8, "min_samples_split": 2,
                                     "min_samples_leaf": 2},
    "WATER FLOW RATE (M3 PER HR)": {"n_estimators": 200, "max_depth": 14, "min_samples_split": 5,
                                    "min_samples_leaf": 3},
    "SPEED (MM PER MIN)": {"n_estimators": 200, "max_depth": 12, "min_samples_split": 6, "min_samples_leaf": 3},
    "BEND FREQUENCY (HZ)": {"n_estimators": 200, "max_depth": 22, "min_samples_split": 3, "min_samples_leaf": 1},
    "INDUCTION POWER (KW)": {"n_estimators": 500, "max_depth": 20, "min_samples_split": 4, "min_samples_leaf": 2},
    "WATER PRESSURE (KG PER CM2)": {"n_estimators": 200, "max_depth": 7, "min_samples_split": 8, "min_samples_leaf": 5},
    "AIR PRESSURE (KG PER CM2)": {"n_estimators": 200, "max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 5}
}

# Winsorization to cap outliers
def winsorize_columns(df, columns, limits=(0.05, 0.05)):
    for col in columns:
        df[col] = winsorize(df[col], limits=limits)
    return df

df = winsorize_columns(df, independent_vars + dependent_vars)

# Ensure the model directory exists
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

# Store model performance results
results = {}

# Train models separately with custom hyperparameters
for target in dependent_vars:
    print(f"\n🔹 Training Model for: {target}")

    X = df[independent_vars]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Apply Robust Scaling only on training data
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)  # Fit and transform training data
    X_test = scaler.transform(X_test)  # Only transform test data

    # Get hyperparameters for the target variable
    params = hyperparameters[target]

    # Train Random Forest Model with custom hyperparameters
    rf = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=42
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Evaluate performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ MAE: {mae:.3f}")
    print(f"✅ MSE: {mse:.3f}")
    print(f"✅ R² Score: {r2:.3f}")

    results[target] = {"MAE": mae, "MSE": mse, "R2": r2}

    # Save the trained model
    model_path = os.path.join(model_dir, f"rf_model_{target}.pkl")
    joblib.dump(rf, model_path)

# Save performance summary as a text file
results_df = pd.DataFrame(results).T
results_df.to_csv("model_performance_summary.csv", index=True)

# Plot only R² values with labels
plt.figure(figsize=(8, 4))
ax = sns.barplot(x=results_df.index, y=results_df["R2"], hue=results_df.index, palette="viridis", legend=False)
plt.title("R² Scores for Different Models")
plt.xlabel("Parameters")
plt.ylabel("R² Score")
plt.xticks(rotation=0, fontsize=8)
plt.yticks(fontsize=8)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Add value labels
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=8, color='black')

# Save and show plot
plt.savefig("r2_scores.png")
plt.show()

print("\n✅ Model training complete with custom hyperparameters. R² scores plot saved as 'r2_scores.png'.")