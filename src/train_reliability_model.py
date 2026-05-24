import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# =========================================================
# ML-Based Camera Reliability Prediction Engine
# =========================================================

print("Loading synthetic training dataset...\n")

# =========================================================
# SYNTHETIC TRAINING DATA
# =========================================================

data = {
    "brightness": [200, 180, 40, 50, 120, 220, 30, 90],
    "blur_score": [500, 450, 20, 35, 200, 600, 15, 120],
    "edge_count": [12000, 11000, 3000, 4000, 7000, 14000, 2500, 6000],
    "contrast": [80, 75, 20, 25, 50, 90, 15, 40],
    "reliability": [0.95, 0.90, 0.20, 0.30, 0.70, 0.98, 0.15, 0.55]
}

df = pd.DataFrame(data)

print("Training Dataset:")
print(df)

# =========================================================
# FEATURES AND LABELS
# =========================================================

# Input features
X = df[["brightness", "blur_score", "edge_count", "contrast"]]

# Output labels
y = df["reliability"]

print("\nFeatures:")
print(X)

print("\nLabels:")
print(y)

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# =========================================================
# DECISION TREE MODEL
# =========================================================

print("\nTraining Decision Tree Model...")

tree_model = DecisionTreeRegressor(random_state=42)

tree_model.fit(X_train, y_train)

print("Decision Tree trained successfully.")

# Predictions
tree_predictions = tree_model.predict(X_test)

print("\nDecision Tree Predictions:")
print(tree_predictions)

# Evaluation
tree_mae = mean_absolute_error(y_test, tree_predictions)

print("\nDecision Tree MAE:", tree_mae)

# =========================================================
# RANDOM FOREST MODEL
# =========================================================

print("\nTraining Random Forest Model...")

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, y_train)

print("Random Forest trained successfully.")

# Predictions
forest_predictions = forest_model.predict(X_test)

print("\nRandom Forest Predictions:")
print(forest_predictions)

# Evaluation
forest_mae = mean_absolute_error(y_test, forest_predictions)

print("\nRandom Forest MAE:", forest_mae)

# =========================================================
# MODEL COMPARISON
# =========================================================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print("Decision Tree MAE:", tree_mae)
print("Random Forest MAE:", forest_mae)

# =========================================================
# SAVE MODEL
# =========================================================

print("\nSaving trained model...")

# Create models folder if not exists
import os
os.makedirs("models", exist_ok=True)

joblib.dump(
    forest_model,
    "models/reliability_model.pkl"
)

print("Model saved successfully.")

# =========================================================
# LOAD MODEL AGAIN
# =========================================================

loaded_model = joblib.load(
    "models/reliability_model.pkl"
)

print("\nModel loaded successfully.")

# =========================================================
# CUSTOM PREDICTION TEST
# =========================================================

print("\nTesting custom environment prediction...")

# Example:
# [brightness, blur_score, edge_count, contrast]

sample = [[50, 30, 4000, 25]]

prediction = loaded_model.predict(sample)

print("\nCustom Reliability Prediction:", prediction[0])

# =========================================================
# SAVE RESULTS
# =========================================================

results = {
    "decision_tree_mae": [tree_mae],
    "random_forest_mae": [forest_mae],
    "sample_prediction": [prediction[0]]
}

results_df = pd.DataFrame(results)

# Create outputs folder if not exists
os.makedirs("outputs", exist_ok=True)

results_df.to_csv(
    "outputs/ml_model_results.csv",
    index=False
)

print("\nML results saved to outputs/ml_model_results.csv")

print("\n===================================================")
print("ML-Based Reliability Prediction Engine Completed")
print("===================================================")