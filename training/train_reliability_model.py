import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# LOAD DATASET
# =========================================================

DATASET = "outputs/bdd100k_training_dataset.csv"

df = pd.read_csv(DATASET)

print(f"Loaded {len(df)} samples")

# =========================================================
# ENCODE CATEGORICAL FEATURES
# =========================================================

df["weather"] = df["weather"].astype("category").cat.codes
df["timeofday"] = df["timeofday"].astype("category").cat.codes

# =========================================================
# FEATURES
# =========================================================

X = df[
    [
        "brightness",
        "contrast",
        "blur_score",
        "edge_count",
        "weather",
        "timeofday"
    ]
]

# =========================================================
# TARGET
# =========================================================

y = df["reliability_score"]

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples : {len(X_test)}")

# =========================================================
# RANDOM FOREST
# =========================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")

# =========================================================
# PREDICTIONS
# =========================================================

predictions = model.predict(X_test)

# =========================================================
# METRICS
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print(feature_importance)

# =========================================================
# SAVE FEATURE IMPORTANCE
# =========================================================

os.makedirs("outputs", exist_ok=True)

feature_importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

# =========================================================
# SAVE MODEL
# =========================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/reliability_model.pkl"
)

print("\nModel saved:")
print("models/reliability_model.pkl")