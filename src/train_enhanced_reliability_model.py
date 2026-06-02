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

DATASET_PATH = (
    "outputs/enhanced_training_dataset.csv"
)

df = pd.read_csv(DATASET_PATH)

print(f"Loaded {len(df)} samples")

# =========================================================
# ENCODE CATEGORICAL FEATURES
# =========================================================

df["weather"] = (
    df["weather"]
    .astype("category")
    .cat.codes
)

df["timeofday"] = (
    df["timeofday"]
    .astype("category")
    .cat.codes
)

# =========================================================
# FEATURES
# =========================================================

FEATURE_COLUMNS = [
    "brightness",
    "contrast",
    "blur_score",
    "edge_count",
    "edge_density",
    "entropy",
    "object_count",
    "person_count",
    "vehicle_count",
    "weather",
    "timeofday"
]

X = df[FEATURE_COLUMNS]

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

print(f"\nTraining Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# =========================================================
# RANDOM FOREST MODEL
# =========================================================

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
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

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        by="importance",
        ascending=False
    )
)

print("\n================================")
print("FEATURE IMPORTANCE")
print("================================")

print(importance_df)

# =========================================================
# SAVE FEATURE IMPORTANCE
# =========================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

importance_df.to_csv(
    "outputs/enhanced_feature_importance.csv",
    index=False
)

# =========================================================
# SAVE TEST PREDICTIONS
# =========================================================

prediction_df = pd.DataFrame({
    "actual": y_test.values,
    "predicted": predictions
})

prediction_df.to_csv(
    "outputs/model_predictions.csv",
    index=False
)

# =========================================================
# SAVE MODEL
# =========================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/enhanced_reliability_model.pkl"
)

print("\n================================")
print("MODEL SAVED")
print("================================")

print(
    "models/enhanced_reliability_model.pkl"
)

print(
    "\nFeature importance saved:"
)

print(
    "outputs/enhanced_feature_importance.csv"
)

print(
    "\nPredictions saved:"
)

print(
    "outputs/model_predictions.csv"
)