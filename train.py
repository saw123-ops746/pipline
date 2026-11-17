
import os
import json
import ast
import yaml
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

DATA_PATH = "flipkart_com-ecommerce_sample.csv"
PARAMS_PATH = "params.yaml"

MODEL_DIR = "models"
REPORT_DIR = "reports"


# ==========================================
# 2. LOAD PARAMETERS FROM YAML
# ==========================================

print("\nLoading parameters...")

with open(PARAMS_PATH, "r") as file:
    params = yaml.safe_load(file)

N_ESTIMATORS = params["train"]["n_estimators"]
MAX_DEPTH = params["train"]["max_depth"]
TEST_SIZE = params["train"]["test_size"]
RANDOM_STATE = params["train"]["random_state"]

print("Parameters loaded:")
print("N Estimators:", N_ESTIMATORS)
print("Max Depth:", MAX_DEPTH)
print("Test Size:", TEST_SIZE)
print("Random State:", RANDOM_STATE)


# ==========================================
# 3. LOAD DATASET
# ==========================================

print("\nLoading Flipkart dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip().str.lower()

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 4. DATA CLEANING
# ==========================================

print("\nCleaning dataset...")

TARGET = "discounted_price"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' does not exist."
    )

df[TARGET] = pd.to_numeric(
    df[TARGET].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

df = df.dropna(subset=[TARGET])
df = df[df[TARGET] > 0]

print("Cleaned dataset shape:", df.shape)


# ==========================================
# 5. FEATURE SELECTION
# ==========================================

possible_features = [
    "retail_price",
    "product_rating",
    "overall_rating",
    "brand",
    "product_category_tree"
]

features = [
    col for col in possible_features
    if col in df.columns
]

if not features:
    raise ValueError(
        "No suitable feature columns found."
    )

X = df[features].copy()
y = df[TARGET].copy()


# Extract product category

if "product_category_tree" in X.columns:

    def extract_category(value):

        try:
            parsed = ast.literal_eval(str(value))

            if isinstance(parsed, list) and parsed:
                return str(parsed[0])

        except (ValueError, SyntaxError):
            pass

        return str(value)

    X["product_category_tree"] = (
        X["product_category_tree"].apply(
            extract_category
        )
    )


# ==========================================
# 6. FEATURE PREPROCESSING
# ==========================================

numeric_features = [
    col for col in X.columns
    if col in [
        "retail_price",
        "product_rating",
        "overall_rating"
    ]
]

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

for col in numeric_features:
    X[col] = pd.to_numeric(
        X[col], errors="coerce"
    )

for col in categorical_features:
    X[col] = X[col].fillna("Unknown").astype(str)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ==========================================
# 7. CREATE PREPROCESSING PIPELINE
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ==========================================
# 8. CREATE RANDOM FOREST MODEL
# ==========================================

print("\nCreating Random Forest model...")

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


# ==========================================
# 9. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 10. TRAIN MODEL
# ==========================================

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed successfully!")


# ==========================================
# 11. MODEL PREDICTION
# ==========================================

print("\nMaking predictions...")

predictions = model.predict(X_test)


# ==========================================
# 12. MODEL EVALUATION
# ==========================================

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

metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2_Score": float(r2),
    "Training_Samples": int(len(X_train)),
    "Testing_Samples": int(len(X_test)),
    "N_Estimators": int(N_ESTIMATORS),
    "Max_Depth": int(MAX_DEPTH)
}


# ==========================================
# 13. SAVE MODEL
# ==========================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

model_path = os.path.join(
    MODEL_DIR,
    "price_model.joblib"
)

joblib.dump(
    model,
    model_path
)

print("\nModel saved:", model_path)


# ==========================================
# 14. SAVE METRICS
# ==========================================

metrics_path = os.path.join(
    REPORT_DIR,
    "metrics.json"
)

with open(metrics_path, "w") as file:

    json.dump(
        metrics,
        file,
        indent=4
    )

print("Metrics saved:", metrics_path)


# ==========================================
# 15. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 45)
print("          MODEL EVALUATION RESULTS")
print("=" * 45)

print(f"MAE:       {mae:.4f}")
print(f"RMSE:      {rmse:.4f}")
print(f"R2 Score:  {r2:.4f}")

print("=" * 45)

print("\nTraining completed successfully!")

# ==========================================
# RANDOM FOREST - DECISION TREE VISUALIZATION
# ==========================================

# Create graph folder
os.makedirs("reports/graphs", exist_ok=True)

# Get Random Forest from Pipeline
rf_model = model[-1]

# Select first tree from Random Forest
tree_model = rf_model.estimators_[0]

# Get feature names after preprocessing
try:
    feature_names = model[:-1].get_feature_names_out()
except (AttributeError, ValueError):
    feature_names = None

# Create decision tree graph
plt.figure(figsize=(25, 12))

plot_tree(
    tree_model,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title("Random Forest - Decision Tree Visualization")

# Save graph
graph_path = os.path.abspath(
    "reports/graphs/random_forest_tree.png"
)

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Decision Tree graph generated successfully!")
print("Saved at:", graph_path)

# Open graph automatically in Windows
os.startfile(graph_path)
