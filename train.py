
import os
import json
import ast
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------------
# 1. Project configuration
# ----------------------------------

DATA_PATH = "flipkart_com-ecommerce_sample.csv"

MODEL_DIR = "models"
REPORT_DIR = "reports"

RANDOM_STATE = 42


# ----------------------------------
# 2. Load dataset
# ----------------------------------

print("Loading Flipkart dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Available columns:", df.columns.tolist())


# ----------------------------------
# 3. Check target column
# ----------------------------------

TARGET = "discounted_price"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found. "
        "Please check your dataset columns."
    )

# Convert target to numeric
df[TARGET] = pd.to_numeric(
    df[TARGET].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

# Remove rows with missing or invalid prices
df = df.dropna(subset=[TARGET])
df = df[df[TARGET] > 0]


# ----------------------------------
# 4. Select useful features
# ----------------------------------

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

# Extract the main category from category tree
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
        X["product_category_tree"]
        .apply(extract_category)
    )


# ----------------------------------
# 5. Identify numeric and categorical columns
# ----------------------------------

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

# Convert numeric columns
for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Convert categorical columns
for col in categorical_features:
    X[col] = X[col].fillna("Unknown").astype(str)

# Replace infinite numeric values
X = X.replace([float("inf"), float("-inf")], float("nan"))


# ----------------------------------
# 6. Preprocessing
# ----------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ----------------------------------
# 7. Create ML model
# ----------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


# ----------------------------------
# 8. Split dataset
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ----------------------------------
# 9. Train model
# ----------------------------------

print("Training Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ----------------------------------
# 10. Evaluate model
# ----------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(y_test, predictions)

metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2_Score": float(r2),
    "Training_Samples": int(len(X_train)),
    "Testing_Samples": int(len(X_test))
}


# ----------------------------------
# 11. Save trained model
# ----------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

model_path = os.path.join(
    MODEL_DIR,
    "price_model.joblib"
)

joblib.dump(model, model_path)


# ----------------------------------
# 12. Save evaluation metrics
# ----------------------------------

metrics_path = os.path.join(
    REPORT_DIR,
    "metrics.json"
)

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)


# ----------------------------------
# 13. Display results
# ----------------------------------

print("\n========== MODEL RESULTS ==========")

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

print("\nModel saved:", model_path)
print("Metrics saved:", metrics_path)

print("\nTraining completed successfully!")
