import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from training.train_utils import DATA_FILE_PATH, MODEL_DIR, MODEL_PATH

### ----------------- 1. LOAD + CLEAN -----------------
def load_and_clean_data(path):
    df = pd.read_csv(path)

    numeric_columns = ['mileage_mpg', 'engine_cc', 'max_power_bhp', 'torque_nm', 'seats', 'km_driven', 'selling_price']
    for col in numeric_columns:
        df[col] = (
            df[col].astype(str).str.extract(r'([0-9]+(?:\.[0-9]+)?)')[0].astype(float)
        )
        df[col].fillna(df[col].median(), inplace=True)

    # Remove selling_price outliers
    Q1, Q3 = df['selling_price'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[df['selling_price'] <= (Q3 + 1.5 * IQR)]

    return df.reset_index(drop=True)


### ----------------- 2. FEATURE ENGINEERING -----------------
def feature_engineering(df):
    df["car_age"] = pd.Timestamp.now().year - df["year"].astype(float)
    df["price_log"] = np.log1p(df["selling_price"])

    df["km_log"] = np.log1p(df["km_driven"].clip(lower=0))
    df.drop(["year", "km_driven", "name", "edition"], axis=1, errors="ignore", inplace=True)

    df["performance_ratio"] = df["max_power_bhp"] / df["engine_cc"].replace(0, np.nan)
    df["performance_ratio"].replace([np.nan, np.inf, -np.inf], 0, inplace=True)

    return df


### ----------------- 3. PREPROCESS -----------------
owner_mapping = {'Test Drive Car': 5, 'First': 4, 'Second': 3, 'Third': 2, 'Fourth & Above': 1}

def preprocess(df):
    X = df.drop(["selling_price", "price_log"], axis=1)
    y = df["price_log"]

    X["owner"] = X["owner"].astype(str).map(owner_mapping).fillna(2).astype(int)

    categorical_cols = ["fuel", "seller_type", "transmission", "company", "model"]
    X = pd.get_dummies(X, columns=[c for c in categorical_cols if c in X.columns], drop_first=True)

    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns


### ----------------- 4. TRAIN MODEL -----------------
def train_model(X_train, y_train):
    param_grid = {
        "n_estimators": [800, 1200],
        "learning_rate": [0.01, 0.03],
        "max_depth": [5, 6],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9]
    }

    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(model, param_grid, cv=3, n_iter=10, random_state=42, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    print("\n✔ Best Params:", search.best_params_)
    return search.best_estimator_


### ----------------- 5. SAVE PIPELINE -----------------
if __name__ == "__main__":
    df = load_and_clean_data(DATA_FILE_PATH)
    df = feature_engineering(df)

    (X_train, X_test, y_train, y_test), feature_names = preprocess(df)

    model = train_model(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump({
        "model": model,
        "features": list(feature_names),
        "owner_mapping": owner_mapping
    }, MODEL_PATH)

    print("\n🎉 Model Training Complete and Saved Successfully!")

   