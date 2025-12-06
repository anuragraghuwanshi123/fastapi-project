import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from training.train_utils import DATA_FILE_PATH, MODEL_DIR, MODEL_PATH
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV


def load_and_clean_data(path):
    """Loads data, cleans it, and removes outliers (IQR on selling_price)."""
    df = pd.read_csv(DATA_FILE_PATH)

    # Attempt to coerce commonly messy numeric columns to numeric (strip non-numeric chars)
    def clean_numeric(col):
        if col in df.columns:
            df[col] = (df[col]
                       .astype(str)
                       .str.extract(r'([0-9]+(?:\.[0-9]+)?)')[0]  # grab first numeric part
                       .astype(float)
                      )

    for col in ['mileage_mpg', 'engine_cc', 'max_power_bhp', 'torque_nm', 'seats', 'km_driven', 'selling_price']:
        clean_numeric(col)

    # 1. Missing Values (Median Imputation) for selected numeric cols
    numeric_cols = ['mileage_mpg', 'engine_cc', 'max_power_bhp', 'torque_nm', 'seats', 'km_driven', 'selling_price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 2. Outlier Removal (IQR) on selling_price if present
    if 'selling_price' in df.columns:
        Q1 = df['selling_price'].quantile(0.25)
        Q3 = df['selling_price'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        df = df[df['selling_price'] <= upper_bound].copy()

    return df.reset_index(drop=True)

def feature_engineering(df):
    """Creates new smart features and drops less useful cols."""
    current_year = pd.Timestamp.now().year

    # 1. Car Age
    if 'year' in df.columns:
        df['car_age'] = current_year - df['year'].astype(float)
        df = df.drop('year', axis=1)

    # 2. Log Transform Target (Price) if selling_price exists
    if 'selling_price' in df.columns:
        df['price_log'] = np.log1p(df['selling_price'])

    # 3. Log Transform Skewed Feature (KM Driven)
    if 'km_driven' in df.columns:
        df['km_log'] = np.log1p(df['km_driven'].clip(lower=0))
        df = df.drop('km_driven', axis=1)

    # 4. Performance Ratio (Power per CC) with safety for zeros/missing
    if {'max_power_bhp', 'engine_cc'}.issubset(df.columns):
        df['engine_cc'] = df['engine_cc'].replace({0: np.nan})
        df['performance_ratio'] = df['max_power_bhp'] / df['engine_cc']
        # replace inf/nan with small number
        df['performance_ratio'] = df['performance_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Drop IDs/text columns that won't help the model directly
    df = df.drop(columns=['name', 'edition'], errors='ignore')

    return df

def target_encode(train_df, test_df, col, target_col):
    """
    Replaces a category with the average target value of that category.
    Computed on TRAIN only to avoid data leakage.
    Returns two Series: train_encoded, test_encoded (aligned by index of inputs)
    """
    means = train_df.groupby(col)[target_col].mean()
    train_encoded = train_df[col].map(means)
    test_encoded = test_df[col].map(means)

    global_mean = train_df[target_col].mean()
    train_encoded = train_encoded.fillna(global_mean)
    test_encoded = test_encoded.fillna(global_mean)

    return train_encoded, test_encoded

def preprocess_and_split(df):
    """
    Splits data FIRST, then applies encoding to prevent leakage.
    Returns X_train, X_test, y_train_log, y_test_log, y_test_actual
    """
    # Ensure target exists
    if 'price_log' not in df.columns or 'selling_price' not in df.columns:
        raise ValueError("DataFrame must contain 'price_log' and 'selling_price'.")

    X = df.drop(['selling_price', 'price_log'], axis=1)
    y_log = df['price_log']
    y_actual = df['selling_price']

    # Single consistent split for X and both label variants
    X_train, X_test, y_train_log, y_test_log, y_train_actual, y_test_actual = train_test_split(
        X, y_log, y_actual, test_size=0.2, random_state=42
    )

    # Target Encoding for high-cardinality columns if they exist
    for col in ['company', 'model']:
        if col in X_train.columns:
            train_temp = X_train.copy()
            train_temp['target'] = y_train_log.values  # attach target for grouping

            train_enc, test_enc = target_encode(train_temp, X_test, col, 'target')

            X_train[f'{col}_encoded'] = train_enc.values
            X_test[f'{col}_encoded'] = test_enc.values

            X_train = X_train.drop(col, axis=1)
            X_test = X_test.drop(col, axis=1)

    # Owner mapping (robust: use contains keyword and default)
    owner_mapping = {'Test Drive Car': 5, 'First': 4, 'Second': 3, 'Third': 2, 'Fourth & Above': 1}
    if 'owner' in X_train.columns:
        X_train['owner'] = (X_train['owner']
                            .astype(str)
                            .map(owner_mapping)
                            .fillna(2).astype(int))
        X_test['owner'] = (X_test['owner']
                           .astype(str)
                           .map(owner_mapping)
                           .fillna(2).astype(int))

    # One-Hot Encoding for categorical columns if present
    cat_cols = [c for c in ['fuel', 'seller_type', 'transmission'] if c in X_train.columns]
    if cat_cols:
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

    # Ensure train and test have same columns
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    return X_train, X_test, y_train_log, y_test_log, y_test_actual

def tune_and_train(X_train, y_train):
    print("Starting Hyperparameter Tuning for XGBoost...")

    param_grid = {
        'n_estimators': [1000, 1500],
        'learning_rate': [0.01, 0.02],
        'max_depth': [5, 6],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8]
    }

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

    random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid,
                                       n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)

    random_search.fit(X_train, y_train)
    print(f"Best Parameters: {random_search.best_params_}")

    return random_search.best_estimator_

if __name__ == "__main__":
    # 1. Load
    df = load_and_clean_data(DATA_FILE_PATH)   # adjust path if needed

    # 2. Engineer Features
    df_processed = feature_engineering(df)

    # 3. Split & Preprocess
    X_train, X_test, y_train_log, y_test_log, y_test_actual = preprocess_and_split(df_processed)

    # 4. Train (tuning)
    rf_model = tune_and_train(X_train, y_train_log)


os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(rf_model, MODEL_PATH)
   