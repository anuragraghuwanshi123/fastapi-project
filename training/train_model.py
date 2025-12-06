import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from scipy.stats import uniform, randint
from training.train_utils import DATA_FILE_PATH, MODEL_DIR, MODEL_PATH

# Load and clean data
df = (
    pd
    .read_csv(DATA_FILE_PATH)
    .drop_duplicates()
    .drop(columns=['name', 'model', 'edition'])
)

X = df.drop(columns='selling_price')
y = df.selling_price.copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and categorical columns
num_cols = X_train.select_dtypes(include='number').columns.tolist()
cat_cols = [col for col in X_train.columns if col not in num_cols]

# Preprocessing pipelines
num_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

# XGBoost regressor
xgb_model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

# Full pipeline
pipeline = Pipeline(steps=[
    ('pre', preprocessor),
    ('reg', xgb_model)
])

# Define parameter grid for RandomizedSearchCV
param_dist = {
    'reg__n_estimators': randint(100, 1000),
    'reg__max_depth': randint(3, 10),
    'reg__learning_rate': uniform(0.01, 0.3),
    'reg__subsample': uniform(0.6, 0.4),
    'reg__colsample_bytree': uniform(0.6, 0.4)
}

# RandomizedSearchCV
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,           # number of random combinations
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Train
search.fit(X_train, y_train)

print("✔ Best Parameters:", search.best_params_)

# Save the best pipeline
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(search.best_estimator_, MODEL_PATH)

print("✅ XGBoost Model with RandomizedSearchCV Trained and Saved Successfully!")




   