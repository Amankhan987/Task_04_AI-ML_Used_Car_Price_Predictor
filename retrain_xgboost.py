"""Rebuilds the XGBoost model on THIS computer, from dataset/cleaned_car_data.csv.
Run from the project folder:   python retrain_xgboost.py
Uses the same split (random_state=42) and the tuned parameters from the notebook."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

DATA = os.path.join("dataset", "cleaned_car_data.csv")

df = pd.read_csv(DATA)
X = df.drop(columns="selling_price")
y = df["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Shapes (should be (6918, 14) (5534, 13) (1384, 13)):", df.shape, X_train.shape, X_test.shape)

num_features = ["km_driven", "mileage", "engine", "max_power", "seats",
                "car_age", "km_per_year", "power_per_litre", "owner_num"]
cat_features = ["fuel", "seller_type", "transmission", "brand"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)])
preprocessor.set_output(transform="pandas")

regressor = XGBRegressor(random_state=42, n_jobs=1, subsample=0.7, reg_lambda=1, n_estimators=800,
                         min_child_weight=1, max_depth=3, learning_rate=0.05, colsample_bytree=1.0)
model = TransformedTargetRegressor(
    regressor=Pipeline([("preprocess", preprocessor), ("model", regressor)]),
    func=np.log1p, inverse_func=np.expm1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Test R2 (notebook: about 0.898):", round(r2_score(y_test, pred), 4))
print("Test MAE (notebook: about 78,075):", round(mean_absolute_error(y_test, pred)))

# Save XGBoost in its own portable format (works on Windows, Linux and every Python version):
# the fitted preprocessing (joblib) + the booster (json). The app puts them back together.
fitted_pipe = model.regressor_
BOOSTER = os.path.join("models", "xgboost_booster.json")
PREPROC = os.path.join("models", "xgboost_preprocess.joblib")
fitted_pipe.named_steps["model"].save_model(BOOSTER)
joblib.dump(fitted_pipe.named_steps["preprocess"], PREPROC)

# check: rebuild from the two saved files and compare with the original predictions
reg2 = XGBRegressor()
reg2.load_model(BOOSTER)
pre2 = joblib.load(PREPROC)
pred2 = np.expm1(reg2.predict(pre2.transform(X_test)))
print("Rebuilt model gives the same predictions:", np.allclose(pred, pred2))

# the old pickle file is damaged on this computer: move it away so it is not used or uploaded
OLD = os.path.join("models", "xgboost.joblib")
if os.path.exists(OLD):
    os.replace(OLD, OLD + ".old")
    print("Moved the old xgboost.joblib to xgboost.joblib.old")
