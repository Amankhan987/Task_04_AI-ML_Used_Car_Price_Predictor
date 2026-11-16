"""Run from the project folder:  python check_models.py
Shows the size of every model file and whether it loads."""
import os
import joblib

folder = "models"
for f in sorted(os.listdir(folder)):
    if not f.endswith(".joblib"):
        continue
    path = os.path.join(folder, f)
    size_mb = os.path.getsize(path) / 1024 / 1024
    try:
        joblib.load(path)
        status = "OK"
    except Exception as e:
        status = f"FAILED ({type(e).__name__}: {e})"
    print(f"{f:28s} {size_mb:7.2f} MB   {status}")

booster = os.path.join(folder, "xgboost_booster.json")
if os.path.exists(booster):
    try:
        from xgboost import XGBRegressor
        XGBRegressor().load_model(booster)
        print(f"{'xgboost_booster.json':28s} {os.path.getsize(booster) / 1024 / 1024:7.2f} MB   OK (portable XGBoost file)")
    except Exception as e:
        print(f"xgboost_booster.json FAILED ({type(e).__name__}: {e})")
