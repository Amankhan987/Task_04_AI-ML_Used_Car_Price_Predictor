import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Used Car Price Predictor",
                   page_icon="🚗", layout="wide")

AUTHOR = "Aman Hameed Ullah"
BASE = os.path.dirname(os.path.abspath(__file__))


def find_folder(*parts):
    """Looks next to app.py first, then one level up (so both folder layouts work)."""
    for root in (BASE, os.path.join(BASE, "..")):
        path = os.path.normpath(os.path.join(root, *parts))
        if os.path.exists(path):
            return path
    return os.path.normpath(os.path.join(BASE, *parts))


MODELS = find_folder("models")
DATA = find_folder("dataset", "cleaned_car_data.csv")


# ------------------------------------------------------------------ loading helpers
@st.cache_data
def load_info():
    with open(os.path.join(MODELS, "model_info.json")) as f:
        return json.load(f)


class NativeXGBModel:
    """XGBoost rebuilt from its portable files: fitted preprocessing + booster (json)."""

    def __init__(self, pre_path, booster_path):
        from xgboost import XGBRegressor
        self.pre = joblib.load(pre_path)
        self.reg = XGBRegressor()
        self.reg.load_model(booster_path)

    def predict(self, X):
        # the model works on log(price)
        return np.expm1(self.reg.predict(self.pre.transform(X)))


@st.cache_resource
def load_model(file_name):
    if file_name == "xgboost.joblib":
        booster = os.path.join(MODELS, "xgboost_booster.json")
        pre = os.path.join(MODELS, "xgboost_preprocess.joblib")
        if os.path.exists(booster) and os.path.exists(pre):
            return NativeXGBModel(pre, booster)
    return joblib.load(os.path.join(MODELS, file_name))


@st.cache_data
def load_data():
    return pd.read_csv(DATA)


@st.cache_data
def load_csv(name, index_col=0):
    return pd.read_csv(os.path.join(MODELS, name), index_col=index_col)


def inr(x):
    return f"₹ {x:,.0f}"


def lakh(x):
    return f"{x / 1e5:.2f} lakh"


try:
    info = load_info()
    data = load_data()
except Exception as e:
    st.error("The model files or the dataset could not be loaded. Put the folders `models` and `dataset` "
             "in the same folder as `app.py`.")
    st.write("Looking for models in:", MODELS)
    st.write("Looking for data in:", DATA)
    st.exception(e)
    st.stop()

CATS = info["categories"]
RNG = info["input_ranges"]
REF_YEAR = info["ref_year"]
MODEL_NAMES = list(info["model_files"].keys())
OWNER_MAP = {name: i + 1 for i, name in enumerate(CATS["owner"])}


def build_input(brand, year, km, fuel, seller, transmission, owner, mileage, engine, power, seats):
    """Turns the values entered by the user into the columns the model expects."""
    age = REF_YEAR - year
    return pd.DataFrame([{
        "km_driven": km, "mileage": mileage, "engine": engine, "max_power": power, "seats": seats,
        "car_age": age, "km_per_year": round(km / max(age, 1)),
        "power_per_litre": round(power / engine * 1000, 2), "owner_num": OWNER_MAP[owner],
        "fuel": fuel, "seller_type": seller, "transmission": transmission, "brand": brand}])


def clip(value, low, high):
    return max(low, min(high, value))


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🚗 Used Car Price Predictor")
page = st.sidebar.radio("Go to", ["🔮 Predict Price", "⭐ Feature Importance", "📈 Data Visualization",
                                  "🏁 Model Performance", "ℹ️ About"])
st.sidebar.markdown("---")
model_name = st.sidebar.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(info["final_model"]),
                                  help="All five tuned models can be used for the prediction. "
                                       "The Random Forest is the final model.")
if model_name == info["final_model"]:
    st.sidebar.success("Final model")
m = info["metrics_after_tuning"][model_name]
st.sidebar.markdown(f"**Test R²:** {m['r2']:.4f}  \n**Average error (MAE):** {inr(m['mae'])}  \n"
                    f"**Average % error:** {m['mape']:.1f}%")
st.sidebar.caption(
    "Trained on CarDekho listings (India, up to 2020). The estimate is not a valuation.")

# ================================================================== PAGE 1: PREDICT
if page == "🔮 Predict Price":
    st.title("🔮 Predict the Price of a Used Car")
    st.write("Enter the details of the car and click **Predict Price**.")

    with st.form("car_form"):
        c1, c2, c3 = st.columns(3)
        brand = c1.selectbox("Brand", CATS["brand"],
                             index=CATS["brand"].index("Maruti") if "Maruti" in CATS["brand"] else 0)
        year = c2.number_input("Manufacturing year", RNG["year_min"], RNG["year_max"],
                               clip(2015, RNG["year_min"], RNG["year_max"]), step=1)
        km = c3.number_input("Kilometers driven", 0,
                             RNG["km_driven_max"], 50000, step=1000)
        c4, c5, c6 = st.columns(3)
        fuel = c4.selectbox("Fuel type", CATS["fuel"], index=CATS["fuel"].index(
            "Petrol") if "Petrol" in CATS["fuel"] else 0)
        seller = c5.selectbox("Seller type", CATS["seller_type"],
                              index=CATS["seller_type"].index("Individual") if "Individual" in CATS["seller_type"] else 0)
        transmission = c6.selectbox("Transmission", CATS["transmission"])
        c7, c8, c9 = st.columns(3)
        owner = c7.selectbox("Owner", CATS["owner"])
        mileage = c8.number_input("Mileage (km per litre, km/kg for CNG/LPG)", float(RNG["mileage"][0]),
                                  float(RNG["mileage"][1]), clip(19.0, RNG["mileage"][0], RNG["mileage"][1]), step=0.1)
        engine = c9.number_input("Engine size (cc)", int(RNG["engine"][0]), int(RNG["engine"][1]),
                                 int(clip(1248, RNG["engine"][0], RNG["engine"][1])), step=50)
        c10, c11, _ = st.columns(3)
        power = c10.number_input("Maximum power (bhp)", float(RNG["max_power"][0]), float(RNG["max_power"][1]),
                                 clip(82.0, RNG["max_power"][0], RNG["max_power"][1]), step=1.0)
        seats = c11.selectbox("Seats", list(range(int(RNG["seats"][0]), int(RNG["seats"][1]) + 1)),
                              index=list(range(int(RNG["seats"][0]), int(
                                  RNG["seats"][1]) + 1)).index(5)
                              if 5 in range(int(RNG["seats"][0]), int(RNG["seats"][1]) + 1) else 0)
        submitted = st.form_submit_button("Predict Price", type="primary")

    if submitted:
        st.session_state["car"] = build_input(brand, int(year), km, fuel, seller, transmission, owner,
                                              mileage, engine, power, seats)
        st.session_state["car_label"] = f"{int(year)} {brand} ({fuel}, {transmission})"

    if "car" in st.session_state:
        car = st.session_state["car"]
        try:
            model = load_model(info["model_files"][model_name])
            price = float(model.predict(car)[0])
        except Exception as e:
            st.error("The model could not be loaded or used. This is usually a library version problem: install "
                     f"scikit-learn {info.get('sklearn_version')} and xgboost {info.get('xgboost_version')} "
                     "(see requirements.txt).")
            st.exception(e)
            st.stop()
        rg = info["prediction_range"][model_name]
        low, high = price * rg["lower_factor"], price * rg["upper_factor"]

        st.markdown("---")
        st.subheader(f"Result for: {st.session_state['car_label']}")
        r1, r2, r3 = st.columns(3)
        r1.metric("Estimated price", lakh(price),
                  inr(price), delta_color="off")
        r2.metric("Likely price range",
                  f"{low / 1e5:.2f} to {high / 1e5:.2f} lakh")
        r3.metric("Chance the real price is inside the range",
                  f"about {rg['test_coverage'] * 100:.0f}%")
        st.success(f"The {model_name} model estimates **{inr(price)}** ({lakh(price)}). "
                   f"The real price is likely between **{inr(low)}** and **{inr(high)}**.")
        if price > 3000000:
            st.warning(
                "Prices above about 30 lakh are less reliable, because there are few such cars in the data.")
        if int(car["km_driven"][0]) > 250000 or int(car["car_age"][0]) > 25:
            st.warning(
                "This car is very old or has very high mileage, which is rare in the training data.")

        st.subheader("What the model sees")
        left, right = st.columns(2)
        with left:
            st.markdown("**Calculated features of this car**")
            st.table(pd.DataFrame({
                "Feature": ["Car age (years)", "Kilometers per year", "Power per litre (bhp/l)", "Owner number"],
                "Value": [f"{int(car['car_age'][0])}", f"{int(car['km_per_year'][0]):,}",
                          f"{car['power_per_litre'][0]:.1f}", f"{int(car['owner_num'][0])}"]}))
            perm = load_csv("permutation_importance.csv")[
                "importance"].sort_values(ascending=False).head(3)
            st.caption("Most important inputs of the model: " + ", ".join(
                f"{c.replace('_', ' ')}" for c in perm.index) + ". See the Feature Importance page.")
        with right:
            similar = data[(data["brand"] == car["brand"][0]) & (
                (data["car_age"] - car["car_age"][0]).abs() <= 2)]
            label = f"{car['brand'][0]} cars of a similar age (±2 years)"
            if len(similar) < 5:
                similar = data[data["brand"] == car["brand"][0]]
                label = f"{car['brand'][0]} cars of all ages"
            if len(similar) >= 3:
                st.markdown(
                    f"**Compared with {len(similar)} {label} in the data**")
                st.table(pd.DataFrame({
                    "Price": ["Lowest 25%", "Median", "Highest 25%"],
                    "Rupees": [inr(similar["selling_price"].quantile(0.25)), inr(similar["selling_price"].median()),
                               inr(similar["selling_price"].quantile(0.75))]}))
            else:
                st.info(
                    "There are too few cars of this brand in the data for a comparison.")

        st.subheader("All five models for this car")
        rows, failed = [], []
        for name in MODEL_NAMES:
            try:
                p = float(load_model(
                    info["model_files"][name]).predict(car)[0])
            except Exception as e:
                failed.append(f"{name} ({type(e).__name__})")
                continue
            r = info["prediction_range"][name]
            rows.append({"Model": name, "Estimated price": p, "Lower": p *
                        r["lower_factor"], "Upper": p * r["upper_factor"]})
        if failed:
            st.warning(
                "These models could not be loaded and are skipped: " + ", ".join(failed))
        comp = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=comp["Model"], y=comp["Estimated price"] / 1e5, name="Estimated price",
                             marker_color="#2a9d8f",
                             error_y=dict(type="data", symmetric=False,
                                          array=(
                                              comp["Upper"] - comp["Estimated price"]) / 1e5,
                                          arrayminus=(comp["Estimated price"] - comp["Lower"]) / 1e5)))
        fig.update_layout(title="Estimated price and range of each model", yaxis_title="Price (lakh INR)",
                          xaxis_title="Model", height=420)
        st.plotly_chart(fig)
        st.dataframe(comp.style.format({"Estimated price": "{:,.0f}", "Lower": "{:,.0f}", "Upper": "{:,.0f}"}),
                     hide_index=True)
    else:
        st.info(
            "Fill in the form above and click **Predict Price** to see the estimate.")

# ================================================================== PAGE 2: FEATURE IMPORTANCE
elif page == "⭐ Feature Importance":
    st.title("⭐ Which Car Details Matter Most?")
    st.write("Two ways of measuring the importance of the inputs for the final model "
             f"({info['final_model']}).")
    perm = load_csv("permutation_importance.csv")[
        "importance"].sort_values(ascending=True)
    fig = px.bar(perm, orientation="h", labels={"value": "Drop in R² when the column is shuffled", "index": "Column"},
                 title="Permutation importance (calculated on the test cars)", color_discrete_sequence=["#e76f51"])
    fig.update_layout(showlegend=False, height=520)
    st.plotly_chart(fig)
    st.caption("If the values of a column are shuffled randomly, the model gets worse by this much. "
               "A large drop means that the model depends strongly on the column.")

    imp = load_csv("model_importance.csv")[
        "importance"].sort_values(ascending=False).head(15)
    imp.index = [i.split("__", 1)[-1] for i in imp.index]
    fig2 = px.bar(imp.sort_values(), orientation="h", labels={"value": "Importance", "index": "Feature"},
                  title="Model importance (top 15 encoded features)", color_discrete_sequence=["#2a9d8f"])
    fig2.update_layout(showlegend=False, height=520)
    st.plotly_chart(fig2)
    st.caption("How much each feature helped the trees to reduce the error during training. A brand appears "
               "as many separate columns here, which hides its total effect.")
    top = perm.sort_values(ascending=False)
    st.info(f"**Reading the charts:** the most important inputs are **{top.index[0].replace('_', ' ')}**, "
            f"**{top.index[1].replace('_', ' ')}** and **{top.index[2].replace('_', ' ')}**. "
            "A powerful, new car is expensive; the kilometers, the transmission and the brand have a smaller effect.")

# ================================================================== PAGE 3: DATA VISUALIZATION
elif page == "📈 Data Visualization":
    st.title("📈 Explore the Used Car Data")
    d = data.assign(price_lakh=data["selling_price"] / 1e5)
    f1, f2, f3 = st.columns(3)
    brands = f1.multiselect("Brand", sorted(d["brand"].unique()), default=[])
    fuels = f2.multiselect("Fuel type", sorted(
        d["fuel"].unique()), default=sorted(d["fuel"].unique()))
    ages = f3.slider("Car age (years)", 0, int(
        d["car_age"].max()), (0, int(d["car_age"].max())))
    d = d[d["fuel"].isin(fuels) & d["car_age"].between(*ages)]
    if brands:
        d = d[d["brand"].isin(brands)]
    if d.empty:
        st.warning("No cars match the selected filters.")
        st.stop()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cars", f"{len(d):,}")
    k2.metric("Median price", lakh(d["selling_price"].median()))
    k3.metric("Average age", f"{d['car_age'].mean():.1f} years")
    k4.metric("Median kilometers", f"{d['km_driven'].median():,.0f}")

    t1, t2, t3, t4 = st.tabs(
        ["Price distribution", "Price and age", "Brands and fuel", "Correlations"])
    with t1:
        fig = px.histogram(d, x="price_lakh", nbins=60, color_discrete_sequence=["#2a9d8f"],
                           labels={"price_lakh": "Selling price (lakh INR)"}, title="Distribution of the selling price")
        fig.update_layout(yaxis_title="Number of cars")
        st.plotly_chart(fig)
        st.caption(
            "Most cars cost a few lakh rupees; a small number of luxury cars are far more expensive.")
    with t2:
        s = d.sample(min(3000, len(d)), random_state=42)
        fig = px.scatter(s, x="car_age", y="price_lakh", color="fuel", opacity=0.5, log_y=True,
                         labels={
                             "car_age": "Car age (years)", "price_lakh": "Price (lakh INR, log scale)"},
                         title="Selling price against car age")
        st.plotly_chart(fig)
        fig = px.scatter(s, x="max_power", y="price_lakh", color="transmission", opacity=0.5, log_y=True,
                         labels={
                             "max_power": "Maximum power (bhp)", "price_lakh": "Price (lakh INR, log scale)"},
                         title="Selling price against maximum power")
        st.plotly_chart(fig)
    with t3:
        counts = d["brand"].value_counts()
        big = counts[counts >= 10].index
        med = d[d["brand"].isin(big)].groupby(
            "brand")["price_lakh"].median().sort_values(ascending=False).head(15)
        fig = px.bar(med.sort_values(), orientation="h", labels={"value": "Median price (lakh INR)", "index": "Brand"},
                     title="Median price of the most expensive brands (at least 10 cars)",
                     color_discrete_sequence=["#e76f51"])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)
        g = d.groupby(["fuel", "transmission"])[
            "price_lakh"].median().reset_index()
        fig = px.bar(g, x="fuel", y="price_lakh", color="transmission", barmode="group",
                     labels={
                         "price_lakh": "Median price (lakh INR)", "fuel": "Fuel type"},
                     title="Median price by fuel type and transmission")
        st.plotly_chart(fig)
    with t4:
        cols = ["selling_price", "km_driven", "mileage", "engine", "max_power", "seats", "car_age",
                "km_per_year", "power_per_litre", "owner_num"]
        fig = px.imshow(d[cols].corr().round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlation heatmap")
        fig.update_layout(height=600)
        st.plotly_chart(fig)
        st.caption(
            "The price is most strongly related to the maximum power and (negatively) to the age of the car.")

# ================================================================== PAGE 4: MODEL PERFORMANCE
elif page == "🏁 Model Performance":
    st.title("🏁 How Good Are the Models?")
    am = pd.DataFrame(info["metrics_after_tuning"]).T
    am.index.name = "Model"
    am = am.rename(columns={"mae": "MAE (₹)", "rmse": "RMSE (₹)", "r2": "R² after tuning",
                            "mape": "MAPE (%)", "r2_before": "R² before tuning"}).sort_values("R² after tuning", ascending=False)
    st.subheader("Test-set results of the tuned models")
    st.dataframe(am[["MAE (₹)", "RMSE (₹)", "R² before tuning", "R² after tuning", "MAPE (%)"]].style.format(
        {"MAE (₹)": "{:,.0f}", "RMSE (₹)": "{:,.0f}", "R² before tuning": "{:.4f}", "R² after tuning": "{:.4f}",
         "MAPE (%)": "{:.2f}"}))
    st.caption("MAE is the average error in rupees, RMSE punishes large errors more, R² is the share of the price "
               "variation that the model explains (1.0 is perfect), MAPE is the average percentage error.")

    ba = am[["R² before tuning", "R² after tuning"]].reset_index().melt(
        id_vars="Model", var_name="Stage", value_name="R²")
    fig = px.bar(ba, x="Model", y="R²", color="Stage", barmode="group", range_y=[0.8, 0.95],
                 title="R² score before and after hyperparameter tuning",
                 color_discrete_sequence=["#2a9d8f", "#e76f51"])
    st.plotly_chart(fig)

    st.subheader("Cross-validation (5 folds, training data)")
    cv = load_csv("cross_validation_results.csv")
    st.dataframe(cv.style.format(
        {"CV R2": "{:.4f}", "CV R2 Std": "{:.4f}", "CV MAE": "{:,.0f}", "CV RMSE": "{:,.0f}"}))

    st.subheader(f"Actual vs predicted prices: {model_name}")
    tp = load_csv("test_predictions.csv", index_col=None)
    lim = [tp["actual_price"].min() / 1e5, tp["actual_price"].max() / 1e5]
    fig = px.scatter(x=tp["actual_price"] / 1e5, y=tp[model_name] / 1e5, opacity=0.5, log_x=True, log_y=True,
                     labels={"x": "Actual price (lakh INR, log scale)",
                             "y": "Predicted price (lakh INR, log scale)"},
                     title="Test cars: actual and predicted price")
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                  name="Perfect prediction", line=dict(color="red", dash="dash")))
    st.plotly_chart(fig)
    err = ((tp[model_name] - tp["actual_price"]) /
           tp["actual_price"] * 100).clip(-100, 100)
    fig = px.histogram(err, nbins=50, color_discrete_sequence=["#e9c46a"],
                       labels={"value": "Prediction error (% of the real price)"}, title="Distribution of the percentage error")
    fig.update_layout(showlegend=False, yaxis_title="Number of cars")
    st.plotly_chart(fig)

    st.subheader("Prediction ranges")
    pr = pd.DataFrame(info["prediction_range"]).T
    pr.index.name = "Model"
    pr.columns = ["Lower factor", "Upper factor",
                  "Share of test cars inside the range"]
    st.dataframe(pr.style.format({"Lower factor": "{:.3f}", "Upper factor": "{:.3f}",
                                  "Share of test cars inside the range": "{:.1%}"}))
    with st.expander("Best hyperparameters found by the tuning"):
        st.json(info["best_params"])

# ================================================================== PAGE 5: ABOUT
else:
    st.title("ℹ️ About This Application")
    st.markdown(f"""
**Devixo Solutions | AI/ML Internship Program | Task 04: End-to-End ML Application with Deployment**

**Author:** Aman Hameed Ullah

**Goal:** estimate the selling price of a used car from its details, and show how reliable the estimate is.

**Data:** the CarDekho used-car listings from Kaggle (8,128 cars; {len(data):,} remain after cleaning). Duplicates, test-drive
cars and impossible values were removed, text columns with units were converted to numbers, and five features were
engineered (car age, brand, owner number, kilometers per year and power per litre).

**Models:** Linear Regression, Decision Tree, Random Forest, Gradient Boosting and XGBoost were trained on the logarithm of the
price, compared with MAE, MSE, RMSE and R², and tuned with GridSearchCV and RandomizedSearchCV. The final model is the tuned
**{info['final_model']}**; the others can be selected in the sidebar.

**Prediction range:** a single number can be wrong, so the application shows a range that contains the real price for about
80% of the cars in the test data.

**Limitations:** the prices are Indian listings up to the year 2020, so they do not follow today's market; very expensive
cars are less reliable; the condition of the car, accidents and colour are not known.
""")
