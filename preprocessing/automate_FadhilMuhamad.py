import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import mlflow
import mlflow.sklearn

def preprocess_housing(df):
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']

    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

    skewed_features = ['total_rooms', 'total_bedrooms', 'population', 'households']
    for feature in skewed_features:
        df[feature] = np.log1p(df[feature])

    scaler = StandardScaler()
    numerical_features = [
        'longitude', 'latitude', 'housing_median_age', 'median_income',
        'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
    ] + skewed_features

    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df, scaler

df = pd.read_csv("CaliforniaHousing.csv")

df_preprocessed, scaler = preprocess_housing(df)
output_csv = os.path.join(os.path.dirname(__file__), "CaliforniaHousing_preprocessed.csv")
df_preprocessed.to_csv(output_csv, index=False)

X = df_preprocessed.drop("median_house_value", axis=1)
y = df_preprocessed["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.sklearn.autolog()

with mlflow.start_run():
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest MSE  : {mse:.4f}")
    print(f"Random Forest RMSE : {rmse:.4f}")
    print(f"Random Forest R2   : {r2:.4f}")

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_artifact(output_csv)