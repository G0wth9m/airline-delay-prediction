import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import argparse


def main(csv_path):
    # Load dataset
    df = pd.read_csv(csv_path)

    print("✅ Data loaded. Shape:", df.shape)
    print("📌 Columns:", df.columns.tolist())

    # --- Choose a target column ---
    # Adjust this depending on your dataset
    target_col = "carrier_delay"
   # Example: predict carrier delay
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")

    # --- Drop rows where target is missing ---
    df = df.dropna(subset=[target_col])

    # --- Select features ---
    # Remove non-numeric columns and the target
    X = df.select_dtypes(include=["number"]).drop(columns=[target_col], errors="ignore")
    y = df[target_col]

    print("✅ Features shape:", X.shape)
    print("✅ Target shape:", y.shape)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Train model ---
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Model Evaluation:")
    print("Mean Squared Error:", mse)
    print("R² Score:", r2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    args = parser.parse_args()
    main(args.csv)
