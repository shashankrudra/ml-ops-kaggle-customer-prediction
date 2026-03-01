
# Author: GitHub Copilot

from src.data_loader import load_data
from src.preprocessing import prepare_data, remove_id
from src.train import split_data, train_lightgbm, train_xgboost
from src.predict import create_submission
import mlflow
import mlflow.sklearn

def main():
    train, test = load_data("data/santander_train.csv", "data/santander_test.csv")
    X, y, X_test = prepare_data(train, test)
    X_train, X_val, y_train, y_val = split_data(X, y)
    X_train_clean = remove_id(X_train)
    X_val_clean = remove_id(X_val)
    X_test_clean = remove_id(X_test)

    # Log LightGBM model to MLflow
    with mlflow.start_run(run_name="LightGBM"):
        lgb_model, lgb_auc = train_lightgbm(X_train_clean, y_train, X_val_clean, y_val)
        print(f"AUC LightGBM on validation: {lgb_auc:.4f}")
        mlflow.log_metric("val_auc", lgb_auc)
        mlflow.sklearn.log_model(lgb_model, "model", registered_model_name="SantanderLightGBM")
        mlflow.log_artifact("outputs/submission.csv")
        mlflow.set_tag("model_type", "LightGBM")

    # Log XGBoost model to MLflow
    with mlflow.start_run(run_name="XGBoost"):
        xgb_model, xgb_auc = train_xgboost(X_train_clean, y_train, X_val_clean, y_val)
        print(f"AUC XGBoost on validation: {xgb_auc:.4f}")
        mlflow.log_metric("val_auc", xgb_auc)
        mlflow.sklearn.log_model(xgb_model, "model", registered_model_name="SantanderXGBoost")
        mlflow.log_artifact("outputs/submission.csv")
        mlflow.set_tag("model_type", "XGBoost")

    submission = create_submission(xgb_model, X_test_clean, test)
    print(submission.head())
    print(submission.shape)
    print(submission.columns)

if __name__ == "__main__":
    main()