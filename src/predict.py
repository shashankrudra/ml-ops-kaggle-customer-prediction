import os
import pandas as pd

def create_submission(model, X_test, test, filename="outputs/submission.csv"):
    # Convert to an absolute path based on where this script is located
    abs_path = os.path.abspath(filename)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    y_test_pred = model.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": y_test_pred
    })
    submission.to_csv(filename, index=False)
    return submission