import xgboost as xgb
import numpy as np

# Create a tiny dummy dataset
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

# Try to initialize with all cores
model = xgb.XGBClassifier(n_jobs=-1) 
model.fit(X, y)
print("XGBoost is running successfully with multi-threading.")