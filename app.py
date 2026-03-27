import os
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Import your project's preprocessing logic
# Ensure your 'src' folder is in the python path
from src.preprocessing import remove_id

class PredictionRequest(BaseModel):
    # This should match the raw feature list before remove_id
    # e.g., if you have 200 features + ID_code
    data: List[dict] 

class PredictionResponse(BaseModel):
    id_codes: List[str]
    probabilities: List[float]

app = FastAPI(title="Santander Prediction API")

# Global variable to hold the model
model = None

@app.on_event("startup")
def load_registered_model():
    """
    Loads the XGBoost model from MLflow Registry on startup.
    """
    global model
    try:
        # Pulls the latest version of the model you registered in main.py
        model_name = "SantanderXGBoost"
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Successfully loaded {model_name} from MLflow.")
    except Exception as e:
        print(f"Failed to load model from MLflow: {e}")
        # Fallback to local if MLflow server is down
        # model = joblib.load("outputs/model.joblib")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # 1. Convert incoming JSON to DataFrame
        df = pd.DataFrame(request.data)
        id_codes = df["ID_code"].tolist()
        
        # 2. Apply the exact same preprocessing used in main.py
        # This ensures the 'ID_code' is stripped before hitting the model
        df_clean = remove_id(df)
        
        # 3. Predict probabilities (column 1 is usually the 'target' class)
        probs = model.predict_proba(df_clean)[:, 1]
        
        return {
            "id_codes": id_codes,
            "probabilities": probs.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ready", "model": "SantanderXGBoost"}
