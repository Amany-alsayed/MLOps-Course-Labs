import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
from typing import Union
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(r"api\app.log"),
        logging.StreamHandler()
    ]
)
logger=logging.getLogger(__name__)
try:
    model=joblib.load(r"api\SVM_model.pkl")
    logger.info("model loaded successfully.")
except Exception as e:
    logger.error(f"failed to load model {e}")
    model=None

try:
    preprocess=joblib.load(r"api\column_transformer.pkl")
    logger.info("preprocessing file loaded successfully.")
except Exception as e:
    logger.error(f"failed to load preprosessing file {e}")
    preprocess=None

app=FastAPI()
class PredictRequest(BaseModel):
    features:list[Union[str, float, int]]


@app.get("/")
async def home():
    logger.info("Home endpoint accessed.")
    return {"message": "Welcome to the ML Model API"}


@app.get("/health")
async def health():
    if model:
        if preprocess:
            status="healthy"
        else:
            status="preprocessing file not loaded"      
    else:
        status="model not loaded"    
    logger.info(f"Health check accessed. Status: {status}")
    return {"status": status}


feature_columns = [
    "CreditScore","Geography","Gender","Age","Tenure","Balance",
    "NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"
    ]
@app.post("/predict")
async def predict(request:PredictRequest):
    if not model:
        logger.error("Prediction request received but model not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not preprocess:
        logger.error("Prediction request received but preprocessing file not loaded.")
        raise HTTPException(status_code=503, detail="preprocessing file not loaded")

    try:
      logger.info(f"Prediction request received: features={request.features}")
      input_df = pd.DataFrame([request.features], columns=feature_columns)
      processed = preprocess.transform(input_df) 
      logger.info(f"data after preprocessing phase: features={processed}")
      prediction=model.predict(processed)
      logger.info(f"Prediction Result: {prediction}")
      return {"prediction": prediction.tolist()}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail="Prediction failed")    



if __name__ == "__main__":
    uvicorn.run(app,port=8000)        




