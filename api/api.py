import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger=logging.getLogger(__name__)
try:
    model=joblib.load("SVM_model.pkl")
    logger.info("model loaded successfully.")
except Exception as e:
    logger.error(f"failed to load model {e}")
    model=None

app=FastAPI()
class PredictRequest(BaseModel):
    features:list[float]


@app.get("/")
async def home():
    logger.info("Home endpoint accessed.")
    return {"message": "Welcome to the ML Model API"}


@app.get("/health")
async def health():
    status="healthy" if model else "model not loaded"
    logger.info(f"Health check accessed. Status: {status}")
    return {"status": status}



@app.post("/predict")
async def predict(request:PredictRequest):
    if not model:
        logger.error("Prediction request received but model not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
      logger.info(f"Prediction request received: features={request.features}")
      prediction=model.predict([request.features])
      logger.info(f"Prediction Result: {prediction}")
      return {"prediction": prediction.tolist()}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail="Prediction failed")    



if __name__ == "__main__":
    uvicorn.run(app,port=8000)        




