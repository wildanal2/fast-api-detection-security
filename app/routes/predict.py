import pandas as pd
from fastapi import APIRouter, HTTPException
from app.models.form_req import FormReq
from app.utils.preprocess import preprocess_input, preprocess_data
from joblib import load
import logging

router = APIRouter()
  
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/predict-knn-label1")
async def prediction_label1(data: FormReq): 
    """
    API endpoint untuk melakukan prediksi Label 1, menggunakan KNN-scalled-no-canberra-k3 Accuracy: 0.959184. Example ini baris 1.
    
    Args:
        data (FormReq): 36 features.

    Returns:
        JSON: {status, message, prediction}.
    """
    try:
        # Konversi input POST request ke DataFrame
        df = pd.DataFrame([data.dict()])
        df = df.astype(object)  # Ubah semua tipe data ke Python-native
        
        # Logging input data
        logger.info(f"Received data: {df}")
        
        # Preprocess input
        preprocessed_input = preprocess_input(df)
        
        # Preprocess Data
        newdata = preprocess_data(preprocessed_input, "app/models/data/sources_v1")
        logger.info("Data preprocessing completed.")
        
        # Load the saved model
        model_path = "app/models/trained/label1-KNN-scalled-tfidf-no-canberra-k3-0.96.joblib"
        loaded_model = load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Make prediction
        predictions = loaded_model.predict(newdata)
        logger.info(f"Prediction result: {predictions}")
        
        # Return result
        return {
            "status": "success",
            "message": "Prediction completed successfully.",
            "prediction": predictions[0]
        }
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=500, detail="Model file or data sources are missing.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")

@router.post("/predict-knn-label2")
async def prediction_label2(data: FormReq): 
    """
    API endpoint untuk melakukan prediksi Label 2, menggunakan KNN-scalled-no-hamming-k3 Accuracy: 0.964286. Example ini baris 1.
    
    Args:
        data (FormReq): 36 features.

    Returns:
        JSON: {status, message, prediction}.
    """
    try:
        # Konversi input POST request ke DataFrame
        df = pd.DataFrame([data.dict()])
        df = df.astype(object)  # Ubah semua tipe data ke Python-native
        
        # Logging input data
        logger.info(f"Received data: {df}")
        
        # Preprocess input
        preprocessed_input = preprocess_input(df)
        
        # Preprocess Data
        newdata = preprocess_data(preprocessed_input, "app/models/data/sources_v1")
        logger.info("Data preprocessing completed.")
        
        # Load the saved model
        model_path = "app/models/trained/label2-KNN-scalled-tfidf-no-hamming-k3-0.964.joblib"
        loaded_model = load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Make prediction
        predictions = loaded_model.predict(newdata)
        logger.info(f"Prediction result: {predictions}")
        
        # Return result
        return {
            "status": "success",
            "message": "Prediction completed successfully.",
            "prediction": predictions[0]
        }
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=500, detail="Model file or data sources are missing.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")
