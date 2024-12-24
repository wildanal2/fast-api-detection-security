import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from app.models.form_req import FormReq
from app.utils.preprocess_knn_v1 import preprocess_knn_input, preprocess_knn_data, predict_knn
from joblib import load
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
SERVICE_DIR = f"app/services"

@router.post("/predict-label1")
async def prediction_label1(data: FormReq): 
    """
    API endpoint untuk melakukan prediksi Label 1, menggunakan KNN-scalled-no-hamming-k4 Accuracy: 0.9795918367346939. Example ini baris 1.
    
    Path & Raw body => only use Bag of Words CountVectorizer

    ipv4 & headers.user-agent => tf-idf

    Label Encoding with `OrdinalEncoder` = [
        'method', 'httpVersion', 'headers.accept', 'headers.accept-encoding',
        'remotePort', 'protocol', 'responseStatusCode', 'country', 'isp',
        'org', 'headers.accept-language', 'headers.upgrade-insecure-requests',
        'headers.connection', 'headers.x-requested-with', 'headers.content-type'
    ]

    boolean_columns = [
        'mobile', 'proxy', 'hosting', 'is_abuser', 'is_attacker', 'is_bogon',
        'is_cloud_provider', 'is_proxy', 'is_relay', 'is_tor', 'is_tor_exit',
        'is_vpn', 'is_anonymous', 'is_threat'
    ]

    Args:
        data (FormReq): 36 features.

    Returns:
        JSON: {status, message, prediction [high,medium,low], confidence}.
    """
    try:
        # Konversi input POST request ke DataFrame
        df = pd.DataFrame([data.dict()])
        df = df.astype(object)  # Ubah semua tipe data ke Python-native
        
        # Preprocess input
        preprocessed_input = preprocess_knn_input(df)

        # Preprocess Data
        newdata = preprocess_knn_data(preprocessed_input, f"{SERVICE_DIR}/data/sources_knn1_v09795918367346939")

        # Prediction
        result = predict_knn(newdata, f"{SERVICE_DIR}/trained/label1-KNN-scaled-tfidf-no-hamming-k4-0.9795918367346939.joblib")

        return {
            "status": "success",
            "message": "Prediction Label 1 successfully.",
            "prediction": result["prediction"],
            'confidence': result["confidence"]
        }
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=500, detail="Model file or data sources are missing.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")

@router.post("/predict-label2")
async def prediction_label2(data: FormReq): 
    """
    API endpoint untuk melakukan prediksi Label 2, menggunakan KNN-scalled-no-hamming-k3 Accuracy: 0.9642857142857143. Example ini baris 1.
    
    Args:
        data (FormReq): 36 features.

    Returns:
        JSON: {status, message, prediction [high,medium,low], confidence}.
    """
    try:
        # Konversi input POST request ke DataFrame
        df = pd.DataFrame([data.dict()])
        df = df.astype(object)  # Ubah semua tipe data ke Python-native
        
        # Preprocess input
        preprocessed_input = preprocess_knn_input(df)
        
        # Preprocess Data
        newdata = preprocess_knn_data(preprocessed_input, f"{SERVICE_DIR}/data/sources_knn2_v09642857142857143") 
        
        # Prediction
        result = predict_knn(newdata, f"{SERVICE_DIR}/trained/label2-KNN-scaled-tfidf-no-hamming-k3-0.9642857142857143.joblib")

        return {
            "status": "success",
            "message": "Prediction Label 2 successfully.",
            "prediction": result["prediction"],
            'confidence': result["confidence"]
        }
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=500, detail="Model file or data sources are missing.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")

@router.post("/predict-label3")
async def prediction_label3(data: FormReq): 
    """
    API endpoint untuk melakukan prediksi Label 3, menggunakan KNN-scalled-no-hamming-k4 Accuracy: 0.9948979591836735. Example ini baris 1.

    Args:
        data (FormReq): 36 features.

    Returns:
        JSON: {status, message, prediction [no,yes], confidence}.
    """
    try:
        # Konversi input POST request ke DataFrame
        df = pd.DataFrame([data.dict()])
        df = df.astype(object)  # Ubah semua tipe data ke Python-native
         
        # Preprocess input
        preprocessed_input = preprocess_knn_input(df)
        
        # Preprocess Data
        newdata = preprocess_knn_data(preprocessed_input, f"{SERVICE_DIR}/data/sources_knn3_v09948979591836735")
        
        # Prediction
        result = predict_knn(newdata, f"{SERVICE_DIR}/trained/label3-KNN-scaled-tfidf-no-hamming-k4-0.9948979591836735.joblib")

        return {
            "status": "success",
            "message": "Prediction Label 3 successfully.",
            "prediction": result["prediction"],
            'confidence': result["confidence"]
        }
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=500, detail="Model file or data sources are missing.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")

@router.post("/predict-label4")
async def prediction_label4(data: FormReq): 
    """
    API endpoint untuk melakukan prediksi Label 4, menggunakan KNN-scalled-no-hamming-k4 Accuracy: 0.9948979591836735. Example ini baris 1.

    Args:
        data (FormReq): 36 features.

    Returns:
        JSON: {status, message, prediction[high,medium,low]}.
    """
    try:
        # Konversi input POST request ke DataFrame
        df = pd.DataFrame([data.dict()])
        df = df.astype(object)  # Ubah semua tipe data ke Python-native
        
        # Preprocess input
        preprocessed_input = preprocess_knn_input(df)
        
        # Preprocess Data
        newdata = preprocess_knn_data(preprocessed_input, f"{SERVICE_DIR}/data/sources_knn4_v09693877551020408")

        # Prediction
        result = predict_knn(newdata, f"{SERVICE_DIR}/trained/label4-KNN-scaled-tfidf-no-cosine-k3-0.9693877551020408.joblib")

        return {
            "status": "success",
            "message": "Prediction Label 4 successfully.",
            "prediction": result["prediction"],
            'confidence': result["confidence"]
        }
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=500, detail="Model file or data sources are missing.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")
