import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from app.models.form_req import FormReq
from app.utils.preprocess_catboost_v1 import preprocess_catboost_input, preprocess_catboost_data, prdiction_catboost
from joblib import load 
import logging

logger = logging.getLogger(__name__)
SERVICE_DIR = f"app/services"

router = APIRouter()

@router.post("/predict-label1")
async def prediction_label1(data: FormReq): 
    """
    API endpoint untuk melakukan prediksi Label 1, menggunakan 'CatBoost Classifier', Accuracy: 0.9795918367346939. Example ini baris 1.
    
    Path & Raw body => TF-IDF

    ipv4 & headers.user-agent => TF-IDF

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
        preprocessed_input = preprocess_catboost_input(df)
        
        # Preprocess Data
        newdata = preprocess_catboost_data(preprocessed_input, f"{SERVICE_DIR}/data/sources_catboost1_09795918367346939")
        
        # Prediction
        result = prdiction_catboost(newdata, f"{SERVICE_DIR}/trained/label1-catboost-model-tfidf-0.9795918367346939.joblib")
        
        # Return result
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
    API endpoint untuk melakukan prediksi Label 2, menggunakan 'CatBoost Classifier', Accuracy: 0.9795918367346939. Example ini baris 1.
    
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
        preprocessed_input = preprocess_catboost_input(df)
        
        # Preprocess Data
        newdata = preprocess_catboost_data(preprocessed_input, f"{SERVICE_DIR}/data/sources_catboost1_09795918367346939")
        
        # Prediction
        result = prdiction_catboost(newdata, f"{SERVICE_DIR}/trained/label2-catboost_model-tfidf-0.9744897959183674.joblib")
        
        # Return result
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
    API endpoint untuk melakukan prediksi Label 3, menggunakan 'CatBoost Classifier', Accuracy: 1.0. Example ini baris 1.

    Args:
        data (FormReq): 36 features.

    Returns:
        JSON: {status, message, prediction [no,yes], confidence}.
    """
    try:
        # Konversi input POST request ke DataFrame
        df = pd.DataFrame([data.dict()])
        df = df.astype(object)  # Ubah semua tipe data ke Python-native
        
        # Logging input data
        logger.info(f"Received data: {df}")
        
        # Preprocess input
        preprocessed_input = preprocess_catboost_input(df)
        
        # Preprocess Data
        newdata = preprocess_catboost_data(preprocessed_input, f"{SERVICE_DIR}/data/sources_catboost3_10")
        logger.info("Data preprocessing completed.")
        
        # Prediction
        result = prdiction_catboost(newdata, f"{SERVICE_DIR}/trained/label3-catboost_model-tfidf-1.0.joblib")
        
        # Return result
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
    API endpoint untuk melakukan prediksi Label 4, menggunakan 'CatBoost Classifier', Accuracy: 0.9642857142857143. Example ini baris 1.

    Args:
        data (FormReq): 36 features.

    Returns:
        JSON: {status, message, prediction[high,medium,low]}.
    """
    try:
        # Konversi input POST request ke DataFrame
        df = pd.DataFrame([data.dict()])
        df = df.astype(object)  # Ubah semua tipe data ke Python-native
        
        # Logging input data
        logger.info(f"Received data: {df}")
        
        # Preprocess input
        preprocessed_input = preprocess_catboost_input(df)
        
        # Preprocess Data
        newdata = preprocess_catboost_data(preprocessed_input, "app/services/data/sources_catboost4_09642857142857143")
        logger.info("Data preprocessing completed.")
        
        # Load the saved model
        model_path = "app/services/trained/label4-catboost_model-tfidf-0.9642857142857143.joblib"
        loaded_model = load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Make prediction
        # Get the list of categorical features used during training
        categorical_features = loaded_model.get_param('cat_features') 

        # Ensure all categorical features in new_data are of type string
        for feature in categorical_features:
            newdata[feature] = newdata[feature].astype(str)

        # Predict the probabilities and label for the new data (assuming single row in newdata)
        probabilities = loaded_model.predict_proba(newdata)[0]  # Take the first (and only) row
        prediction = loaded_model.predict(newdata)[0]  # Take the first prediction

        # Get the confidence for the predicted label
        class_index = loaded_model.classes_.tolist().index(prediction)
        confidence = probabilities[class_index] * 100  # Convert to percentage
        
        # Return result
        return {
            "status": "success",
            "message": "Prediction Label 4 successfully.",
            "prediction": prediction[0],
            'confidence': confidence
        }
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=500, detail="Model file or data sources are missing.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")
