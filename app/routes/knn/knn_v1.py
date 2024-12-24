import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from app.models.form_req import FormReq
from app.utils.preprocess import preprocess_input, preprocess_data
from joblib import load
import logging

router = APIRouter()
  
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Logging input data
        logger.info(f"Received data: {df}")
        
        # Preprocess input
        preprocessed_input = preprocess_input(df)
        
        # Preprocess Data
        newdata = preprocess_data(preprocessed_input, "app/services/data/sources_knn1_v09795918367346939")
        logger.info("Data preprocessing completed.")
        
        # Load the saved model
        model_path = "app/services/trained/label1-KNN-scaled-tfidf-no-hamming-k4-0.9795918367346939.joblib"
        loaded_model = load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Make prediction
        predictions = loaded_model.predict(newdata)
        confidence = loaded_model.predict_proba(newdata)
        # Dapatkan indeks numerik dari prediksi
        predicted_class = predictions[0]
        class_index = np.where(loaded_model.classes_ == predicted_class)[0][0] 
        confidence = confidence[0][class_index] * 100
        
        logger.info(f"Prediction result: {predicted_class}, Confidence: {confidence}")
        
        # Return result
        return {
            "status": "success",
            "message": "Prediction Label 1 successfully.",
            "prediction": predicted_class,
            'confidence': confidence
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
        
        # Logging input data
        logger.info(f"Received data: {df}")
        
        # Preprocess input
        preprocessed_input = preprocess_input(df)
        
        # Preprocess Data
        newdata = preprocess_data(preprocessed_input, "app/services/data/sources_knn2_v09642857142857143")
        logger.info("Data preprocessing completed.")
        
        # Load the saved model
        model_path = "app/services/trained/label2-KNN-scaled-tfidf-no-hamming-k3-0.9642857142857143.joblib"
        loaded_model = load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Make prediction
        predictions = loaded_model.predict(newdata)
        confidence = loaded_model.predict_proba(newdata)
        logger.info(f"Prediction result: {predictions}, Confidence: {confidence}")
        # Dapatkan indeks numerik dari prediksi
        predicted_class = predictions[0]
        class_index = np.where(loaded_model.classes_ == predicted_class)[0][0] 
        confidence = confidence[0][class_index] * 100
        
        logger.info(f"Prediction result: {predicted_class}, Confidence: {confidence}")
        # Return result
        return {
            "status": "success",
            "message": "Prediction Label 2 successfully.",
            "prediction": predictions[0],
            'confidence': confidence
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
        
        # Logging input data
        logger.info(f"Received data: {df}")
        
        # Preprocess input
        preprocessed_input = preprocess_input(df)
        
        # Preprocess Data
        newdata = preprocess_data(preprocessed_input, "app/services/data/sources_knn3_v09948979591836735")
        logger.info("Data preprocessing completed.")
        
        # Load the saved model
        model_path = "app/services/trained/label3-KNN-scaled-tfidf-no-hamming-k4-0.9948979591836735.joblib"
        loaded_model = load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Make prediction
        predictions = loaded_model.predict(newdata)
        confidence = loaded_model.predict_proba(newdata)
        logger.info(f"Prediction result: {predictions}, Confidence: {confidence}")
        # Dapatkan indeks numerik dari prediksi
        predicted_class = predictions[0]
        class_index = np.where(loaded_model.classes_ == predicted_class)[0][0] 
        confidence = confidence[0][class_index] * 100
        
        logger.info(f"Prediction result: {predicted_class}, Confidence: {confidence}")
        # Return result
        return {
            "status": "success",
            "message": "Prediction Label 3 successfully.",
            "prediction": predictions[0],
            'confidence': confidence
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
        
        # Logging input data
        logger.info(f"Received data: {df}")
        
        # Preprocess input
        preprocessed_input = preprocess_input(df)
        
        # Preprocess Data
        newdata = preprocess_data(preprocessed_input, "app/services/data/sources_knn4_v09693877551020408")
        logger.info("Data preprocessing completed.")
        
        # Load the saved model
        model_path = "app/services/trained/label4-KNN-scaled-tfidf-no-cosine-k3-0.9693877551020408.joblib"
        loaded_model = load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Make prediction
        predictions = loaded_model.predict(newdata)
        confidence = loaded_model.predict_proba(newdata)
        logger.info(f"Prediction result: {predictions}, Confidence: {confidence}")
        # Dapatkan indeks numerik dari prediksi
        predicted_class = predictions[0]
        class_index = np.where(loaded_model.classes_ == predicted_class)[0][0] 
        confidence = confidence[0][class_index] * 100
        
        logger.info(f"Prediction result: {predicted_class}, Confidence: {confidence}")
        # Return result
        return {
            "status": "success",
            "message": "Prediction Label 4 successfully.",
            "prediction": predictions[0],
            'confidence': confidence
        }
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=500, detail="Model file or data sources are missing.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")
