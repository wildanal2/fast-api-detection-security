import pandas as pd
import re
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from app.utils.constants import SUSPICIOUS_KEYWORDS
import logging

def preprocess_catboost_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fungsi untuk melakukan preprocessing pada data input.
    Args:
        data (pd.DataFrame): Data input dalam bentuk DataFrame.
    Returns:
        pd.DataFrame: Data yang sudah di-preprocess default.
    """
    logging.info(f"Request body: {data}")
    # Mengisi nilai kosong dengan default
    # Konfigurasi Pandas untuk future behavior
    pd.set_option('future.no_silent_downcasting', True)
    # Step 1
    # Kolom yang secara defalut bernilai 0 jika kosong
    data['headers_upgrade_insecure_requests'] = data['headers_upgrade_insecure_requests'].fillna(0)
    data['headers_content_length'] = data['headers_content_length'].fillna(0)
    data['duration'] = data['duration'].fillna(0)
    data['remotePort'] = data['remotePort'].fillna(0)
    # Step 2
    # Mengisi kolom teks dengan string kosong jika tidak ada data 
    text_columns = ['path', 'method', 'httpVersion', 'headers_user_agent', 'headers_accept_encoding',
                'headers_accept', 'headers_x_requested_with', 'headers_content_type', 'headers_accept_language',
                'rawBody']
    for col in text_columns:
        data[col] = data[col].fillna("").infer_objects(copy=False)
    # Step 3
    # Mengisi kolom boolean dengan False jika kosong
    boolean_columns = ['is_abuser', 'is_attacker', 'is_bogon', 'is_cloud_provider', 'is_proxy',
                    'is_relay', 'is_tor', 'is_tor_exit', 'is_vpn', 'is_anonymous', 'is_threat', 'hosting']
    data[boolean_columns] = data[boolean_columns].fillna(False)
    # Step 4
    # Mengisi kolom kategorikal (misalnya, negara dan ISP) dengan nilai 'unknown' jika kosong
    categorical_columns = ['country', 'isp', 'org', 'protocol']
    data[categorical_columns] = data[categorical_columns].fillna('unknown')
    # Step 5
    # Memastikan seluruh data yang kosong diisi sesuai tipe kolom
    data.fillna('', inplace=True)  # Mengisi sisa kolom yang mungkin masih kosong dengan string kosong
    # Callback
    return data

def preprocess_catboost_data(new_data: pd.DataFrame, DUMP_DIR: str) -> pd.DataFrame:
    """
    Fungsi untuk melakukan preprocessing pada data untuk diolah.
    Args:
        data (pd.DataFrame): Data input dalam bentuk DataFrame.
    Returns:
        pd.DataFrame: Data yang sudah di-preprocess dan siap digunakan.
    """
    # Penyesuaian Nama feature
    new_data = new_data.rename(columns={'headers_user_agent':'headers.user-agent'})
    new_data = new_data.rename(columns={'headers_accept':'headers.accept'})
    new_data = new_data.rename(columns={'headers_accept_encoding':'headers.accept-encoding'})
    new_data = new_data.rename(columns={'headers_accept_language':'headers.accept-language'})
    new_data = new_data.rename(columns={'headers_upgrade_insecure_requests':'headers.upgrade-insecure-requests'})
    new_data = new_data.rename(columns={'headers_connection':'headers.connection'})
    new_data = new_data.rename(columns={'headers_x_requested_with':'headers.x-requested-with'})
    new_data = new_data.rename(columns={'headers_content_type':'headers.content-type'})
    new_data = new_data.rename(columns={'headers_content_length':'headers.content-length'})

    # Mengisi nilai kosong dengan default
    # Step 1
    # Load TF-IDF vectorizer yang telah disimpan
    tfidf_path = load(f"{DUMP_DIR}/tfidf_path.pkl")
    tfidf_user_agent = load(f"{DUMP_DIR}/tfidf_user-agent.pkl")
    tfidf_ip_v4 = load(f"{DUMP_DIR}/tfidf_ip_v4.pkl")
    tfidf_rawBody = load(f"{DUMP_DIR}/tfidf_rawBody.pkl")
    # Transformasikan 'path'
    tfidf_path_features = tfidf_path.transform(new_data['path']).toarray()
    df_path_tfidf_new_columns = ['path___' + feature for feature in tfidf_path.get_feature_names_out()]
    df_path_tfidf = pd.DataFrame(tfidf_path_features, columns=df_path_tfidf_new_columns)
    # Transformasikan 'headers.user-agent'
    tfidf_user_agent_features = tfidf_user_agent.transform(new_data['headers.user-agent']).toarray()
    df_user_agent_tfidf_new_columns = ['huser_agent___' + feature for feature in tfidf_user_agent.get_feature_names_out()]
    df_user_agent_tfidf = pd.DataFrame(tfidf_user_agent_features, columns=df_user_agent_tfidf_new_columns)
    # Transformasikan 'ip_v4'
    tfidf_ip_v4_features = tfidf_ip_v4.transform(new_data['ip_v4']).toarray()
    df_ip_v4_tfidf_new_columns = ['ip_v4___' + feature for feature in tfidf_ip_v4.get_feature_names_out()] # Buat list nama kolom baru dengan prefiks "ip_v4"
    df_ip_v4_tfidf = pd.DataFrame(tfidf_ip_v4_features, columns=df_ip_v4_tfidf_new_columns)
    # Transformasikan 'rawBody'
    tfidf_rawBody_features = tfidf_rawBody.transform(new_data['rawBody']).toarray()
    df_rawBody_tfidf_new_columns = ['rawBody___' + feature for feature in tfidf_rawBody.get_feature_names_out()]
    df_rawBody_tfidf = pd.DataFrame(tfidf_rawBody_features, columns=df_rawBody_tfidf_new_columns)
    # Gabungkan hasil TF-IDF ke dalam dataset asli
    new_data = pd.concat([new_data.drop(['path', 'headers.user-agent', 'ip_v4', 'rawBody'], axis=1), df_path_tfidf, df_user_agent_tfidf, df_ip_v4_tfidf, df_rawBody_tfidf], axis=1)

    # Step 2. Transformasi Kolom dengan Label Encoding
    label_columns = [
        'method', 'httpVersion', 'headers.accept', 'headers.accept-encoding',
        'remotePort', 'protocol', 'responseStatusCode', 'country', 'isp',
        'org', 'headers.accept-language', 'headers.upgrade-insecure-requests',
        'headers.connection', 'headers.x-requested-with', 'headers.content-type'
    ]
    for col in label_columns:
        # Load encoder dari file
        loaded_encoder = load(f"{DUMP_DIR}/label_ordinal_encoder_{col}.pkl")
        new_data[col] = loaded_encoder.transform(new_data[[col]]) # Changed from new_data[col] to new_data[[col]]

    # 3. Kolom Boolean yang Dikoversi ke Integer (0 dan 1)
    boolean_columns = [
        'mobile', 'proxy', 'hosting', 'is_abuser', 'is_attacker', 'is_bogon',
        'is_cloud_provider', 'is_proxy', 'is_relay', 'is_tor', 'is_tor_exit',
        'is_vpn', 'is_anonymous', 'is_threat'
    ]
    for col in boolean_columns:
        new_data[col] = new_data[col].astype(int)
    # 5. Kolom dengan Min-Max Scaling
    # Load scaler dari file
    scaler = load(f"{DUMP_DIR}/label_scaler_headers.content-length.pkl")
    new_data['headers.content-length'] = scaler.transform(new_data[['headers.content-length']])
    logging.info(f"New scaled data shape: {new_data.shape}")    
    # Callback
    return new_data

def prdiction_catboost(new_data: pd.DataFrame, DUMP_DIR: str) -> pd.DataFrame:
    # Load the saved model 
    loaded_model = load(DUMP_DIR)
    logging.info(f"Model loaded from {DUMP_DIR}")
    
    # Make prediction
    # Get the list of categorical features used during training
    categorical_features = loaded_model.get_param('cat_features') 

    # Ensure all categorical features in new_data are of type string
    for feature in categorical_features:
        new_data[feature] = new_data[feature].astype(str)

    # Predict the probabilities and label for the new data (assuming single row in new_data)
    probabilities = loaded_model.predict_proba(new_data)[0]  # Take the first (and only) row
    prediction = loaded_model.predict(new_data)[0]  # Take the first prediction

    # Get the confidence for the predicted label
    class_index = loaded_model.classes_.tolist().index(prediction)
    confidence = probabilities[class_index] * 100  # Convert to percentage
    
    logging.info(f"Prediction result: {prediction[0]}, Confidence: {confidence}")
    # Return as a dictionary
    return {
        "prediction": prediction[0],
        "confidence": confidence
    }