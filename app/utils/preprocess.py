import pandas as pd
import re
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from app.utils.constants import SUSPICIOUS_KEYWORDS

def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fungsi untuk melakukan preprocessing pada data input.
    Args:
        data (pd.DataFrame): Data input dalam bentuk DataFrame.
    Returns:
        pd.DataFrame: Data yang sudah di-preprocess default.
    """
    # Mengisi nilai kosong dengan default
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
        data[col] = data[col].fillna("")
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

def preprocess_data(new_data: pd.DataFrame, DUMP_DIR: str) -> pd.DataFrame:
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
    # Buat CountVectorizer
    vectorizer = CountVectorizer(vocabulary=SUSPICIOUS_KEYWORDS)
    # 1 Custom bagwords rawBody
    new_data['rawBody'] = new_data['rawBody'].apply(preprocess_text)
    f_body_features = vectorizer.transform(new_data['rawBody'])
    f_body_df = pd.DataFrame(f_body_features.toarray(), columns=vectorizer.get_feature_names_out())
    new_data['rawBody'] = f_body_df.sum(axis=1)
    # 1 Custom bagwords Path
    new_data['path'] = new_data['path'].apply(preprocess_text)
    f_path_features = vectorizer.transform(new_data['path'])
    f_path_df = pd.DataFrame(f_path_features.toarray(), columns=vectorizer.get_feature_names_out())
    new_data['path'] = f_path_df.sum(axis=1)
    # Step 2 Kolom dengan TF-IDF Vectorization
    # Load TF-IDF vectorizer yang telah disimpan
    tfidf_user_agent = load(f"{DUMP_DIR}/tfidf_user_agent.pkl")
    tfidf_ip_v4 = load(f"{DUMP_DIR}/tfidf_ip_v4.pkl")

    # Transformasikan 'headers.user-agent'
    tfidf_user_agent_features = tfidf_user_agent.transform(new_data['headers.user-agent']).toarray()
    df_user_agent_tfidf_new_columns = ['user_agent_' + feature for feature in tfidf_user_agent.get_feature_names_out()]
    df_user_agent_tfidf = pd.DataFrame(tfidf_user_agent_features, columns=df_user_agent_tfidf_new_columns)
    # Transformasikan 'ip_v4'
    tfidf_ip_v4_features = tfidf_ip_v4.transform(new_data['ip_v4']).toarray()
    df_ip_v4_tfidf_new_columns = ['ip_v4_' + feature for feature in tfidf_ip_v4.get_feature_names_out()] # Buat list nama kolom baru dengan prefiks "ip_v4"
    df_ip_v4_tfidf = pd.DataFrame(tfidf_ip_v4_features, columns=df_ip_v4_tfidf_new_columns)

    # Gabungkan hasil TF-IDF ke dalam dataset asli
    new_data = pd.concat([new_data.drop(['headers.user-agent', 'ip_v4'], axis=1), df_user_agent_tfidf, df_ip_v4_tfidf], axis=1)

    # Step 3. Transformasi Kolom dengan Label Encoding
    label_columns = [
        'method', 'httpVersion', 'headers.accept', 'headers.accept-encoding',
        'remotePort', 'protocol', 'responseStatusCode', 'country', 'isp',
        'org', 'headers.accept-language', 'headers.upgrade-insecure-requests',
        'headers.connection', 'headers.x-requested-with', 'headers.content-type'
    ]
    for col in label_columns:
        # Load encoder dari file
        loaded_encoder = load(f"{DUMP_DIR}/label_encoder_{col}.pkl")
        new_data[col] = loaded_encoder.transform(new_data[col])
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
    # 6. Scale the new data using the same scaler
    # Load scaler dari file
    scaler = load(f"{DUMP_DIR}/scaler.pkl")
    new_scaled = scaler.transform(new_data)
    newtest_scaled_df = pd.DataFrame(new_scaled) # Use the original column names from X_train
    # Callback
    return newtest_scaled_df

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text