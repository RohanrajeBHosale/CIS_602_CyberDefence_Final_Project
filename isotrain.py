# train_model_isolation.py

import pandas as pd
import pickle
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def train_isolation_forest(csv_path='data/FIAL.csv', save_path='models/anomaly_model.pkl'):
    try:
        # Create models directory if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Load dataset
        logging.info("🔵 Loading dataset...")
        df = pd.read_csv(csv_path, encoding_errors='replace')
        df = df[['Protocol', 'Length', 'Source', 'Destination']].dropna()

        # Feature extraction
        logging.info("🔵 Extracting features...")
        df['Protocol_encoded'] = LabelEncoder().fit_transform(df['Protocol'].astype(str).str.strip())
        features = df[['Length', 'Protocol_encoded']]

        # Scaling features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # Isolation Forest model
        logging.info("🔵 Training Isolation Forest...")
        iso_model = IsolationForest(
            n_estimators=100,
            contamination=0.01,  # 1% anomalies
            random_state=42,
            n_jobs=-1
        )
        iso_model.fit(X_scaled)

        # Save model and preprocessing info
        protocol_encoder = LabelEncoder()
        protocol_encoder.fit(df['Protocol'].astype(str).str.strip())

        quantile_99 = df['Length'].quantile(0.99)

        model_data = {
            'model': iso_model,
            'scaler': scaler,
            'protocol_encoder': protocol_encoder,
            'quantile_99': quantile_99,
            'protocols': list(df['Protocol'].unique()),
            'known_ips': list(set(df['Source'].unique()) | set(df['Destination'].unique()))
        }

        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)

        logging.info(f"✅ Model trained and saved at: {save_path}")

    except Exception as e:
        logging.error(f"❌ Training failed: {e}")x

if __name__ == "__main__":
    train_isolation_forest(csv_path='data/FIAL.csv', save_path='models/anomaly_model.pkl')