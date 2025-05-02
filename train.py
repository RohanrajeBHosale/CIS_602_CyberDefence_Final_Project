# train_models.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# === Create models folder if not exists ===
os.makedirs("models", exist_ok=True)

print("\n🔵 Step 1: Loading dataset...")
# === Load your dataset ===
df = pd.read_csv("Data/packets.csv", encoding_errors='replace')

print("✅ Loaded!")

# === Step 2: Feature Extraction ===
print("\n🔵 Step 2: Feature Extraction...")

df = df[['Protocol', 'Length', 'Source', 'Destination']].dropna()

df['protocol_num'] = pd.factorize(df['Protocol'])[0]
df['src_hash'] = df['Source'].apply(lambda x: hash(str(x)) % 1000)
df['dst_hash'] = df['Destination'].apply(lambda x: hash(str(x)) % 1000)

features = df[['protocol_num', 'Length', 'src_hash', 'dst_hash']]

print("✅ Features extracted!")

# === Step 3: Normalization ===
print("\n🔵 Step 3: Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Save the scaler
joblib.dump(scaler, "models/scaler_auto.pkl")
print("✅ Scaler saved!")

# === Step 4: Train Isolation Forest ===
print("\n🔵 Step 4: Training Isolation Forest...")

iso_model = IsolationForest(
    n_estimators=100,
    contamination=0.01,   # fine-tuned!
    random_state=42
)
iso_model.fit(X_scaled)

joblib.dump(iso_model, "models/model_iso.pkl")
print("✅ Isolation Forest model saved!")

# === Step 5: Train Autoencoder ===
print("\n🔵 Step 5: Training Autoencoder...")

input_dim = X_scaled.shape[1]

autoencoder = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

autoencoder.save("models/model_auto.h5")
print("✅ Autoencoder model saved!")

print("\n🎯 Training complete! All models are ready in the 'models/' folder.")