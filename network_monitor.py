import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from scapy.all import AsyncSniffer, IP, UDP, ARP, TCP
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from threading import Thread

warnings.filterwarnings('ignore')

class NetworkMonitor:
    def __init__(self, model_choice):
        self.model_choice = model_choice
        self.model = None
        self.scaler = StandardScaler()
        self.protocol_map = {}
        self.sniffer = None
        self.end_time = 0
        self.anomaly_scores = []
        self.stats = {
            'total_packets': 0,
            'protocols': defaultdict(int),
            'sources': defaultdict(int)
        }
        self.syn_counter = defaultdict(int)
        self.live_plotting = False
        self.anomaly_score_buffer = []
        self.packet_rate_buffer = []
        self.last_packet_time = time.time()

        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

    def _load_data(self, data_path, sample_frac=0.05):
        df = pd.read_csv(data_path, encoding='latin1', on_bad_lines='skip')
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
            print(f"Sampled {sample_frac*100:.1f}% of total data -> {len(df)} rows")
        required_cols = ['Time', 'Source', 'Destination', 'Protocol', 'Length']
        return df[required_cols].dropna()

    def _extract_features(self, df):
        self.protocol_map = {proto: idx for idx, proto in enumerate(df['Protocol'].unique())}
        df['Time'] = pd.to_datetime(df['Time'])
        df['time_diff'] = df['Time'].diff().dt.total_seconds().fillna(0)
        return pd.DataFrame({
            'protocol': df['Protocol'].map(self.protocol_map),
            'length': df['Length'],
            'time_diff': df['time_diff'],
            'hour': df['Time'].dt.hour,
            'SYN': 0,
            'ACK': 0,
            'FIN': 0,
            'RST': 0,
            'port': 0
        })

    def train_models(self, data_path="data/FIAL.csv", sample_frac=0.05):
        print("Training model(s)...")
        df = self._load_data(data_path, sample_frac=sample_frac)
        features = self._extract_features(df)
        X = self.scaler.fit_transform(features)

        if self.model_choice in ["IF", "ALL"]:
            print("Training Isolation Forest...")
            model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, verbose=1)
            model.fit(X)
            joblib.dump(model, "models/model_iso.pkl")
            joblib.dump(self.scaler, "models/scaler_iso.pkl")
            print("Isolation Forest trained and saved.")

        if self.model_choice in ["OCSVM", "ALL"]:
            print("Training One-Class SVM...")
            model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
            model.fit(X)
            joblib.dump(model, "models/model_ocsvm.pkl")
            joblib.dump(self.scaler, "models/scaler_ocsvm.pkl")
            print("One-Class SVM trained and saved.")

        if self.model_choice in ["AE", "ALL"]:
            print("Training Autoencoder...")
            input_dim = X.shape[1]
            autoencoder = Sequential([
                Dense(32, activation='relu', input_shape=(input_dim,)),
                Dense(16, activation='relu'),
                Dense(32, activation='relu'),
                Dense(input_dim, activation='linear')
            ])
            autoencoder.compile(optimizer=Adam(), loss='mse')
            autoencoder.fit(X, X, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
            autoencoder.save("models/model_auto.h5")
            joblib.dump(self.scaler, "models/scaler_auto.pkl")
            print("Autoencoder trained and saved.")

    def load_model_for_monitoring(self):
        if self.model_choice == "IF":
            self.model = joblib.load("models/model_iso.pkl")
            self.scaler = joblib.load("models/scaler_iso.pkl")
        elif self.model_choice == "OCSVM":
            self.model = joblib.load("models/model_ocsvm.pkl")
            self.scaler = joblib.load("models/scaler_ocsvm.pkl")
        elif self.model_choice == "AE":
            self.model = load_model("models/model_auto.h5", compile=False)
            self.scaler = joblib.load("models/scaler_auto.pkl")
        print("Model loaded successfully!")

    def monitor_network(self, minutes=5):
        print(f"\nStarting network monitoring for {minutes} minutes...")
        self.end_time = time.time() + minutes * 60
        self.sniffer = AsyncSniffer(prn=self.packet_handler, store=False)
        self.sniffer.start()

        while time.time() < self.end_time:
            time.sleep(1)

        self.sniffer.stop()
        self._generate_final_report()

    def packet_handler(self, packet):
        if time.time() > self.end_time:
            return
        try:
            packet_info = self._process_packet(packet)
            self._update_stats(packet_info)
            if self._detect_anomaly(packet_info):
                self._display_alert(packet_info)

            now = time.time()
            elapsed = now - self.last_packet_time
            self.last_packet_time = now

            if elapsed > 0:
                self.packet_rate_buffer.append(1 / elapsed)

            if len(self.anomaly_score_buffer) > 100:
                self.anomaly_score_buffer.pop(0)

            if self.anomaly_scores:
                self.anomaly_score_buffer.append(self.anomaly_scores[-1])

        except Exception:
            pass

    def _process_packet(self, packet):
        proto = "Unknown"
        src = "Unknown"
        dst = "Unknown"
        port = 0
        flags = {'SYN': 0, 'ACK': 0, 'FIN': 0, 'RST': 0}

        if IP in packet:
            src = packet[IP].src
            dst = packet[IP].dst
            if TCP in packet:
                proto = "TCP"
                tcp_flags = packet[TCP].flags
                flags['SYN'] = int(bool(tcp_flags & 0x02))
                flags['ACK'] = int(bool(tcp_flags & 0x10))
                flags['FIN'] = int(bool(tcp_flags & 0x01))
                flags['RST'] = int(bool(tcp_flags & 0x04))
                port = packet[TCP].dport
            elif UDP in packet:
                proto = "UDP"
                port = packet[UDP].dport
            else:
                proto = "IP"
        elif ARP in packet:
            src = packet[ARP].psrc
            dst = packet[ARP].pdst
            proto = "ARP"

        return {
            'time': datetime.now().strftime("%H:%M:%S"),
            'src': src,
            'dst': dst,
            'proto': proto,
            'size': len(packet),
            'port': port,
            **flags
        }

    def _update_stats(self, packet):
        self.stats['total_packets'] += 1
        self.stats['protocols'][packet['proto']] += 1
        self.stats['sources'][packet['src']] += 1

        if packet.get('SYN', 0) == 1 and packet.get('ACK', 0) == 0:
            self.syn_counter[packet['src']] += 1

    def _detect_anomaly(self, packet):
        if self.syn_counter[packet['src']] > 100:
            print(f"\nWARNING: Potential SYN Flood detected from {packet['src']} ({self.syn_counter[packet['src']]} SYN packets!)")
            print("-" * 60)

        features = {
            'protocol': self.protocol_map.get(packet['proto'], -1),
            'length': packet['size'],
            'time_diff': 0,
            'hour': datetime.now().hour,
            'SYN': packet.get('SYN', 0),
            'ACK': packet.get('ACK', 0),
            'FIN': packet.get('FIN', 0),
            'RST': packet.get('RST', 0),
            'port': packet.get('port', 0)
        }
        feature_values = [[features[k] for k in features]]
        X = self.scaler.transform(feature_values)

        if self.model_choice == "AE":
            reconstructed = self.model.predict(X)
            mse = np.mean(np.power(X - reconstructed, 2))
            score = -mse
        else:
            score = self.model.decision_function(X)[0]

        self.anomaly_scores.append(score)
        packet['score'] = score
        return score < -110

    def _display_alert(self, packet):
        print(f"\nALERT: Anomalous packet detected")
        print(f"Time: {packet['time']} | Source: {packet['src']} | Destination: {packet['dst']} | Protocol: {packet['proto']}")
        print(f"Size: {packet['size']} bytes | Score: {packet['score']:.2f}")
        print("-" * 60)

    def _generate_final_report(self):
        print("\n=== MONITORING REPORT ===")
        print("Total Packets:", self.stats['total_packets'])

    def start_live_dashboard(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        def animate(i):
            ax1.clear()
            ax2.clear()

            if self.anomaly_score_buffer:
                ax1.plot(self.anomaly_score_buffer)
                ax1.axhline(y=-0.3, color='r', linestyle='--')
                ax1.set_title("Live Anomaly Score")

            if self.packet_rate_buffer:
                ax2.plot(self.packet_rate_buffer[-100:], color='orange')
                ax2.set_title("Live Packet Rate (Packets/sec)")

            plt.tight_layout()

        ani = animation.FuncAnimation(fig, animate, interval=1000)
        plt.show()

if __name__ == "__main__":
    monitor = NetworkMonitor(model_choice="AE")
    monitor.load_model_for_monitoring()

    t1 = Thread(target=monitor.monitor_network, args=(5,))
    t2 = Thread(target=monitor.start_live_dashboard)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print("✅ Monitoring session complete.")
