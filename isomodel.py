import pandas as pd
import numpy as np
import pickle
import time
import os
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from scapy.all import sniff, IP, Ether, ARP, UDP, TCP
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler('anomaly_detection.log'),
        logging.StreamHandler()
    ]
)

class IsolationForestAnomalyDetector:
    def __init__(self, model_path='models/anomaly_model.pkl', window_size=1000):
        self.model_path = Path(model_path)
        self.scaler = StandardScaler()
        self.protocol_encoder = LabelEncoder()
        self.model = None
        self.quantile_99 = None
        self.packet_window = deque(maxlen=window_size)
        self.protocols = set()
        self.known_ips = set()
        self.total_packets = 0
        self.anomaly_count = 0
        self.anomalies_list = []  # store anomaly details
        self.anomaly_reasons_counter = {}

        self._load_model()

    def _load_model(self):
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.scaler = data['scaler']
                    self.protocol_encoder = data['protocol_encoder']
                    self.quantile_99 = data['quantile_99']
                    self.protocols = set(data.get('protocols', []))
                    self.known_ips = set(data.get('known_ips', []))
                logging.info("✅ Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def _extract_packet_info(self, packet):
        packet_info = {
            'Time': time.time(),
            'Source': 'Unknown',
            'Destination': 'Unknown',
            'Protocol': 'Unknown',
            'Length': len(packet)
        }

        try:
            if Ether in packet:
                if packet[Ether].dst == 'ff:ff:ff:ff:ff:ff':
                    packet_info['Destination'] = 'Broadcast'

            if IP in packet:
                packet_info['Source'] = packet[IP].src
                packet_info['Destination'] = packet[IP].dst

                if TCP in packet:
                    packet_info['Protocol'] = 'TCP'
                elif UDP in packet:
                    packet_info['Protocol'] = 'UDP'
                elif packet[IP].proto == 1:
                    packet_info['Protocol'] = 'ICMP'
                else:
                    packet_info['Protocol'] = str(packet[IP].proto)

            elif ARP in packet:
                packet_info['Protocol'] = 'ARP'
                packet_info['Source'] = packet[ARP].psrc
                packet_info['Destination'] = packet[ARP].pdst

        except Exception as e:
            logging.debug(f"Packet parsing error: {e}")

        return packet_info

    def process_packet(self, packet):
        self.total_packets += 1

        try:
            packet_info = self._extract_packet_info(packet)
            self.packet_window.append(packet_info)

            is_anomaly, reasons = self._detect(packet_info)

            if is_anomaly:
                self.anomaly_count += 1
                self._alert(packet_info, reasons)

            if self.total_packets % 1000 == 0:
                self._log_status()

        except Exception as e:
            logging.error(f"Packet processing error: {e}")

    def _detect(self, packet_info):
        reasons = []

        # ML Isolation Forest
        if self.model:
            try:
                proto = packet_info['Protocol']
                protocol_encoded = (
                    self.protocol_encoder.transform([proto])[0]
                    if proto in self.protocols else len(self.protocols)
                )
                features = [[packet_info['Length'], protocol_encoded]]
                scaled = self.scaler.transform(features)
                prediction = self.model.predict(scaled)

                if prediction[0] == -1:
                    reasons.append("ML_ISOLATION_FOREST")

            except Exception as e:
                logging.debug(f"ML prediction error: {e}")

        # Rule-based: Large Packet
        if self.quantile_99 and packet_info['Length'] > self.quantile_99:
            reasons.append(f"LARGE_PACKET({packet_info['Length']} bytes)")

        # Rule-based: Broadcast Storm
        if packet_info['Destination'] == 'Broadcast':
            recent_broadcasts = sum(
                1 for p in self.packet_window
                if p['Destination'] == 'Broadcast' and p['Time'] > packet_info['Time'] - 1.0
            )
            if recent_broadcasts > 100:
                reasons.append("BROADCAST_STORM")

        # 4. Rule: Too many ARPs within short time
        if packet_info['Protocol'] == 'ARP':
            recent_arps = sum(
                1 for p in self.packet_window
                if p['Protocol'] == 'ARP' and p['Time'] > packet_info['Time'] - 10.0
            )
            if recent_arps > 10:  # Threshold (lower for testing)
                reasons.append("ARP_STORM")

        return (len(reasons) > 0), reasons

    def _alert(self, packet_info, reasons):
        alert = [
            "\n🚨 === NETWORK ANOMALY DETECTED ===",
            f"Time: {time.ctime(packet_info['Time'])}",
            f"Source: {packet_info['Source']}",
            f"Destination: {packet_info['Destination']}",
            f"Protocol: {packet_info['Protocol']}",
            f"Length: {packet_info['Length']} bytes",
            f"Triggered: {', '.join(reasons)}",
            "=" * 40
        ]
        logging.warning("\n".join(alert))

        anomaly_entry = {
            'Time': time.ctime(packet_info['Time']),
            'Source': packet_info['Source'],
            'Destination': packet_info['Destination'],
            'Protocol': packet_info['Protocol'],
            'Length': packet_info['Length'],
            'Reasons': reasons
        }
        self.anomalies_list.append(anomaly_entry)

        for r in reasons:
            self.anomaly_reasons_counter[r] = self.anomaly_reasons_counter.get(r, 0) + 1

    def _log_status(self):
        logging.info(
            f"Processed {self.total_packets} packets | "
            f"Anomalies detected: {self.anomaly_count} | "
            f"Window: {len(self.packet_window)} packets"
        )

        if self.anomalies_list:
            self._save_summary()

    def _save_summary(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs('reports', exist_ok=True)

        # Save CSV
        df = pd.DataFrame(self.anomalies_list)
        csv_path = f'reports/anomalies_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        logging.info(f"📄 Anomaly summary saved to {csv_path}")

        # Bar graph: Packets vs Anomalies
        plt.figure(figsize=(10,6))
        plt.bar(['Total Packets', 'Anomalies'], [self.total_packets, self.anomaly_count], color=['blue', 'red'])
        plt.title('Traffic Summary')
        plt.ylabel('Count')
        plt.savefig(f'reports/traffic_summary_{timestamp}.png')
        plt.close()

        # Pie chart: Anomaly reasons
        if self.anomaly_reasons_counter:
            plt.figure(figsize=(8,8))
            plt.pie(
                list(self.anomaly_reasons_counter.values()),
                labels=list(self.anomaly_reasons_counter.keys()),
                autopct='%1.1f%%',
                startangle=140
            )
            plt.title('Anomaly Reasons Distribution')
            plt.savefig(f'reports/anomaly_reasons_{timestamp}.png')
            plt.close()

            logging.info(f"📈 Graphs saved in reports/ folder")

    def start_live_capture(self, interface="en0", timeout=300):
        """Start capturing with proper fixes"""
        logging.info(f"Starting live capture on interface {interface or 'default'}...")

        try:
            sniff(
                iface=interface,
                prn=self._process_packet_with_log,  # modified function
                timeout=timeout,
                store=False
            )
        except KeyboardInterrupt:
            logging.info("🔵 Stopped manually.")
        except Exception as e:
            logging.error(f"Sniffer error: {e}")
        finally:
            self._log_status()

    def _process_packet_with_log(self, packet):
        """Modified packet processing to log live"""
        packet_info = self._extract_packet_info(packet)
        print(f"[Captured] {packet_info}")  # ✅ live print

        self.total_packets += 1
        self.packet_window.append(packet_info)

        is_anomaly, reasons = self._detect(packet_info)

        if is_anomaly:
            self.anomaly_count += 1
            self._alert(packet_info, reasons)

        # Light logging every 100 packets
        if self.total_packets % 100 == 0:
            logging.info(f"Processed {self.total_packets} packets so far.")

if __name__ == "__main__":
    detector = IsolationForestAnomalyDetector()
    detector.start_live_capture(timeout=300)