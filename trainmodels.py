import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from datetime import datetime
from collections import defaultdict
from scapy.all import AsyncSniffer, IP, UDP, ARP, TCP
from tensorflow.keras.models import load_model
from threading import Thread

warnings.filterwarnings('ignore')

class TriModelNetworkMonitor:
    def __init__(self):
        print("\n🔵 Loading models and scaler...")
        self.model_iforest = joblib.load("models/model_iso.pkl")
        self.model_auto = load_model("models/model_auto.h5", compile=False)
        self.scaler = joblib.load("models/scaler_auto.pkl")

        self.protocol_map = {'TCP': 0, 'UDP': 1, 'IP': 2, 'ARP': 3, 'Unknown': -1}
        self.end_time = 0
        self.anomaly_score_buffer = []
        self.packet_rate_buffer = []
        self.last_packet_time = time.time()
        self.live_plotting = False
        self.anomaly_count = 0

    def monitor_network(self, minutes=5):
        print(f"\n🚀 Monitoring started for {minutes} minutes...")
        self.end_time = time.time() + minutes * 60
        self.sniffer = AsyncSniffer(prn=self.packet_handler, store=False)
        self.sniffer.start()

        while time.time() < self.end_time:
            time.sleep(1)

        self.sniffer.stop()
        print(f"\n✅ Monitoring finished. Total Anomalies detected: {self.anomaly_count}")

    def packet_handler(self, packet):
        if time.time() > self.end_time:
            return

        try:
            pkt = self._process_packet(packet)
            features = self._prepare_features(pkt)
            X = self.scaler.transform([features])

            pred_iforest = self.model_iforest.predict(X)[0]

            reconstructed = self.model_auto.predict(X)
            mse = np.mean(np.power(X - reconstructed, 2))
            pred_auto = -1 if mse > 0.01 else 1

            votes = [pred_iforest, pred_auto]

            # Isolation Forest and Autoencoder normal = 1, anomaly = -1
            if votes.count(-1) >= 2 or (pred_iforest == -1 and pred_auto == -1):
                self._raise_alert(pkt)

            # Record scores for plotting
            self.anomaly_score_buffer.append(mse)

            now = time.time()
            elapsed = now - self.last_packet_time
            self.last_packet_time = now
            if elapsed > 0:
                self.packet_rate_buffer.append(1 / elapsed)

            if len(self.anomaly_score_buffer) > 200:
                self.anomaly_score_buffer.pop(0)

        except Exception as e:
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
            'src': src,
            'dst': dst,
            'proto': proto,
            'size': len(packet),
            'port': port,
            **flags
        }

    def _prepare_features(self, pkt):
        return [
            self.protocol_map.get(pkt['proto'], -1),
            pkt['size'],
            0,  # time_diff (set to 0 live)
            datetime.now().hour,
            pkt.get('SYN', 0),
            pkt.get('ACK', 0),
            pkt.get('FIN', 0),
            pkt.get('RST', 0),
            pkt.get('port', 0)
        ]

    def _raise_alert(self, pkt):
        self.anomaly_count += 1
        print(f"\n🚨 ALERT: Anomalous Packet Detected!")
        print(f"Source: {pkt['src']} ➔ Destination: {pkt['dst']} | Protocol: {pkt['proto']} | Size: {pkt['size']} bytes")
        print("-" * 60)

    def start_live_dashboard(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        def animate(i):
            ax1.clear()
            ax2.clear()

            if self.anomaly_score_buffer:
                ax1.plot(self.anomaly_score_buffer)
                ax1.set_title("Live MSE Anomaly Score (Autoencoder)")

            if self.packet_rate_buffer:
                ax2.plot(self.packet_rate_buffer[-100:], color='orange')
                ax2.set_title("Packet Rate (packets/sec)")

            plt.tight_layout()

        ani = animation.FuncAnimation(fig, animate, interval=1000)
        plt.show()

if __name__ == "__main__":
    monitor = TriModelNetworkMonitor()

    t1 = Thread(target=monitor.monitor_network, args=(5,))
    t2 = Thread(target=monitor.start_live_dashboard)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print("✅ Monitoring session complete.")
