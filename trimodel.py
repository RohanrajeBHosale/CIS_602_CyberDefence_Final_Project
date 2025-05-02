# autodemo_trimodel.py

import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from datetime import datetime
from scapy.all import AsyncSniffer, IP, UDP, ARP, TCP, send
from tensorflow.keras.models import load_model
from threading import Thread

warnings.filterwarnings('ignore')

class TwoModelNetworkMonitor:
    def __init__(self):
        print("\n🔵 Loading models and scaler...")

        print("\nSelect models to use:")
        print("1. Isolation Forest only")
        print("2. Autoencoder only")
        print("3. Both models")

        model_choice = input("Enter choice (1/2/3): ").strip()

        self.use_iforest = False
        self.use_autoencoder = False

        if model_choice == "1":
            self.use_iforest = True
            self.model_iforest = joblib.load("models/model_iso.pkl")
            print("✅ Loaded Isolation Forest only.")
        elif model_choice == "2":
            self.use_autoencoder = True
            self.model_auto = load_model("models/model_auto.h5", compile=False)
            print("✅ Loaded Autoencoder only.")
        else:
            self.use_iforest = True
            self.use_autoencoder = True
            self.model_iforest = joblib.load("models/model_iso.pkl")
            self.model_auto = load_model("models/model_auto.h5", compile=False)
            print("✅ Loaded BOTH models.")

        self.scaler = joblib.load("models/scaler_auto.pkl")

        self.protocol_map = {'TCP': 0, 'UDP': 1, 'IP': 2, 'ARP': 3, 'Unknown': -1}
        self.end_time = 0
        self.anomaly_score_buffer = []
        self.packet_rate_buffer = []
        self.last_packet_time = time.time()
        self.anomaly_count = 0
        self.anomalies = []

    def monitor_network(self, minutes=5):
        print(f"\n🚀 Monitoring started for {minutes} minutes...")
        self.end_time = time.time() + minutes * 60

        # Force correct WiFi interface (for Mac Wi-Fi use 'en0')
        self.sniffer = AsyncSniffer(prn=self.packet_handler, store=False, iface="en0")
        self.sniffer.start()

        while time.time() < self.end_time:
            time.sleep(1)

        self.sniffer.stop()
        print(f"\n✅ Monitoring finished. Total Anomalies detected: {self.anomaly_count}")
        self._save_report()

    def packet_handler(self, packet):
        if time.time() > self.end_time:
            return

        try:
            # Print every captured packet
            print(f"[Captured] {packet.summary()}")

            pkt = self._process_packet(packet)
            features = self._prepare_features(pkt)
            X = self.scaler.transform([features])

            votes = []

            if self.use_iforest:
                pred_iforest = self.model_iforest.predict(X)[0]
                votes.append(pred_iforest)

            if self.use_autoencoder:
                reconstructed = self.model_auto.predict(X)
                mse = np.mean(np.power(X - reconstructed, 2))
                pred_auto = -1 if mse > 0.005 else 1  # More sensitive now
                votes.append(pred_auto)

                self.anomaly_score_buffer.append(mse)

            now = time.time()
            elapsed = now - self.last_packet_time
            self.last_packet_time = now
            if elapsed > 0:
                self.packet_rate_buffer.append(1 / elapsed)

            if len(self.anomaly_score_buffer) > 200:
                self.anomaly_score_buffer.pop(0)

            if votes.count(-1) >= 1:
                self._raise_alert(pkt, mse if self.use_autoencoder else 0.0)

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
            0,
            datetime.now().hour,
            pkt.get('SYN', 0),
            pkt.get('ACK', 0),
            pkt.get('FIN', 0),
            pkt.get('RST', 0),
            pkt.get('port', 0)
        ]

    def _raise_alert(self, pkt, mse):
        self.anomaly_count += 1
        print(f"\n🚨 ALERT: Anomalous Packet Detected!")
        print(f"Source: {pkt['src']} ➔ Destination: {pkt['dst']} | Protocol: {pkt['proto']} | Size: {pkt['size']} bytes | MSE: {mse:.4f}")
        print("-" * 60)

        self.anomalies.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": pkt['src'],
            "destination_ip": pkt['dst'],
            "protocol": pkt['proto'],
            "size": pkt['size'],
            "port": pkt['port'],
            "mse_score": mse
        })

    def _save_report(self):
        if self.anomalies:
            df = pd.DataFrame(self.anomalies)
            os.makedirs("reports", exist_ok=True)
            filename = f"reports/anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            print(f"\n📄 Anomaly report saved: {filename}")
        else:
            print("\n📄 No anomalies detected, no report generated.")

    def start_live_dashboard(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        def animate(i):
            ax1.clear()
            ax2.clear()

            if self.anomaly_score_buffer:
                ax1.plot(self.anomaly_score_buffer)
                ax1.axhline(y=0.01, color='r', linestyle='--')
                ax1.set_title("Live MSE Anomaly Score (Autoencoder)")

            if self.packet_rate_buffer:
                ax2.plot(self.packet_rate_buffer[-100:], color='orange')
                ax2.set_title("Packet Rate (packets/sec)")

            plt.tight_layout()

        ani = animation.FuncAnimation(fig, animate, interval=1000)
        plt.show()

def inject_anomalies():
    time.sleep(10)  # Wait 10 seconds after monitoring starts

    # Use WiFi IP here instead of localhost
    target_ip = "134.88.131.186"  # ⚡ Change this to your real Wi-Fi IP address ⚡

    print("\n🚀 Injecting anomalous packets...")
    for i in range(10):
        pkt = IP(dst=target_ip)/UDP(dport=9999)/("X"*5000)
        send(pkt, verbose=False)
        print(f"Injected packet {i+1}")
        time.sleep(0.3)
    print("✅ Anomaly injection complete!")

if __name__ == "__main__":
    monitor = TwoModelNetworkMonitor()

    t1 = Thread(target=monitor.monitor_network, args=(5,))
    t2 = Thread(target=monitor.start_live_dashboard)
    t3 = Thread(target=inject_anomalies)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print("✅ Monitoring session complete.")