from scapy.all import IP, UDP, send
import time

target_ip = "134.88.131.186"  # your IP

print("\n🚀 Aggressive anomaly injection started...")

for i in range(300):  # Inject 300 anomalous packets
    # Create much bigger payload: 4000 bytes instead of 1300
    pkt = IP(dst=target_ip)/UDP(dport=4444)/("X"*10000)
    send(pkt, verbose=False)
    print(f"Injected packet {i+1}")
    time.sleep(0.005)  # faster injection (5ms gap)

print("✅ Anomaly injection complete!")