# Network Anomaly Detection System

This project detects real-time network anomalies using multiple machine learning models.

## Folder Structure
- `models/` : Trained model files
- `reports/` : Graphs generated after monitoring
- `data/` : Original packet capture CSV
- `network_monitor.py` : Main monitoring script

## How to Run
1. Install requirements:
   ```
   pip install scapy pandas numpy matplotlib scikit-learn tensorflow joblib
   ```

2. Place your packet CSV file (`FIAL.csv`) inside the `data/` folder.

3. Run the program:
   ```
   python isotrain.py
   python isomodel.py
   python injector.py
   ```

4. Follow the menu options to train/load models and start real-time detection.

## Features
- Supports Isolation Forest,
- Real-time network packet monitoring.
- Visualization of anomaly scores, protocol distribution, and top source IPs.
