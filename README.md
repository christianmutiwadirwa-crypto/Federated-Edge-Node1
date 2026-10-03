#  Federated Edge Node — IIoT Predictive Maintenance & Intrusion Detection

![Architecture](https://img.shields.io/badge/Architecture-Edge%20Computing-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-PyTorch%20%7C%20Scikit--Learn-orange)
![Hardware](https://img.shields.io/badge/Hardware-ESP32%20%7C%20Raspberry%20Pi-red)
![License](https://img.shields.io/badge/License-MIT-green)

A production-ready, decentralized edge computing client designed for **Industrial IoT (IIoT)**. This repository contains the complete implementation for **Node 1**, which acquires high-frequency physical telemetry, executes real-time inference to detect anomalies, and participates in a privacy-preserving Federated Learning network.

---

##  System Architecture

This repository acts as a monorepo for the three critical components of the node's ecosystem:

1. **`esp32/` (Physical Sensor Acquisition):** 
   C++ firmware for an ESP32 microcontroller that interfaces with physical sensors (ADXL345 for vibration via SPI, DHT11 for environment). It calculates statistical features (RMS, Skewness, Kurtosis) on the fly and streams lightweight, packed binary payloads over TCP/WiFi.

2. **`rpi/` (Intelligent Edge Node & FL Client):** 
   The Python-based edge server running on a Raspberry Pi. It ingests sensor data and executes a dual-layer inference pipeline:
   - **Physical Anomaly Detection:** An independent Isolation Forest detecting physical tampering (e.g., shaker manipulation).
   - **Cyber Intrusion Detection:** A PyTorch Multilayer Perceptron (MLP) trained to detect network-level attacks.
   - **Federated Client:** Manages autonomous participation in federated training rounds, pulling global models, computing local gradients, and pushing securely.

3. **`federated_server/` (Cloud Orchestrator):** 
   A lightweight FastAPI cloud server orchestrating the federated network. It aggregates model updates from multiple edge nodes using a customized FedProx algorithm.

---

##  The Machine Learning Engine

This project pushes the boundaries of edge AI by implementing a highly custom training pipeline locally on the edge device (`rpi/src/train_federated_node.py`):

* **Composite Loss Function:** Local models are optimized using a unique composite loss function that combines:
  * Standard Cross-Entropy (for classification accuracy)
  * Knowledge Distillation (penalizing divergence from the global model's soft predictions)
  * FedProx Proximal Penalty (restricting local weight updates from drifting too far from the global baseline)
* **Class-Aware Aggregation:** Handles unbalanced datasets by dynamically freezing output neurons for classes absent in the local node's dataset, preventing catastrophic forgetting during local epochs.
* **Hot-Reloading Inference:** The `InferenceEngine` automatically detects new global weights pulled from the orchestrator and hot-swaps the PyTorch model without dropping a single packet.

---

##  Deployment & Setup

### 1. Flash the ESP32
Navigate to the `esp32/` directory and configure your network:
1. Open `include/Config.h` and set your `WIFI_SSID` and `WIFI_PASSWORD`.
2. Build and upload using PlatformIO:
```bash
cd esp32
pio run --target upload
```

### 2. Start the Raspberry Pi Edge Client
Ensure you are running Python 3.10+.
```bash
cd rpi
pip install -r requirements.txt
./start_server.sh
```
*The node will immediately begin listening on TCP Port 9000 for the ESP32 stream.*

### 3. Deploy the Cloud Orchestrator (Render)
The federated aggregator is designed to be hosted serverlessly on platforms like Render.
1. Connect this GitHub repository to Render.
2. Select **Docker** as the environment.
3. Set the **Root Directory** to `federated_server` (This is critical—it isolates the server deployment).
4. Deploy! Render will automatically build the `Dockerfile` and expose the aggregation endpoints.

---

## 🔬 Research Context
This codebase was developed as part of an applied research study on decentralized machine learning in hostile IIoT environments. The architecture proves that complex defensive mechanisms (Isolation Forests + Distilled MLPs) can operate entirely on constrained edge hardware while maintaining global intelligence through Federated Learning.
