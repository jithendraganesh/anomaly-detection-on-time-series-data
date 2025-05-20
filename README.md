# anomaly-detection-on-time-series-data
The project consists of two complementary approaches to time-series anomaly detection:

1. LSTM Autoencoder-Based Detection
Uses bidirectional LSTM layers to learn temporal dependencies

Trained on normal engine behavior to detect deviations (anomalies)

Incorporates both static and adaptive thresholding

2. Anomaly Transformer with KDE Thresholding
Leverages Transformer-based attention mechanisms for anomaly detection

Incorporates attention-based KL divergence loss to model temporal dependencies

Uses Kernel Density Estimation (KDE) for adaptive anomaly scoring

🧠 Core Components
🔹 CMAPSS Dataset Structure
Inputs:

Operational settings: op1, op2, op3

21 sensors: sensor1 to sensor21

Metadata:

Engine unit ID, cycle number

Target: Remaining Useful Life (RUL) for each engine

🔹 LSTM Autoencoder Architecture
Encoder:

Bidirectional LSTM layers (64 → 32 units)

Bottleneck:

Dense layer (32 units)

Decoder:

Bidirectional LSTM layers (32 → 64 units)

Regularization:

Dropout layers (0.2)

🔍 Anomaly Detection
Model reconstructs input sequences

Anomaly score = Reconstruction error (MSE)

Static Threshold: 95th percentile of training errors

Adaptive Threshold: Based on rolling error windows

🔹 Anomaly Transformer Architecture
Input: Multivariate time-series windows

Layers:

3 Encoder Layers

Multi-head Self-Attention (8 heads)

d_model = 512, d_ff = 512

Activation = GELU, Dropout = 0.1

🔍 Anomaly Detection with KDE
Model learns to reconstruct input sequences

Reconstruction error + KL divergence used during training

KDE is used to estimate the log-density of test errors

Thresholding:

Bottom 5% of log-density → classified as anomaly

📈 Training and Evaluation
✅ Preprocessing
Random Forest for sensor selection based on RUL importance

MinMax normalization across all selected features

Windowed sequence creation for temporal modeling

🧪 Evaluation Metrics
Distribution plots of reconstruction errors

KDE-based log-probability visualizations

Number and distribution of detected anomalies

Feature importance visualization

📊 Visualization Outputs
Histogram of KDE Log-Probabilities

Highlights anomalies as low-density points

Sequence-wise Anomaly Plot

Shows test sequences with log-probabilities

Anomalies marked with red 'x'

💡 Results and Insights
Anomaly Transformer + KDE shows high precision in adaptive anomaly detection

LSTM Autoencoder captures long-term patterns well and benefits from bidirectional encoding

Feature importance plots help interpret which sensors contribute to failure predictions

🔧 Technologies Used
Language: Python

Libraries:

PyTorch (deep learning)

scikit-learn (KDE, Random Forest)

Matplotlib (visualization)

NumPy, pandas (data handling)

💼 Applications
Predictive Maintenance: Proactively flagging component failures

Aerospace Engineering: Diagnosing engine health using multivariate telemetry

Real-time Monitoring: Can be integrated into aircraft systems for continuous surveillance

Cost Optimization: Reducing unplanned downtime and repair costs
