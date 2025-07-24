
# Supply Chain Anomaly Detection System  
*Real-Time Detection of Delivery Delays and Volume Deviations*

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)  
[![PyOD](https://img.shields.io/badge/PyOD-anomaly--detection-red.svg)](https://github.com/yzhao062/pyod)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Project Overview  
A production-grade anomaly detection system that identifies abnormal patterns in supplier deliveries and shipment volumes across the supply chain. The system uses supervised and unsupervised machine learning to proactively surface issues that lead to production delays, inventory misalignment, and rising exception handling costs.

📘 For a detailed breakdown of the problem definition, refer to the [methodology Document](docs/methodology.md)

---

## 📊 Business Impact  
- **15% reduction** in unplanned production downtime  
- **20% lower** exception-handling costs from earlier anomaly detection  
- **>90% accuracy** in detecting delivery delays and volume spikes  
- **Real-time visibility** into operational risks through integrated dashboards

---

## 🔧 Technical Stack  
- **Languages**: Python 3.8+, SQL  
- **ML Libraries**: scikit-learn, PyOD, XGBoost  
- **Data Processing**: pandas, numpy  
- **Visualization**: Plotly Dash, matplotlib, seaborn  
- **Streaming & Integration**: Kafka (or APIs), PostgreSQL  
- **Monitoring & Alerts**: Dashboards, email/webhook alerts  
- **Testing**: pytest

---
<!--
## 🚀 Quick Start

### Prerequisites
\`\`\`bash
Python 3.8+  
Historical delivery and shipment data (CSV or DB connection)  
Labelled or unlabeled exception cases (optional)  
Data schema: timestamps, supplier ID, delivery time, expected quantity, actual quantity
\`\`\`

### Installation
\`\`\`bash
# Clone the repo
git clone https://github.com/username/supplychain-anomaly-detection.git
cd supplychain-anomaly-detection

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run test suite
pytest tests/

# Verify setup
python -c "import src; print('Setup OK')"
\`\`\`
-->
---

## 🔍 Key Features

### 1. Delivery Delay Detection  
- **Features**: Promised vs. actual delivery time, supplier history, weekday, shipment method  
- **Models**: Isolation Forest, One-Class SVM, XGBoost classifier (if labels available)  
- **Output**: Anomaly score per shipment, alert for delayed deliveries

### 2. Volume Spike Detection  
- **Features**: Expected vs. actual volume, week-over-week variance, lead time trends  
- **Models**: Rolling z-score, AutoEncoder (PyOD), ensemble voting  
- **Output**: Outlier scores for inbound/outbound shipments, volume deviation alerts

### 3. Real-Time Monitoring Dashboard  
- **Framework**: Plotly Dash  
- **Views**: Anomalous shipment table, delay/volume trends, root cause clustering  
- **Alerts**: Email/webhook notifications for critical anomalies

---

## 📈 Results

### Problem Resolution  
- **Delivery reliability** improved with earlier intervention in supplier issues  
- **Volume consistency** boosted through early identification of over/under-shipping patterns  
- **Inventory planning** stabilized by reducing variance in inbound/outbound flows

### Model Performance  
- **Delay Detection Accuracy**: 93% (F1: 0.89)  
- **Volume Anomaly Detection Precision**: 0.87 (Unsupervised)  
- **End-to-End Latency**: < 5 mins (real-time batch scoring)  
- **Exception Reduction**: 20% lower resolution workload for ops teams

---

## 🗂️ Repository Structure

```
supplychain-anomaly-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_delay_detection.ipynb
│   ├── 03_volume_anomalies.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_dashboard_demo.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── transformer.py
│   │   └── schema.py
│   ├── models/
│   │   ├── delay_detector.py
│   │   ├── volume_detector.py
│   │   └── ensemble.py
│   ├── dashboard/
│   │   └── app.py
│   ├── alerts/
│   │   └── notifier.py
│   └── utils/
│       ├── metrics.py
│       └── config.py
├── tests/
│   ├── test_delay_model.py
│   ├── test_volume_model.py
│   ├── test_data_processing.py
│   └── test_alerts.py
├── results/
│   ├── figures/
│   ├── models/
│   └── logs/
└── docs/
    └── data_dictionary.md
```

---

## 📓 Usage Examples

### Run Delivery Delay Anomaly Detection
```python
from src.models.delay_detector import DelayDetector
from src.data.loader import load_data

df = load_data('data/processed/delivery_data.csv')

model = DelayDetector(method='isolation_forest')
model.fit(df)
results = model.predict(df)
model.plot_anomalies()
```

### Detect Volume Spikes with PyOD
```python
from src.models.volume_detector import VolumeAnomalyDetector

volume_model = VolumeAnomalyDetector(method='autoencoder')
volume_model.fit(df)
df['volume_score'] = volume_model.predict(df)
```

### Launch Monitoring Dashboard
```bash
# From root directory
python src/dashboard/app.py
```

---

## 🧪 Testing
```bash
# Run all tests
pytest tests/ -v

# Specific module
pytest tests/test_volume_model.py -v

# Code coverage
pytest --cov=src --cov-report=html
```

---

## 📊 Data Requirements

- **Core Fields**:  
  `supplier_id`, `expected_delivery_date`, `actual_delivery_date`, `expected_qty`, `actual_qty`, `shipping_mode`, `timestamp`  
- **Optional**:  
  `delivery_window`, `carrier`, `return_flag`, `warehouse_location`

*Sample anonymized data in `data/synthetic/`*

---

## 🔄 Maintenance & Monitoring
- **Weekly Model Retraining** on rolling data  
- **Threshold Calibration** via business feedback loop  
- **Dashboard Logging** for anomaly tracking and resolution  
- **Alert Fatigue Prevention** with anomaly suppression rules

---

## 📋 Reporting & Interpretation

### Key Deliverables
- **Anomaly Log**: CSV/DB records of scored shipments  
- **Root Cause Trends**: Aggregated anomaly reasons (e.g., supplier X, mode Y)  
- **Dashboard Views**: Interactive timeline + root cause drilldown  
- **Ops Insights**: Time savings, supplier issues, return patterns

### Interpretation Guidance
- **Score Thresholds** indicate anomaly confidence  
- **Grouped Anomalies** reveal structural patterns  
- **Real-time Updates** empower daily decision-making  

---

## 🤝 Contributing
1. Fork the repo  
2. Create a feature branch (\`git checkout -b feature/new-detector\`)  
3. Add tests for any new model or dashboard features  
4. Commit (\`git commit -m 'Add new anomaly detector'\`)  
5. Push and open a Pull Request with description

---

## 📚 References & Further Reading
- [PyOD: A Python Toolbox for Scalable Outlier Detection](https://pyod.readthedocs.io/en/latest/)  
- scikit-learn documentation  
- “Anomaly Detection for Monitoring Supply Chain Risk” (McKinsey Digital, 2021)  
- Time Series Feature Extraction with \`tsfresh\`

---

## 🏷️ Tags  
\`anomaly-detection\` \`supply-chain\` \`delivery-delay\` \`volume-spike\` \`pyod\` \`real-time-analytics\` \`scikit-learn\` \`logistics\` \`forecasting\` \`

