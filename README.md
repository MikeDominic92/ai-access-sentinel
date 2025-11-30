# AI Access Sentinel

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

> Machine Learning-powered anomaly detection and predictive identity governance for modern IAM systems

## Overview

AI Access Sentinel is an advanced Identity and Access Management (IAM) solution that leverages machine learning to detect anomalous access patterns, predict access risks, and discover hidden role structures. It combines traditional IAM principles with modern ML techniques to provide intelligent, proactive security monitoring.

## Motivation

Traditional IAM systems are reactive - they enforce policies but don't predict threats. As organizations grow, access patterns become complex and manual review becomes impossible. This project demonstrates:

- **AI/ML convergence with IAM**: Using data science to enhance identity security
- **Proactive threat detection**: Identifying anomalies before they become breaches
- **Intelligent role management**: Discovering actual vs. assigned roles through clustering
- **Risk-based access control**: Dynamic risk scoring for adaptive security

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Sources                             │
│  (Synthetic IAM Logs: Users, Resources, Access Events)      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Processing Pipeline                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Ingestion  │→ │Preprocessing │→ │   Feature    │      │
│  │              │  │              │  │ Engineering  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   ML Model Suite                             │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │   Anomaly      │  │    Access      │  │     Role     │  │
│  │   Detection    │  │  Prediction    │  │    Mining    │  │
│  │ (Iso. Forest)  │  │(Random Forest) │  │  (K-Means)   │  │
│  └────────────────┘  └────────────────┘  └──────────────┘  │
│  ┌────────────────┐                                         │
│  │   Risk Scorer  │                                         │
│  │   (Ensemble)   │                                         │
│  └────────────────┘                                         │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│   FastAPI    │  │  Streamlit   │
│     API      │  │  Dashboard   │
│              │  │              │
│  - Analyze   │  │ - Real-time  │
│  - Predict   │  │   Monitoring │
│  - Score     │  │ - Viz        │
└──────────────┘  └──────────────┘
```

## ML Models Explained (Non-Technical)

### 1. Anomaly Detection
**What it does**: Finds unusual access patterns that might indicate security threats.

**How it works**: Like a guard who learns normal behavior - if someone usually logs in from New York during business hours, but suddenly logs in from Russia at 3 AM, that's flagged as suspicious.

**Algorithms Used**:
- **Isolation Forest**: Identifies outliers by randomly splitting data (like finding a red apple in a basket of green ones)
- **One-Class SVM**: Learns what "normal" looks like and flags anything different
- **Local Outlier Factor**: Compares each event to its neighbors to find odd ones out

### 2. Access Prediction
**What it does**: Recommends whether an access request should be approved.

**How it works**: Looks at similar users (same department, role, seniority) and checks if they have that access. If 90% of similar users have it, recommendation is "approve."

**Algorithm**: Random Forest (combines multiple decision trees for accurate predictions)

### 3. Role Mining
**What it does**: Discovers hidden role patterns by clustering users with similar access.

**How it works**: Groups users who access the same resources, revealing actual working roles vs. assigned titles. Helps identify role creep and over-privileged accounts.

**Algorithm**: K-Means Clustering (groups similar users together automatically)

### 4. Risk Scoring
**What it does**: Assigns each user a risk score (0-100) based on behavior.

**How it works**: Combines multiple factors:
- Number of anomalies detected
- Deviation from peer behavior
- Sensitive resource access
- Policy violations

## Getting Started

### Prerequisites

- Python 3.9+
- pip package manager
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MikeDominic92/ai-access-sentinel.git
cd ai-access-sentinel
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Generate Synthetic Data

```bash
python -c "from src.data.generators import IAMDataGenerator; gen = IAMDataGenerator(); gen.generate_complete_dataset()"
```

This creates realistic IAM logs with 10,000+ records including normal and anomalous patterns.

### Train Models

```bash
# Using Jupyter notebooks (recommended for learning)
jupyter lab

# Or train via Python script
python -c "from src.models.anomaly_detector import AnomalyDetector; detector = AnomalyDetector(); detector.train_from_file('data/sample_iam_logs.csv')"
```

### Run API Server

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard will open at `http://localhost:8501`

## API Documentation

### Analyze Access Event

```bash
POST /api/v1/analyze/access
Content-Type: application/json

{
  "user_id": "U001",
  "resource": "financial_database",
  "action": "read",
  "timestamp": "2024-01-15T03:45:00Z",
  "source_ip": "192.168.1.100",
  "location": "New York, US"
}

Response:
{
  "is_anomaly": true,
  "risk_score": 87.5,
  "anomaly_score": -0.45,
  "reasons": [
    "Unusual time of access (outside business hours)",
    "High-risk resource accessed",
    "Different location than typical"
  ],
  "recommendation": "BLOCK"
}
```

### Get User Risk Score

```bash
GET /api/v1/user/U001/risk-score

Response:
{
  "user_id": "U001",
  "risk_score": 72.3,
  "risk_level": "HIGH",
  "factors": {
    "anomaly_count": 12,
    "peer_deviation": 2.5,
    "sensitive_access": 8,
    "policy_violations": 3
  },
  "recommendation": "Increase monitoring, require MFA"
}
```

### Discover Roles

```bash
POST /api/v1/roles/discover

Response:
{
  "clusters": 8,
  "roles": [
    {
      "role_id": "R001",
      "name": "Data Analysts",
      "user_count": 45,
      "common_resources": ["analytics_db", "reporting_tool"],
      "description": "Users with data analysis access patterns"
    },
    ...
  ]
}
```

### Model Metrics

```bash
GET /api/v1/model/metrics

Response:
{
  "anomaly_detector": {
    "precision": 0.89,
    "recall": 0.85,
    "f1_score": 0.87,
    "contamination": 0.05
  },
  "access_predictor": {
    "accuracy": 0.92,
    "precision": 0.90,
    "recall": 0.88
  }
}
```

## Use Cases

### 1. Insider Threat Detection
**Scenario**: Employee starts accessing sensitive files outside normal hours.

**Detection**: Anomaly detector flags unusual time + resource combination, risk score jumps to 85.

**Action**: Security team receives alert, reviews activity, discovers compromised credentials.

### 2. Privilege Escalation Prevention
**Scenario**: User requests admin access to production database.

**Analysis**: Access predictor checks peer group - 0% of similar users have this access.

**Action**: Request automatically flagged for manual review with context.

### 3. Role Optimization
**Scenario**: Company has 200+ custom roles, many overlapping.

**Discovery**: Role mining clusters users into 15 natural groups based on actual access.

**Action**: IAM team consolidates roles, reducing complexity by 85%.

### 4. Compliance Monitoring
**Scenario**: Need to identify users with excessive privileges.

**Analysis**: Risk scorer identifies top 10% of users with abnormal access patterns.

**Action**: Automated access recertification triggered for high-risk users.

## Project Structure

```
ai-access-sentinel/
├── src/
│   ├── data/          # Data generation and preprocessing
│   ├── models/        # ML models (anomaly, prediction, clustering)
│   ├── api/           # FastAPI REST endpoints
│   └── utils/         # Visualization and helpers
├── notebooks/         # Jupyter notebooks for experimentation
├── dashboard/         # Streamlit web dashboard
├── tests/            # Unit and integration tests
├── docs/             # Documentation and ADRs
└── docker/           # Containerization
```

## Development

### Running Tests

```bash
pytest tests/ -v --cov=src
```

### Code Quality

```bash
# Linting
flake8 src/ tests/

# Type checking
mypy src/

# Format
black src/ tests/
```

### Docker Deployment

```bash
docker-compose up -d
```

Services:
- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`

## Roadmap

### Phase 1 (Current)
- [x] Anomaly detection with Isolation Forest
- [x] Access prediction with Random Forest
- [x] Role mining with K-Means
- [x] Risk scoring engine
- [x] FastAPI REST API
- [x] Streamlit dashboard

### Phase 2 (Next)
- [ ] UEBA (User and Entity Behavior Analytics) integration
- [ ] Real-time streaming with Apache Kafka
- [ ] Graph-based access analysis (NetworkX)
- [ ] Deep learning models (LSTM for sequence analysis)
- [ ] Integration with SIEM systems

### Phase 3 (Future)
- [ ] SOAR (Security Orchestration) integration
- [ ] Automated remediation workflows
- [ ] Natural language policy queries
- [ ] Multi-tenant support
- [ ] Cloud IAM integration (AWS IAM, Azure AD)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

See [docs/SECURITY.md](docs/SECURITY.md) for security considerations and responsible disclosure.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Author

**Mike Dominic**
- GitHub: [@MikeDominic92](https://github.com/MikeDominic92)
- Portfolio Project: IAM + AI/ML convergence

## Acknowledgments

- Inspired by real-world IAM challenges in enterprise environments
- Built as a portfolio demonstration of ML engineering + security expertise
- Uses industry-standard algorithms and best practices

---

**Note**: This is a demonstration project using synthetic data. For production use, integrate with actual IAM logs and undergo thorough security review.
