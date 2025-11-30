# AI Access Sentinel - Identity Threat Detection and Response (ITDR) Platform

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

> Machine Learning-powered ITDR platform with UEBA, anomaly detection, and predictive identity governance for modern zero-trust environments

## Overview

AI Access Sentinel is an advanced Identity Threat Detection and Response (ITDR) platform that leverages machine learning and User and Entity Behavior Analytics (UEBA) to detect identity-based threats, predict access risks, and discover hidden role structures. It combines traditional IAM principles with modern ML techniques to provide intelligent, proactive security monitoring and real-time identity threat detection.

**What is ITDR?**
Identity Threat Detection and Response (ITDR) is a cybersecurity category focused on defending against identity-based attacks - the leading attack vector in modern breaches. ITDR solutions monitor identity systems, detect credential compromise, privilege escalation, and lateral movement, then respond with automated remediation.

## Motivation

Traditional IAM systems are reactive - they enforce policies but don't predict identity-based threats. As organizations grow, the identity attack surface expands and manual review becomes impossible. This ITDR platform demonstrates:

- **AI/ML convergence with IAM**: Using data science and UEBA to enhance identity security
- **Proactive threat detection**: Identifying credential compromise and anomalies before they become breaches
- **Intelligent role management**: Discovering actual vs. assigned roles through clustering
- **Risk-based access control**: Dynamic risk scoring for adaptive security and identity-based zero trust
- **Identity threat response**: Automated detection and response to lateral movement and privilege escalation

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

## ITDR Capabilities

AI Access Sentinel provides comprehensive Identity Threat Detection and Response capabilities:

### Identity Attack Surface Monitoring
- **Continuous visibility** into all identity-based access across the organization
- **Real-time monitoring** of user and entity behavior patterns
- **Attack path analysis** to identify potential lateral movement routes
- **Privilege mapping** to understand the blast radius of compromised accounts

### Credential Compromise Detection
- **Anomalous authentication patterns**: Impossible travel, unusual times, new devices
- **Suspicious access behavior**: Accessing resources never used before
- **Brute force detection**: Multiple failed login attempts from same or different IPs
- **Credential stuffing identification**: Same credentials used across multiple accounts

### Privilege Escalation Prevention
- **Peer-based access validation**: Compare requests against similar users
- **Role deviation alerts**: Flag users with permissions outside their role cluster
- **Temporary privilege tracking**: Monitor for privilege escalation attempts
- **Administrative action monitoring**: Elevated scrutiny for admin-level operations

### Lateral Movement Detection
- **Cross-resource access patterns**: Identify unusual resource hopping
- **Time-series behavioral analysis**: Detect rapid access to multiple systems
- **Network segmentation awareness**: Flag access across security boundaries
- **Entity relationship mapping**: Understand normal vs. abnormal access paths

### Identity-Based Zero Trust
- **Continuous authentication**: Real-time risk scoring for every access request
- **Context-aware access control**: Location, time, device, behavior all factor into decisions
- **Adaptive security posture**: Automatically increase scrutiny for high-risk users
- **Just-in-time access validation**: Verify necessity for each access attempt

### SIEM/SOAR Integration
- **Structured alert format**: Easy integration with security operations platforms
- **Automated response triggers**: Block/challenge high-risk access automatically
- **Threat intelligence enrichment**: Correlate with external threat feeds
- **Incident investigation support**: Detailed access forensics for IR teams

## UEBA Features

User and Entity Behavior Analytics (UEBA) powers the ITDR capabilities:

### Behavioral Baselining
- **Individual user profiles**: Learn normal patterns for each user (working hours, locations, resources)
- **Peer group analysis**: Compare behavior against similar users (role, department, seniority)
- **Entity behavior modeling**: Track service accounts, applications, and API usage
- **Temporal pattern recognition**: Understand daily, weekly, and seasonal variations

### Anomaly Detection
- **Statistical outlier detection**: Identify behavior that deviates from established baselines
- **Multi-dimensional analysis**: Consider time, location, resource, action, and context
- **Anomaly severity scoring**: Prioritize alerts based on degree of deviation
- **False positive reduction**: ML models adapt to reduce alert fatigue

### Risk-Based Prioritization
- **Dynamic risk scores**: User risk changes in real-time based on current behavior
- **Contextual risk assessment**: Same action can be high-risk or normal based on context
- **Risk aggregation**: Combine multiple low-risk events into high-risk patterns
- **Automated triage**: Focus security teams on highest-risk activities

### Advanced Analytics
- **Graph-based access analysis**: Understand relationships between users, resources, and access patterns
- **Sequence analysis**: Detect multi-step attack patterns (reconnaissance, access, exfiltration)
- **Clustering and segmentation**: Automatically discover user groups and role patterns
- **Predictive modeling**: Forecast future access needs and potential risks

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

### 1. Credential Compromise Detection (ITDR)
**Scenario**: Attacker compromises employee credentials through phishing.

**Detection**: UEBA detects impossible travel (login from US, then Russia 10 minutes later), unusual resource access, and abnormal time of access. Risk score jumps to 95.

**ITDR Response**: Automatically blocks access, forces password reset, triggers MFA verification, alerts SOC team.

**Outcome**: Credential-based attack stopped before lateral movement begins.

### 2. Insider Threat Detection (UEBA)
**Scenario**: Employee planning to leave company starts accessing sensitive files outside normal hours.

**Detection**: Anomaly detector flags unusual time + resource combination + volume spike. UEBA identifies deviation from established baseline.

**Action**: Security team receives high-priority alert, reviews activity timeline, discovers data exfiltration attempt.

**Outcome**: Insider threat neutralized, access revoked, incident documented for legal team.

### 3. Privilege Escalation Prevention (ITDR)
**Scenario**: Compromised low-privilege account attempts to request admin access to production database.

**Analysis**: Access predictor checks peer group - 0% of similar users have this access. ITDR detects privilege escalation pattern.

**Action**: Request automatically denied, SOC alerted, account flagged for investigation.

**Outcome**: Lateral movement blocked at privilege escalation stage.

### 4. Lateral Movement Detection (ITDR)
**Scenario**: Attacker moves from compromised marketing account to finance systems.

**Detection**: UEBA detects unusual cross-departmental resource access pattern. Sequence analysis identifies reconnaissance-to-access pattern typical of lateral movement.

**ITDR Response**: Isolate account, block access to additional resources, initiate incident response.

**Outcome**: Attack contained to single compromised account, no data breach.

### 5. Role Optimization
**Scenario**: Company has 200+ custom roles, many overlapping, creating identity attack surface bloat.

**Discovery**: Role mining clusters users into 15 natural groups based on actual access. Identifies 85 ghost permissions and 30 over-privileged accounts.

**Action**: IAM team consolidates roles, reducing complexity by 85% and shrinking identity attack surface.

**Outcome**: Simplified access management, reduced risk, improved compliance posture.

### 6. Compliance Monitoring
**Scenario**: Need to identify users with excessive privileges for SOX/HIPAA compliance.

**Analysis**: Risk scorer identifies top 10% of users with abnormal access patterns. UEBA flags users accessing sensitive data outside job requirements.

**Action**: Automated access recertification triggered for high-risk users, audit trail generated.

**Outcome**: Compliance requirements met, excessive privileges revoked, documented remediation.

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

## Deployment Verification

This project is fully functional with working ML models, ITDR capabilities, and UEBA features. Comprehensive deployment evidence is available in [docs/DEPLOYMENT_EVIDENCE.md](docs/DEPLOYMENT_EVIDENCE.md).

### Quick Verification Commands

```bash
# 1. Start FastAPI server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 2. Test anomaly detection
curl -X POST http://localhost:8000/api/v1/analyze/access \
  -H "Content-Type: application/json" \
  -d '{"user_id":"U123","resource":"database","action":"read","timestamp":"2024-11-30T03:00:00Z"}'

# 3. Get user risk score
curl http://localhost:8000/api/v1/user/U123/risk-score

# 4. Launch Streamlit dashboard
streamlit run dashboard/app.py
# Open http://localhost:8501
```

### Sample Evidence Included

The deployment evidence documentation provides:
- Anomaly detection API response with 94% confidence
- User risk scoring with detailed factor breakdown
- Role mining cluster results (8 discovered roles)
- ML model training outputs (96%+ accuracy)
- Streamlit dashboard screenshots and features
- Test execution results with 96% code coverage

See [Deployment Evidence](docs/DEPLOYMENT_EVIDENCE.md) for complete verification and outputs.

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
- [ ] Advanced UEBA features (entity behavior profiling, advanced peer analysis)
- [ ] Real-time streaming with Apache Kafka for instant ITDR response
- [ ] Graph-based identity attack path analysis (NetworkX)
- [ ] Deep learning models (LSTM for attack sequence detection)
- [ ] SIEM system integration (Splunk, Sentinel, QRadar)
- [ ] Threat intelligence feed integration for credential compromise detection

### Phase 3 (Future)
- [ ] SOAR (Security Orchestration) integration for automated ITDR workflows
- [ ] Automated remediation and response playbooks
- [ ] Natural language policy queries and threat hunting
- [ ] Multi-tenant ITDR support for MSPs
- [ ] Cloud IAM integration (AWS IAM Identity Center, Azure AD, Okta)
- [ ] Identity attack surface reduction recommendations
- [ ] Zero trust architecture scoring and recommendations

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
