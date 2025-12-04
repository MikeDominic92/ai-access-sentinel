# Changelog

All notable changes to AI Access Sentinel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-12-04

### Added

#### CrowdStrike Falcon ITDR Integration
- **CrowdStrike Connector** (`src/integrations/crowdstrike_connector.py`)
  - Full FalconPy SDK integration for identity protection alerts
  - OAuth2 authentication with automatic token refresh
  - Rate limiting and error handling
  - Mock mode for development/demo without credentials
  - Methods: `connect()`, `get_identity_alerts()`, `check_indicator()`, `enrich_access_event()`

- **Falcon Event Parser** (`src/integrations/falcon_event_parser.py`)
  - Normalizes Falcon ITDR events to AI Access Sentinel format
  - Supports 13 identity attack types (credential theft, lateral movement, etc.)
  - MITRE ATT&CK tactic/technique mapping
  - Risk score calculation based on severity and confidence
  - Webhook payload parsing for real-time integration

- **Alert Correlator** (`src/integrations/alert_correlator.py`)
  - Combines Falcon ITDR alerts with ML-detected anomalies
  - Multi-factor correlation scoring (user, time, IP, pattern, resource)
  - Confidence levels: VERY_HIGH (85+), HIGH (70+), MEDIUM (50+), LOW (30+), MINIMAL
  - Automated response recommendations
  - Time-windowed correlation (configurable, default 1 hour)

#### Enhanced Risk Scoring
- **6-Factor Risk Model** (`src/models/risk_scorer.py`)
  - Added `falcon_threat` factor (25% weight when enabled)
  - Adjusted existing weights proportionally when Falcon enabled
  - Factor breakdown:
    - Anomaly score: 22.5% (was 30%)
    - Peer deviation: 15% (was 20%)
    - Sensitive access: 15% (was 20%)
    - Failed attempts: 11.25% (was 15%)
    - Policy violations: 11.25% (was 15%)
    - Falcon threat: 25% (new)
  - Falcon-aware recommendations in risk reports

#### New API Endpoints
- `POST /api/v1/falcon/webhook` - Receive Falcon ITDR webhook events
- `GET /api/v1/falcon/status` - Check Falcon connection status
- `GET /api/v1/falcon/alerts` - List cached Falcon alerts
- `GET /api/v1/falcon/user/{user_id}/risk` - Falcon-enriched risk score
- `POST /api/v1/falcon/sync` - Manual Falcon alert sync
- `GET /api/v1/falcon/correlations` - View alert correlations

#### New Pydantic Schemas
- `FalconWebhookEvent` - Incoming Falcon alert schema
- `FalconCorrelatedAlert` - Correlated alert response
- `FalconWebhookResponse` - Webhook processing response
- `FalconConnectionStatus` - Connection status schema
- `FalconEnrichedRiskScore` - Enhanced risk score with Falcon context

### Changed
- API version bumped to 1.1.0
- `RiskScorer` constructor now accepts `enable_falcon` parameter
- `calculate_user_risk_score()` accepts optional `falcon_alerts` and `falcon_enrichment`
- `calculate_batch_risk_scores()` supports Falcon alert correlation
- Health check now includes Falcon connector status
- Root endpoint shows Falcon ITDR feature status

### Dependencies
- Added `crowdstrike-falconpy>=1.3.0` - Official CrowdStrike FalconPy SDK
- Added `aiohttp>=3.9.0` - Async HTTP for Falcon webhooks

### Documentation
- Updated README with v1.1 features and Falcon API endpoints
- Added architecture diagram showing Falcon integration
- Added this CHANGELOG entry

---

## Why CrowdStrike Falcon ITDR?

### Integration Rationale

CrowdStrike Falcon Identity Threat Detection and Response was chosen for v1.1 because:

1. **Industry Leadership**: CrowdStrike is a leader in endpoint and identity protection
2. **Comprehensive ITDR**: Falcon provides real-time identity attack detection
3. **API-First Architecture**: FalconPy SDK enables seamless integration
4. **MITRE ATT&CK Alignment**: Native technique/tactic mapping
5. **Enterprise Adoption**: Used by Fortune 500 companies globally

### How It Enhances AI Access Sentinel

| Capability | Before (v1.0) | After (v1.1) |
|------------|---------------|--------------|
| Threat Detection | ML-only (internal data) | ML + Falcon threat intel |
| Risk Factors | 5 factors | 6 factors (+Falcon) |
| Attack Coverage | Behavior anomalies | + Active attacks (credential theft, lateral movement) |
| Response Time | Post-incident analysis | Real-time webhook alerts |
| Threat Intelligence | None | Falcon global threat data |

### Supported Falcon Alert Types

- Credential Theft (T1003)
- Credential Compromise
- Lateral Movement (T1021)
- Privilege Escalation (T1078)
- Impossible Travel
- Brute Force (T1110)
- Password Spray
- MFA Bypass
- Session Hijack
- Golden Ticket (T1558.001)
- Silver Ticket (T1558.002)
- Kerberoasting (T1558.003)
- DCSync (T1003.006)

---

## [1.0.0] - 2024-01-15

### Added
- Initial release of AI Access Sentinel
- Anomaly detection using Isolation Forest, One-Class SVM, and Local Outlier Factor
- Access prediction with Random Forest classifier
- Role mining with K-Means and hierarchical clustering
- Ensemble risk scoring engine
- FastAPI REST API with 7 endpoints
- Streamlit interactive dashboard
- Synthetic IAM data generator (10,000+ records)
- Complete test suite with 80%+ coverage
- Jupyter notebooks for experimentation and learning
- Docker containerization support
- Comprehensive documentation

### Features

#### ML Models
- **Anomaly Detector**: Identifies unusual access patterns with 87% F1 score
- **Access Predictor**: Recommends access approval with 92% accuracy
- **Role Miner**: Discovers implicit roles through clustering
- **Risk Scorer**: Multi-factor risk assessment (0-100 scale)

#### API Endpoints
- `POST /api/v1/analyze/access` - Analyze single access event
- `POST /api/v1/analyze/batch` - Batch analysis
- `GET /api/v1/anomalies` - List detected anomalies
- `GET /api/v1/user/{id}/risk-score` - User risk scoring
- `POST /api/v1/roles/discover` - Role mining
- `GET /api/v1/model/metrics` - Model performance metrics
- `GET /health` - Health check

#### Dashboard
- Real-time anomaly monitoring
- User risk score leaderboard
- Role clustering visualization
- Access pattern heatmaps
- Model performance metrics

#### Data Generation
- Realistic synthetic IAM logs
- Multiple departments and job titles
- Normal and anomalous patterns
- Geographic and temporal diversity
- 10,000+ sample records

### Documentation
- Comprehensive README with architecture diagram
- ML explainer for non-technical audiences
- Architecture Decision Records (ADRs)
- Security and cost analysis documentation
- API documentation with examples
- Contributing guidelines

### Technical
- Python 3.9+ support
- FastAPI for high-performance API
- scikit-learn for ML algorithms
- Streamlit for interactive dashboards
- Comprehensive test coverage
- Type hints throughout codebase
- Docker deployment support

## [Unreleased]

### Planned for 2.0.0
- UEBA (User and Entity Behavior Analytics) integration
- Real-time streaming with Apache Kafka
- Graph-based access analysis
- Deep learning models (LSTM for sequences)
- SIEM system integration

### Planned for 3.0.0
- SOAR (Security Orchestration) integration
- Automated remediation workflows
- Natural language policy queries
- Multi-tenant support
- Cloud IAM integration (AWS, Azure)

---

## Version History

- **1.0.0** - Initial release (2024-01-15)
