# Changelog

All notable changes to AI Access Sentinel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-12-05

### Added - Role Attestation Engine & SoD Detection

This release adds enterprise role attestation capabilities with Segregation of Duties (SoD) conflict detection, campaign management, and compliance evidence generation.

#### Role Attestation Engine (`src/attestation/`)

- **AttestationEngine** (`attestation_engine.py`)
  - Campaign-based role attestation workflows
  - Support for attestation types:
    - **Role Definition Attestation** - Validate role permissions match job functions
    - **Role Assignment Attestation** - Verify users should have assigned roles
    - **Privileged Access Attestation** - Enhanced review for admin/elevated roles
  - Attestation decision tracking (certified, revoked, modified)
  - Multi-level reviewer support (manager, role owner, security team)
  - Evidence capture for audit compliance

- **SoD Conflict Detector** (`sod_detector.py`)
  - Rule-based Segregation of Duties detection
  - Pre-built conflict rules:
    - Approve vs Process payments
    - Create vs Approve purchase orders
    - Admin vs Audit functions
    - Development vs Production access
  - Conflict severity levels (CRITICAL, HIGH, MEDIUM, LOW)
  - Custom rule definition support
  - Conflict remediation recommendations

- **Campaign Manager** (`campaign_manager.py`)
  - Attestation campaign lifecycle management
  - Campaign types:
    - Quarterly access reviews
    - Annual role certification
    - SoD conflict resolution
    - Privileged access reviews
  - Deadline tracking and escalation
  - Completion rate monitoring
  - Reminder notifications

#### New API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/attestation/campaigns` | Create attestation campaign |
| GET | `/api/v1/attestation/campaigns/{id}` | Get campaign status |
| POST | `/api/v1/attestation/decisions` | Submit attestation decision |
| GET | `/api/v1/sod/conflicts` | List SoD conflicts |
| POST | `/api/v1/sod/rules` | Create custom SoD rule |
| GET | `/api/v1/attestation/evidence/{campaign_id}` | Export attestation evidence |

#### Pydantic Schemas

- `AttestationCampaign` - Campaign definition and status
- `AttestationDecision` - Reviewer decision with justification
- `SoDConflict` - Detected conflict with severity and remediation
- `SoDRule` - Conflict detection rule definition
- `AttestationEvidence` - Audit-ready evidence package

### Why This Matters

This release addresses critical IAM governance requirements:

| Problem | Solution | Impact |
|---------|----------|--------|
| Roles accumulate unused permissions | Role definition attestation | Right-sized roles, reduced attack surface |
| Access creep goes undetected | Role assignment attestation | Users have only needed access |
| SoD conflicts enable fraud | Automated conflict detection | Fraud prevention, compliance |
| Audit evidence is scattered | Campaign-based evidence capture | SOC 2/ISO 27001 audit-ready |

### Interview Questions This Answers

| Question | How This Feature Answers It |
|----------|----------------------------|
| "How do you implement role attestation?" | Campaign-based attestation with multi-level reviewers |
| "How do you detect SoD conflicts?" | Rule-based detection with severity scoring |
| "How do you ensure least privilege?" | Regular role definition reviews with evidence |
| "How do you prepare for compliance audits?" | Campaign evidence packages for SOC 2/ISO 27001 |

### Compliance Alignment
- **SOC 2 CC6.1**: Role-based access controls with attestation
- **SOC 2 CC6.3**: Access authorization through attestation decisions
- **ISO 27001 A.5.18**: Access rights review through campaigns
- **ISO 27001 A.8.2**: Privileged access attestation

---

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
