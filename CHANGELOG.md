# Changelog

All notable changes to AI Access Sentinel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
