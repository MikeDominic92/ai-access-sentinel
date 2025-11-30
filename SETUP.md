# Setup Guide - AI Access Sentinel

Complete setup instructions for AI Access Sentinel.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- Git
- 4GB RAM minimum (8GB recommended)
- Windows/Linux/macOS

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/MikeDominic92/ai-access-sentinel.git
cd ai-access-sentinel
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- FastAPI & Uvicorn (API)
- Streamlit (Dashboard)
- scikit-learn (ML)
- pandas, numpy (Data processing)
- matplotlib, plotly (Visualization)
- pytest (Testing)
- And more...

**Installation time:** ~2-5 minutes

### 4. Generate Sample Data

```bash
python generate_data.py
```

This creates:
- `data/sample_iam_logs.csv` - 10,000+ synthetic IAM access events
- Includes normal and anomalous patterns
- Ready for ML model training

**Generation time:** ~10-30 seconds

### 5. Verify Installation

```bash
# Run tests
pytest tests/ -v

# Should see all tests passing
```

## Quick Start Options

### Option 1: Jupyter Notebooks (Recommended for Learning)

```bash
jupyter lab

# Open notebooks/01_data_exploration.ipynb
```

Notebooks walk you through:
1. Data exploration
2. Feature engineering
3. Model training
4. Anomaly detection
5. Role mining

### Option 2: FastAPI Server

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Then open: http://localhost:8000/docs

Interactive API documentation with:
- `/api/v1/analyze/access` - Analyze events
- `/api/v1/user/{id}/risk-score` - Get user risk
- `/api/v1/roles/discover` - Discover roles
- And more...

### Option 3: Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

Opens: http://localhost:8501

Interactive dashboard with:
- Real-time anomaly detection
- User risk scoring
- Role mining visualization
- Access pattern analysis

### Option 4: Docker (Recommended for Production)

```bash
cd docker
docker-compose up -d
```

Services:
- API: http://localhost:8000
- Dashboard: http://localhost:8501

To stop:
```bash
docker-compose down
```

## Training ML Models

### Quick Training (Using Sample Data)

```python
# In Python shell or script
from src.models.anomaly_detector import AnomalyDetector
from src.data.preprocessors import IAMDataPreprocessor
import pandas as pd

# Load data
df = pd.read_csv('data/sample_iam_logs.csv')

# Preprocess
preprocessor = IAMDataPreprocessor()
df = preprocessor.preprocess_for_training(df)

# Train anomaly detector
detector = AnomalyDetector('isolation_forest')
feature_cols = preprocessor.get_feature_columns()
X = df[feature_cols].values
detector.train(X, feature_names=feature_cols)

# Save model
detector.save('models/trained/anomaly_detector_if.joblib')
```

### Train All Models

```bash
# Run the comprehensive training script
python -c "
from src.models.anomaly_detector import AnomalyDetector
from src.models.access_predictor import AccessPredictor
from src.models.role_miner import RoleMiner
from src.data.preprocessors import IAMDataPreprocessor
import pandas as pd

# Load and preprocess data
df = pd.read_csv('data/sample_iam_logs.csv')
preprocessor = IAMDataPreprocessor()
df = preprocessor.preprocess_for_training(df)

# Train and save anomaly detector
print('Training anomaly detector...')
detector = AnomalyDetector('isolation_forest')
feature_cols = preprocessor.get_feature_columns()
X = df[feature_cols].values
detector.train(X, feature_names=feature_cols)
detector.save('models/trained/anomaly_detector_if.joblib')

# Train and save access predictor
print('Training access predictor...')
predictor = AccessPredictor()
predictor.train(df)
predictor.save('models/trained/access_predictor.joblib')

# Train and save role miner
print('Training role miner...')
miner = RoleMiner(n_clusters=8)
miner.train(df)
miner.save('models/trained/role_miner.joblib')

print('All models trained and saved successfully!')
"
```

**Training time:** ~1-2 minutes for all models

## Configuration

### Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to customize:
```ini
# Application
APP_NAME=AI-Access-Sentinel
ENVIRONMENT=development

# API
API_HOST=0.0.0.0
API_PORT=8000

# Model Settings
ANOMALY_CONTAMINATION=0.05
RISK_SCORE_THRESHOLD=75

# Paths
MODEL_PATH=models/trained
DATA_PATH=data
```

## Directory Structure

```
ai-access-sentinel/
├── src/                 # Source code
│   ├── data/           # Data generation & preprocessing
│   ├── models/         # ML models
│   ├── api/            # FastAPI application
│   └── utils/          # Utilities
├── tests/              # Test suite
├── notebooks/          # Jupyter notebooks
├── dashboard/          # Streamlit dashboard
├── data/               # Data files
│   ├── raw/           # Raw data
│   ├── processed/     # Processed data
│   └── sample_iam_logs.csv
├── models/             # Trained models
│   └── trained/
├── docs/               # Documentation
├── docker/             # Docker files
└── .github/            # CI/CD workflows
```

## Common Issues & Troubleshooting

### Issue: Import errors

**Solution:**
```bash
# Make sure you're in the project root
cd ai-access-sentinel

# Make sure virtual environment is activated
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: "No module named 'src'"

**Solution:**
```bash
# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/macOS
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

Or run Python from project root.

### Issue: API returns 503 "No training data"

**Solution:**
```bash
# Generate data first
python generate_data.py

# Or manually create sample_iam_logs.csv
```

### Issue: Dashboard won't start

**Solution:**
```bash
# Check if Streamlit is installed
pip install streamlit

# Run from project root
streamlit run dashboard/app.py

# If port is busy, specify different port
streamlit run dashboard/app.py --server.port=8502
```

### Issue: Tests failing

**Solution:**
```bash
# Install test dependencies
pip install pytest pytest-cov

# Generate test data
python generate_data.py

# Run specific test
pytest tests/test_anomaly_detector.py -v
```

## Performance Tuning

### For Faster Training

```python
# Use fewer estimators (faster but less accurate)
detector = AnomalyDetector('isolation_forest')
detector.model.n_estimators = 50  # Default is 100

# Reduce data size for testing
df_sample = df.sample(frac=0.1)  # Use 10% of data
```

### For Production

1. **Use Docker** - Easier deployment
2. **Enable caching** - Cache model predictions
3. **Add Redis** - For session management
4. **Use Gunicorn** - Production WSGI server

```bash
pip install gunicorn
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Next Steps

1. **Explore the notebooks** - Learn how ML models work
2. **Try the API** - Test endpoints with sample data
3. **View the dashboard** - Visualize insights
4. **Read the docs** - Understand architecture
5. **Customize** - Adapt to your IAM data

## Getting Help

- **Documentation**: See `docs/` folder
- **Examples**: Check `notebooks/` folder
- **Issues**: Open GitHub issue
- **Security**: See `docs/SECURITY.md`

## Production Deployment

See `docs/COST_ANALYSIS.md` for infrastructure guidance.

**Recommended:**
1. Use Docker Compose
2. Set up proper authentication
3. Configure CORS appropriately
4. Use HTTPS (nginx/Traefik)
5. Set up monitoring (Prometheus/Grafana)
6. Configure backup for models and data

## License

MIT License - See LICENSE file

---

**Ready to get started?**

```bash
python generate_data.py
jupyter lab
# or
streamlit run dashboard/app.py
```

Enjoy exploring AI Access Sentinel!
