# Contributing to AI Access Sentinel

Thank you for your interest in contributing to AI Access Sentinel! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a positive learning environment

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. Use the bug report template
3. Include:
   - Python version
   - OS and environment details
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and logs

### Suggesting Enhancements

1. Open an issue with the enhancement label
2. Describe the use case and benefit
3. Provide examples if possible

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest tests/`
6. Update documentation as needed
7. Commit with clear messages
8. Push to your fork
9. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ai-access-sentinel.git
cd ai-access-sentinel

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

## Code Standards

### Python Style
- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for functions and classes
- Maximum line length: 100 characters

### Testing
- Write unit tests for new features
- Maintain test coverage above 80%
- Use pytest fixtures for common setups

### Documentation
- Update README.md for user-facing changes
- Add docstrings to new functions
- Create ADRs for architectural decisions

## Project Structure

```
src/
├── data/          # Data generation and preprocessing
├── models/        # ML models
├── api/           # API endpoints
└── utils/         # Utilities

tests/             # Test files mirror src/ structure
docs/              # Documentation
notebooks/         # Jupyter notebooks
```

## Commit Message Guidelines

Use conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

Example:
```
feat(anomaly): add local outlier factor detector

Implemented LOF algorithm as alternative to Isolation Forest.
Provides better detection for certain access patterns.

Closes #42
```

## Testing Guidelines

### Unit Tests
```python
def test_anomaly_detector_training():
    """Test that anomaly detector trains successfully."""
    detector = AnomalyDetector()
    df = pd.DataFrame(...)  # Sample data
    detector.train(df)
    assert detector.is_trained
```

### Integration Tests
```python
def test_api_analyze_endpoint(client):
    """Test analyze endpoint returns correct format."""
    response = client.post("/api/v1/analyze/access", json={...})
    assert response.status_code == 200
    assert "is_anomaly" in response.json()
```

## Adding New ML Models

1. Create model class in `src/models/`
2. Implement required methods:
   - `train(data)`
   - `predict(data)`
   - `save(path)`
   - `load(path)`
3. Add tests in `tests/`
4. Update API endpoints if needed
5. Document in README and ML_EXPLAINER

## Documentation

- Keep README.md up to date
- Add inline comments for complex logic
- Update API documentation for endpoint changes
- Create ADRs for significant decisions

## Questions?

- Open a discussion in GitHub Discussions
- Tag issues with `question` label
- Reach out to maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
