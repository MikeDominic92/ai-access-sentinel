# ADR-001: ML Algorithm Selection for IAM Anomaly Detection

## Status
Accepted

## Context
We need to select appropriate machine learning algorithms for detecting anomalies in IAM access patterns. The solution must:
- Handle high-dimensional feature spaces
- Detect various types of anomalies (point, contextual, collective)
- Work with limited labeled data (unsupervised/semi-supervised)
- Provide interpretable results for security teams
- Scale to handle thousands of users and millions of events

## Decision
We will implement a multi-algorithm approach with the following models:

### 1. Anomaly Detection: Isolation Forest (Primary)
**Rationale:**
- Excels at identifying outliers in high-dimensional data
- Fast training and prediction (O(n log n))
- Works well with unlabeled data
- Low memory footprint
- Effective for point anomalies

**Alternatives Considered:**
- One-Class SVM: Good but slower, chosen as secondary option
- Local Outlier Factor: Effective for density-based anomalies, included in ensemble
- Autoencoders: Require more data and tuning, overkill for current scale

### 2. Access Prediction: Random Forest Classifier
**Rationale:**
- Handles mixed categorical and numerical features well
- Provides feature importance for interpretability
- Robust to overfitting with proper tuning
- Can model complex non-linear relationships
- Works well with imbalanced data

**Alternatives Considered:**
- Gradient Boosting (XGBoost): More powerful but harder to tune
- Logistic Regression: Too simple for complex IAM patterns
- Neural Networks: Overkill for current data size

### 3. Role Mining: K-Means Clustering
**Rationale:**
- Simple, fast, and interpretable
- Works well for discovering natural groupings
- Scalable to large numbers of users
- Easy to visualize and explain to stakeholders

**Alternatives Considered:**
- Hierarchical Clustering: Included as option, better for role hierarchies
- DBSCAN: Good but requires careful parameter tuning
- Gaussian Mixture Models: More complex, not necessary for our use case

### 4. Risk Scoring: Ensemble Weighted Approach
**Rationale:**
- Combines multiple risk factors (anomalies, peer deviation, policy violations)
- Transparent and auditable
- Easy to adjust weights based on organizational priorities
- Provides clear breakdown for investigation

**Alternatives Considered:**
- Single ML model: Less interpretable, harder to explain
- Simple rule-based: Too rigid, can't adapt to patterns

## Consequences

### Positive
- Multiple algorithms provide redundancy and robustness
- Unsupervised approaches work without labeled data
- All selected algorithms are well-understood and proven
- Implementation using scikit-learn ensures stability and support
- Results are interpretable for security teams
- Performance is adequate for real-time analysis

### Negative
- Multiple models increase complexity
- Need to maintain and tune multiple algorithms
- Ensemble approaches require careful combination logic
- May need retraining as access patterns evolve

### Risks
- **False Positives**: Isolation Forest may flag legitimate unusual access
  - Mitigation: Tune contamination parameter, ensemble voting
- **Concept Drift**: Access patterns change over organizational changes
  - Mitigation: Implement periodic retraining, online learning in future
- **Feature Engineering**: Model quality depends on feature selection
  - Mitigation: Comprehensive feature extraction, domain expert validation

## Implementation Notes
- Start with default hyperparameters, tune based on validation
- Use cross-validation for supervised models
- Monitor model performance metrics over time
- Implement A/B testing for algorithm comparison
- Consider online learning for production deployment

## Future Considerations
- Deep learning (LSTM/Transformer) for sequence modeling
- Graph neural networks for access relationship analysis
- Federated learning for multi-tenant scenarios
- AutoML for automated hyperparameter tuning

## References
- scikit-learn documentation
- "Isolation Forest" (Liu et al., 2008)
- "Random Forests" (Breiman, 2001)
- "Role Mining" RBAC literature

## Date
2024-01-15

## Authors
Mike Dominic
