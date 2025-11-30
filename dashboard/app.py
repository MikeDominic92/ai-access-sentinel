"""
Streamlit dashboard for AI Access Sentinel.

Interactive visualization of IAM analytics and ML model results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.anomaly_detector import AnomalyDetector, EnsembleAnomalyDetector
from src.models.access_predictor import AccessPredictor
from src.models.role_miner import RoleMiner
from src.models.risk_scorer import RiskScorer
from src.data.preprocessors import IAMDataPreprocessor
from src.data.generators import IAMDataGenerator

# Page config
st.set_page_config(
    page_title="AI Access Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache IAM data."""
    data_path = 'data/sample_iam_logs.csv'

    if not os.path.exists(data_path):
        st.warning("Generating sample data...")
        generator = IAMDataGenerator()
        df = generator.generate_complete_dataset(
            num_users=200,
            normal_events_per_user=50,
            output_path=data_path
        )
    else:
        df = pd.read_csv(data_path)

    # Preprocess
    preprocessor = IAMDataPreprocessor()
    df = preprocessor.preprocess_for_training(df)

    return df


@st.cache_resource
def load_models(_df):
    """Load and cache ML models."""
    models = {}

    # Anomaly detector
    model_path = 'models/trained/anomaly_detector_if.joblib'
    if os.path.exists(model_path):
        detector = AnomalyDetector()
        detector.load(model_path)
    else:
        st.info("Training anomaly detector...")
        detector = AnomalyDetector('isolation_forest')
        preprocessor = IAMDataPreprocessor()
        feature_cols = preprocessor.get_feature_columns()
        X = _df[feature_cols].values
        detector.train(X, feature_names=feature_cols)

    models['anomaly_detector'] = detector

    # Risk scorer
    models['risk_scorer'] = RiskScorer()

    return models


def main():
    """Main dashboard application."""

    # Header
    st.title("🛡️ AI Access Sentinel")
    st.markdown("**ML-Powered IAM Anomaly Detection and Governance**")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Anomaly Detection", "Risk Scoring", "Role Mining", "Access Prediction"]
    )

    # Load data and models
    df = load_data()
    models = load_models(df)

    if page == "Overview":
        show_overview(df, models)
    elif page == "Anomaly Detection":
        show_anomaly_detection(df, models)
    elif page == "Risk Scoring":
        show_risk_scoring(df, models)
    elif page == "Role Mining":
        show_role_mining(df)
    elif page == "Access Prediction":
        show_access_prediction(df)


def show_overview(df, models):
    """Display overview dashboard."""
    st.header("📊 System Overview")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Events", f"{len(df):,}")

    with col2:
        anomaly_count = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
        st.metric("Anomalies Detected", f"{anomaly_count:,}")

    with col3:
        unique_users = df['user_id'].nunique()
        st.metric("Active Users", f"{unique_users:,}")

    with col4:
        unique_resources = df['resource'].nunique()
        st.metric("Resources", f"{unique_resources:,}")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Access Events Over Time")
        # Group by date
        if 'timestamp' in df.columns:
            df_temp = df.copy()
            df_temp['date'] = pd.to_datetime(df_temp['timestamp']).dt.date
            daily_events = df_temp.groupby('date').size().reset_index(name='count')

            fig = px.line(daily_events, x='date', y='count', title='Daily Access Events')
            fig.update_traces(line_color='#1f77b4')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Events by Department")
        if 'department' in df.columns:
            dept_counts = df['department'].value_counts().head(10)
            fig = px.bar(
                x=dept_counts.index,
                y=dept_counts.values,
                labels={'x': 'Department', 'y': 'Event Count'},
                title='Top Departments by Access Volume'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Resource access heatmap
    st.subheader("Access Pattern Heatmap")

    if 'hour' in df.columns and 'day_of_week' in df.columns:
        heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='count').fillna(0)

        # Map day numbers to names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot.index = [day_names[int(i)] if i < len(day_names) else i for i in heatmap_pivot.index]

        fig = px.imshow(
            heatmap_pivot,
            labels=dict(x="Hour of Day", y="Day of Week", color="Access Count"),
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            title="Access Patterns by Day and Hour",
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)


def show_anomaly_detection(df, models):
    """Display anomaly detection dashboard."""
    st.header("🚨 Anomaly Detection")

    detector = models['anomaly_detector']

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        preprocessor = IAMDataPreprocessor()
        feature_cols = preprocessor.get_feature_columns()
        X = df[feature_cols].values
        predictions = detector.predict(X)
        anomaly_count = (predictions == -1).sum()
        st.metric("Anomalies Detected", f"{anomaly_count:,}")

    with col2:
        anomaly_ratio = (anomaly_count / len(df)) * 100
        st.metric("Anomaly Rate", f"{anomaly_ratio:.2f}%")

    with col3:
        scores = detector.score_samples(X)
        high_risk_count = (scores < -0.3).sum()
        st.metric("High Risk Events", f"{high_risk_count:,}")

    st.markdown("---")

    # Anomaly score distribution
    st.subheader("Anomaly Score Distribution")

    scores = detector.score_samples(X)
    fig = px.histogram(
        x=scores,
        nbins=50,
        labels={'x': 'Anomaly Score', 'y': 'Count'},
        title='Distribution of Anomaly Scores (lower = more anomalous)'
    )
    fig.add_vline(x=-0.3, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
    st.plotly_chart(fig, use_container_width=True)

    # Recent anomalies table
    st.subheader("Recent Anomalies")

    df_with_scores = df.copy()
    df_with_scores['anomaly_score'] = scores
    df_with_scores['is_detected_anomaly'] = predictions == -1

    anomalies = df_with_scores[df_with_scores['is_detected_anomaly']].sort_values('anomaly_score').head(20)

    if len(anomalies) > 0:
        display_cols = ['user_id', 'resource', 'action', 'anomaly_score']
        if 'timestamp' in anomalies.columns:
            display_cols.insert(0, 'timestamp')
        if 'location' in anomalies.columns:
            display_cols.append('location')

        st.dataframe(anomalies[display_cols], use_container_width=True)
    else:
        st.info("No anomalies detected")


def show_risk_scoring(df, models):
    """Display risk scoring dashboard."""
    st.header("⚠️ User Risk Scoring")

    scorer = models['risk_scorer']

    # Calculate risk scores for all users
    with st.spinner("Calculating risk scores..."):
        risk_scores = scorer.calculate_batch_risk_scores(df)

    # Top risk users
    st.subheader("Highest Risk Users")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Bar chart
        top_users = risk_scores.head(15)
        fig = px.bar(
            top_users,
            x='user_id',
            y='risk_score',
            color='risk_level',
            title='Top 15 Users by Risk Score',
            labels={'risk_score': 'Risk Score', 'user_id': 'User ID'},
            color_discrete_map={
                'CRITICAL': '#d62728',
                'HIGH': '#ff7f0e',
                'MEDIUM': '#ffbb00',
                'LOW': '#2ca02c',
                'MINIMAL': '#1f77b4'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk level distribution
        risk_dist = risk_scores['risk_level'].value_counts()
        fig = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            title='Risk Level Distribution',
            color=risk_dist.index,
            color_discrete_map={
                'CRITICAL': '#d62728',
                'HIGH': '#ff7f0e',
                'MEDIUM': '#ffbb00',
                'LOW': '#2ca02c',
                'MINIMAL': '#1f77b4'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    # User details
    st.subheader("User Risk Details")

    selected_user = st.selectbox("Select User", risk_scores['user_id'].tolist())

    if selected_user:
        user_risk = risk_scores[risk_scores['user_id'] == selected_user].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Risk Score", f"{user_risk['risk_score']:.2f}")

        with col2:
            st.metric("Risk Level", user_risk['risk_level'])

        with col3:
            total_events = len(df[df['user_id'] == selected_user])
            st.metric("Total Events", total_events)

        # Factor breakdown
        st.subheader("Risk Factor Breakdown")

        factors = user_risk['factor_scores']
        fig = go.Figure(go.Bar(
            x=list(factors.values()),
            y=list(factors.keys()),
            orientation='h',
            marker=dict(color='#ff7f0e')
        ))
        fig.update_layout(
            title='Risk Factors',
            xaxis_title='Score',
            yaxis_title='Factor',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.subheader("Recommendations")
        for rec in user_risk['recommendations']:
            st.info(f"• {rec}")


def show_role_mining(df):
    """Display role mining dashboard."""
    st.header("👥 Role Mining")

    # Parameters
    col1, col2 = st.columns([1, 3])

    with col1:
        n_clusters = st.slider("Number of Roles", min_value=3, max_value=15, value=8)

    # Train role miner
    with st.spinner("Discovering roles..."):
        miner = RoleMiner(n_clusters=n_clusters)
        miner.train(df, auto_tune_clusters=False)

    # Role summary
    role_summary = miner.get_role_summary()

    st.subheader("Discovered Roles")

    col1, col2 = st.columns(2)

    with col1:
        # Role size distribution
        fig = px.bar(
            role_summary,
            x='role_id',
            y='user_count',
            color='department',
            title='Users per Role',
            labels={'user_count': 'User Count', 'role_id': 'Role ID'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Department distribution
        dept_dist = role_summary['department'].value_counts()
        fig = px.pie(
            values=dept_dist.values,
            names=dept_dist.index,
            title='Roles by Department'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Role details table
    st.subheader("Role Details")
    st.dataframe(role_summary, use_container_width=True)

    # Role health
    st.subheader("Role Health Analysis")

    health = miner.detect_role_explosion()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Roles", health['total_roles'])

    with col2:
        st.metric("Avg Users per Role", f"{health['avg_users_per_role']:.1f}")

    with col3:
        st.metric("Small Roles", health['small_roles_count'])

    if health['is_role_explosion']:
        st.warning(f"⚠️ {health['recommendation']}")
    else:
        st.success(f"✅ {health['recommendation']}")


def show_access_prediction(df):
    """Display access prediction interface."""
    st.header("🔮 Access Request Prediction")

    st.write("Predict whether an access request should be approved based on peer analysis and ML models.")

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        department = st.selectbox("Department", df['department'].unique().tolist() if 'department' in df.columns else ['Engineering'])
        job_title = st.selectbox("Job Title", df['job_title'].unique().tolist() if 'job_title' in df.columns else ['Software Engineer'])

    with col2:
        resource = st.selectbox("Resource", df['resource'].unique().tolist())
        action = st.selectbox("Action", df['action'].unique().tolist())

    if st.button("Predict Access"):
        # Train predictor if needed
        with st.spinner("Analyzing..."):
            predictor = AccessPredictor()
            predictor.train(df)

            user_info = {
                'department': department,
                'job_title': job_title,
                'user_total_events': 100,
                'user_unique_resources': 10,
                'user_success_rate': 0.95,
                'resource_sensitivity_score': 2,
                'action_risk_score': 2,
                'combined_risk_score': 4,
                'is_business_hours': 1,
                'is_suspicious_location': 0
            }

            result = predictor.predict_access(user_info, resource, action)

        # Display result
        st.markdown("---")
        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            approval_status = "✅ APPROVE" if result['should_approve'] else "❌ DENY"
            st.markdown(f"### {approval_status}")

        with col2:
            st.metric("Confidence", f"{result['confidence']:.1%}")

        with col3:
            st.metric("Recommendation", result['recommendation'])

        # Probability breakdown
        st.subheader("Probability Breakdown")

        fig = go.Figure(data=[
            go.Bar(
                x=['Approve', 'Deny'],
                y=[result['probability_approve'], result['probability_deny']],
                marker_color=['green', 'red']
            )
        ])
        fig.update_layout(
            title='Approval Probabilities',
            yaxis_title='Probability',
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
