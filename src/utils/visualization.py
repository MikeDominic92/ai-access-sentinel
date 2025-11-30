"""
Visualization utilities for IAM analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict, Optional


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def plot_anomaly_score_distribution(scores: np.ndarray, threshold: float = -0.3):
    """
    Plot distribution of anomaly scores.

    Args:
        scores: Anomaly scores from detector
        threshold: Threshold for high-risk classification
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'High Risk Threshold ({threshold})')

    ax.set_xlabel('Anomaly Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Anomaly Scores\n(Lower = More Anomalous)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_risk_scores_by_user(risk_df: pd.DataFrame, top_n: int = 20):
    """
    Plot top users by risk score.

    Args:
        risk_df: DataFrame with user_id and risk_score columns
        top_n: Number of top users to display
    """
    top_users = risk_df.nlargest(top_n, 'risk_score')

    fig = px.bar(
        top_users,
        x='user_id',
        y='risk_score',
        color='risk_level' if 'risk_level' in top_users.columns else None,
        title=f'Top {top_n} Users by Risk Score',
        labels={'risk_score': 'Risk Score', 'user_id': 'User ID'},
        color_discrete_map={
            'CRITICAL': '#d62728',
            'HIGH': '#ff7f0e',
            'MEDIUM': '#ffbb00',
            'LOW': '#2ca02c',
            'MINIMAL': '#1f77b4'
        }
    )

    fig.update_layout(xaxis_tickangle=-45)
    return fig


def plot_access_heatmap(df: pd.DataFrame):
    """
    Plot access pattern heatmap by day and hour.

    Args:
        df: DataFrame with hour and day_of_week columns
    """
    if 'hour' not in df.columns or 'day_of_week' not in df.columns:
        raise ValueError("DataFrame must have 'hour' and 'day_of_week' columns")

    # Create pivot table
    heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='count').fillna(0)

    # Map day numbers to names
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    heatmap_pivot.index = [day_names[int(i)] if i < len(day_names) else i for i in heatmap_pivot.index]

    fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(
        heatmap_pivot,
        cmap='YlOrRd',
        annot=False,
        fmt='g',
        cbar_kws={'label': 'Access Count'},
        ax=ax
    )

    ax.set_title('Access Patterns by Day and Hour', fontsize=14)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Day of Week', fontsize=12)

    plt.tight_layout()
    return fig


def plot_role_clusters(user_features: pd.DataFrame, labels: np.ndarray, method: str = 'pca'):
    """
    Plot role clustering results.

    Args:
        user_features: User feature matrix
        labels: Cluster labels
        method: Dimensionality reduction method ('pca' or 'tsne')
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Reduce to 2D
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(user_features.values)
        title_suffix = f"(PCA - {reducer.explained_variance_ratio_.sum():.1%} variance)"
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(user_features)-1))
        coords = reducer.fit_transform(user_features.values)
        title_suffix = "(t-SNE)"

    # Create scatter plot
    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=labels.astype(str),
        labels={'x': 'Component 1', 'y': 'Component 2', 'color': 'Role'},
        title=f'Role Clustering Visualization {title_suffix}',
        hover_data={'user_id': user_features.index}
    )

    fig.update_traces(marker=dict(size=10, opacity=0.7))
    return fig


def plot_feature_importance(importances: pd.DataFrame, top_n: int = 15):
    """
    Plot feature importance from model.

    Args:
        importances: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
    """
    top_features = importances.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = None):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if labels is None:
        labels = ['Normal', 'Anomaly']

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_title('Confusion Matrix', fontsize=14)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    return fig


def plot_time_series_anomalies(df: pd.DataFrame, anomaly_col: str = 'is_anomaly'):
    """
    Plot time series with anomalies highlighted.

    Args:
        df: DataFrame with timestamp and anomaly indicator
        anomaly_col: Column name for anomaly indicator
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column")

    df_sorted = df.sort_values('timestamp')

    # Aggregate by date
    df_sorted['date'] = pd.to_datetime(df_sorted['timestamp']).dt.date
    daily_counts = df_sorted.groupby('date').size()
    daily_anomalies = df_sorted[df_sorted[anomaly_col]].groupby('date').size()

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(daily_counts.index, daily_counts.values, label='Total Events',
            color='steelblue', linewidth=2)
    ax.plot(daily_anomalies.index, daily_anomalies.values, label='Anomalies',
            color='red', linewidth=2, marker='o')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Event Count', fontsize=12)
    ax.set_title('Access Events Over Time (with Anomalies)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_risk_factor_breakdown(factor_scores: Dict[str, float]):
    """
    Plot risk factor breakdown for a user.

    Args:
        factor_scores: Dictionary of factor names and scores
    """
    factors = list(factor_scores.keys())
    scores = list(factor_scores.values())

    fig = go.Figure(go.Bar(
        x=scores,
        y=factors,
        orientation='h',
        marker=dict(
            color=scores,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title='Score')
        )
    ))

    fig.update_layout(
        title='Risk Factor Breakdown',
        xaxis_title='Score (0-100)',
        yaxis_title='Risk Factor',
        height=400
    )

    return fig


def create_dashboard_summary(df: pd.DataFrame, risk_scores: pd.DataFrame):
    """
    Create summary dashboard with multiple plots.

    Args:
        df: Main IAM data
        risk_scores: User risk scores
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Event counts by department
    if 'department' in df.columns:
        dept_counts = df['department'].value_counts().head(10)
        axes[0, 0].bar(range(len(dept_counts)), dept_counts.values, color='steelblue')
        axes[0, 0].set_xticks(range(len(dept_counts)))
        axes[0, 0].set_xticklabels(dept_counts.index, rotation=45, ha='right')
        axes[0, 0].set_title('Events by Department')
        axes[0, 0].set_ylabel('Event Count')

    # Plot 2: Risk level distribution
    if 'risk_level' in risk_scores.columns:
        risk_dist = risk_scores['risk_level'].value_counts()
        colors = {'CRITICAL': 'red', 'HIGH': 'orange', 'MEDIUM': 'yellow',
                 'LOW': 'lightgreen', 'MINIMAL': 'green'}
        axes[0, 1].pie(risk_dist.values, labels=risk_dist.index,
                       autopct='%1.1f%%',
                       colors=[colors.get(level, 'gray') for level in risk_dist.index])
        axes[0, 1].set_title('Risk Level Distribution')

    # Plot 3: Top resources accessed
    if 'resource' in df.columns:
        resource_counts = df['resource'].value_counts().head(10)
        axes[1, 0].barh(range(len(resource_counts)), resource_counts.values, color='coral')
        axes[1, 0].set_yticks(range(len(resource_counts)))
        axes[1, 0].set_yticklabels(resource_counts.index)
        axes[1, 0].set_title('Top 10 Accessed Resources')
        axes[1, 0].set_xlabel('Access Count')
        axes[1, 0].invert_yaxis()

    # Plot 4: Action distribution
    if 'action' in df.columns:
        action_counts = df['action'].value_counts()
        axes[1, 1].bar(range(len(action_counts)), action_counts.values, color='teal')
        axes[1, 1].set_xticks(range(len(action_counts)))
        axes[1, 1].set_xticklabels(action_counts.index, rotation=45)
        axes[1, 1].set_title('Action Distribution')
        axes[1, 1].set_ylabel('Count')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Visualization utilities loaded successfully")
