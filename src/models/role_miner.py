"""
Role mining through clustering.

Discovers implicit roles by clustering users with similar access patterns.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib
from typing import Dict, List, Optional, Tuple


class RoleMiner:
    """
    Discover roles through access pattern clustering.

    Uses K-Means and hierarchical clustering to identify groups of users
    with similar access patterns.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        algorithm: str = 'kmeans',
        random_state: int = 42
    ):
        """
        Initialize role miner.

        Args:
            n_clusters: Number of roles to discover
            algorithm: 'kmeans' or 'hierarchical'
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.user_features = None
        self.cluster_profiles = None

    def create_user_access_matrix(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create user-resource access matrix.

        Args:
            df: IAM logs DataFrame

        Returns:
            Tuple of (user_matrix, user_ids)
        """
        # Create pivot: users vs resources (with access counts)
        access_matrix = df.pivot_table(
            index='user_id',
            columns='resource',
            values='event_id',  # or any column
            aggfunc='count',
            fill_value=0
        )

        # Also add action diversity per user
        action_diversity = df.groupby('user_id')['action'].nunique()
        action_diversity.name = 'action_diversity'

        # Department (one-hot encoded)
        if 'department' in df.columns:
            dept_dummies = pd.get_dummies(
                df.groupby('user_id')['department'].first(),
                prefix='dept'
            )
            access_matrix = access_matrix.join(dept_dummies)

        # Add action diversity
        access_matrix = access_matrix.join(action_diversity)

        # Normalize by total events (relative access pattern)
        total_events = access_matrix.sum(axis=1)
        normalized_matrix = access_matrix.div(total_events, axis=0).fillna(0)

        user_ids = normalized_matrix.index.tolist()

        return normalized_matrix, user_ids

    def find_optimal_clusters(
        self,
        X: np.ndarray,
        max_clusters: int = 15
    ) -> int:
        """
        Find optimal number of clusters using silhouette score.

        Args:
            X: Feature matrix
            max_clusters: Maximum clusters to try

        Returns:
            Optimal number of clusters
        """
        print("Finding optimal number of clusters...")

        scores = []
        K_range = range(2, min(max_clusters + 1, len(X)))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            scores.append(score)
            print(f"  k={k}: silhouette score = {score:.3f}")

        optimal_k = K_range[np.argmax(scores)]
        print(f"\nOptimal number of clusters: {optimal_k}")

        return optimal_k

    def train(
        self,
        df: pd.DataFrame,
        auto_tune_clusters: bool = False
    ) -> 'RoleMiner':
        """
        Train role mining model.

        Args:
            df: IAM logs DataFrame
            auto_tune_clusters: Whether to automatically find optimal k

        Returns:
            self for method chaining
        """
        print(f"Creating user access matrix...")

        # Create user-resource matrix
        user_matrix, user_ids = self.create_user_access_matrix(df)
        self.user_features = user_matrix

        # Scale features
        X = self.scaler.fit_transform(user_matrix.values)

        # Find optimal clusters if requested
        if auto_tune_clusters:
            self.n_clusters = self.find_optimal_clusters(X)

        print(f"\nTraining {self.algorithm} with {self.n_clusters} clusters...")

        # Create and train model
        if self.algorithm == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
        elif self.algorithm == 'hierarchical':
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Fit model
        labels = self.model.fit_predict(X)

        # Create cluster profiles
        self.cluster_profiles = self._create_cluster_profiles(
            user_matrix, labels, user_ids
        )

        self.is_trained = True
        print("Training complete!")

        return self

    def _create_cluster_profiles(
        self,
        user_matrix: pd.DataFrame,
        labels: np.ndarray,
        user_ids: List[str]
    ) -> Dict[int, Dict]:
        """
        Create profile for each discovered role.

        Args:
            user_matrix: User feature matrix
            labels: Cluster assignments
            user_ids: User IDs

        Returns:
            Dictionary of cluster profiles
        """
        profiles = {}

        for cluster_id in range(self.n_clusters):
            # Users in this cluster
            cluster_mask = labels == cluster_id
            cluster_users = [user_ids[i] for i, m in enumerate(cluster_mask) if m]

            # Get cluster data
            cluster_data = user_matrix[cluster_mask]

            # Find top resources (highest average access in cluster)
            resource_cols = [col for col in user_matrix.columns
                           if not col.startswith('dept_') and col != 'action_diversity']

            top_resources = cluster_data[resource_cols].mean().nlargest(5)

            # Find dominant department
            dept_cols = [col for col in user_matrix.columns if col.startswith('dept_')]
            if dept_cols:
                dept_dist = cluster_data[dept_cols].mean()
                dominant_dept = dept_dist.idxmax().replace('dept_', '') if len(dept_dist) > 0 else 'Mixed'
            else:
                dominant_dept = 'Unknown'

            # Generate role name
            role_name = self._generate_role_name(cluster_id, top_resources, dominant_dept)

            profiles[cluster_id] = {
                'role_id': f'R{cluster_id:03d}',
                'role_name': role_name,
                'user_count': len(cluster_users),
                'users': cluster_users,
                'top_resources': top_resources.to_dict(),
                'dominant_department': dominant_dept,
                'avg_action_diversity': cluster_data.get('action_diversity', pd.Series([0])).mean()
            }

        return profiles

    def _generate_role_name(
        self,
        cluster_id: int,
        top_resources: pd.Series,
        department: str
    ) -> str:
        """Generate descriptive role name."""
        # Map resources to role names
        role_keywords = {
            'source_code': 'Developer',
            'financial': 'Finance',
            'payroll': 'HR/Payroll',
            'crm': 'Sales/Customer',
            'analytics': 'Analyst',
            'admin': 'Administrator',
            'security': 'Security',
            'aws': 'DevOps/Cloud'
        }

        # Check top resources for keywords
        for keyword, role_type in role_keywords.items():
            if any(keyword in res.lower() for res in top_resources.index):
                return f"{department} {role_type}"

        # Default
        return f"{department} Role {cluster_id}"

    def predict_role(self, user_id: str) -> Optional[Dict]:
        """
        Predict role for a user.

        Args:
            user_id: User to classify

        Returns:
            Role information or None
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        if user_id not in self.user_features.index:
            return None

        # Get user features
        user_vec = self.user_features.loc[user_id].values.reshape(1, -1)
        user_vec_scaled = self.scaler.transform(user_vec)

        # Predict cluster
        if self.algorithm == 'kmeans':
            cluster_id = self.model.predict(user_vec_scaled)[0]
        else:
            # For hierarchical, need to refit or store labels
            # For simplicity, find closest cluster center
            cluster_id = 0  # Placeholder

        return self.cluster_profiles.get(cluster_id)

    def get_role_summary(self) -> pd.DataFrame:
        """
        Get summary of all discovered roles.

        Returns:
            DataFrame with role information
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        summaries = []
        for cluster_id, profile in self.cluster_profiles.items():
            summaries.append({
                'role_id': profile['role_id'],
                'role_name': profile['role_name'],
                'user_count': profile['user_count'],
                'department': profile['dominant_department'],
                'top_resource': list(profile['top_resources'].keys())[0] if profile['top_resources'] else None
            })

        return pd.DataFrame(summaries)

    def detect_role_explosion(self) -> Dict[str, any]:
        """
        Detect signs of role explosion (too many small roles).

        Returns:
            Analysis of role health
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        user_counts = [p['user_count'] for p in self.cluster_profiles.values()]

        # Role explosion indicators
        total_users = sum(user_counts)
        avg_users_per_role = np.mean(user_counts)
        small_roles = sum(1 for c in user_counts if c < 5)

        is_explosion = (
            self.n_clusters > 10 and
            small_roles > self.n_clusters * 0.4
        )

        return {
            'total_roles': self.n_clusters,
            'total_users': total_users,
            'avg_users_per_role': avg_users_per_role,
            'small_roles_count': small_roles,
            'small_roles_ratio': small_roles / self.n_clusters,
            'is_role_explosion': is_explosion,
            'recommendation': 'Consider role consolidation' if is_explosion else 'Role structure is healthy'
        }

    def save(self, path: str) -> None:
        """Save model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'n_clusters': self.n_clusters,
            'algorithm': self.algorithm,
            'user_features': self.user_features,
            'cluster_profiles': self.cluster_profiles,
            'is_trained': self.is_trained
        }, path)

        print(f"Model saved to {path}")

    def load(self, path: str) -> 'RoleMiner':
        """Load model from disk."""
        data = joblib.load(path)

        self.model = data['model']
        self.scaler = data['scaler']
        self.n_clusters = data['n_clusters']
        self.algorithm = data['algorithm']
        self.user_features = data['user_features']
        self.cluster_profiles = data['cluster_profiles']
        self.is_trained = data['is_trained']

        print(f"Model loaded from {path}")
        return self


if __name__ == "__main__":
    # Example usage
    from src.data.generators import IAMDataGenerator
    from src.data.preprocessors import IAMDataPreprocessor

    # Generate data
    print("Generating data for role mining...")
    generator = IAMDataGenerator()
    df = generator.generate_complete_dataset(
        num_users=200,
        normal_events_per_user=50,
        output_path='data/sample_iam_logs.csv'
    )

    # Preprocess
    print("\nPreprocessing...")
    preprocessor = IAMDataPreprocessor()
    df = preprocessor.preprocess_for_training(df)

    # Train role miner
    print("\n" + "="*50)
    print("Training Role Miner")
    print("="*50)

    miner = RoleMiner(n_clusters=8, algorithm='kmeans')
    miner.train(df, auto_tune_clusters=False)

    # Get role summary
    print("\n" + "="*50)
    print("Discovered Roles")
    print("="*50)
    print(miner.get_role_summary())

    # Check for role explosion
    print("\n" + "="*50)
    print("Role Health Analysis")
    print("="*50)
    health = miner.detect_role_explosion()
    for key, value in health.items():
        print(f"  {key}: {value}")

    # Save model
    miner.save('models/trained/role_miner.joblib')

    print("\nRole mining complete!")
