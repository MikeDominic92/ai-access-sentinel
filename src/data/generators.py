"""
Synthetic IAM data generator.

Generates realistic IAM access logs with both normal and anomalous patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import random
from faker import Faker

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)


class IAMDataGenerator:
    """Generate synthetic IAM access logs for testing and training."""

    def __init__(self):
        self.departments = [
            'Engineering', 'Finance', 'HR', 'Sales', 'Marketing',
            'Operations', 'Legal', 'IT', 'Product', 'Customer Support'
        ]

        self.job_titles = {
            'Engineering': ['Software Engineer', 'Senior Engineer', 'Engineering Manager', 'DevOps Engineer'],
            'Finance': ['Financial Analyst', 'Accountant', 'Finance Manager', 'Controller'],
            'HR': ['HR Specialist', 'Recruiter', 'HR Manager', 'HR Director'],
            'Sales': ['Sales Rep', 'Account Executive', 'Sales Manager', 'VP Sales'],
            'Marketing': ['Marketing Specialist', 'Content Creator', 'Marketing Manager', 'CMO'],
            'Operations': ['Operations Analyst', 'Operations Manager', 'COO'],
            'Legal': ['Legal Counsel', 'Paralegal', 'General Counsel'],
            'IT': ['IT Support', 'System Admin', 'IT Manager', 'CISO'],
            'Product': ['Product Manager', 'Senior PM', 'VP Product'],
            'Customer Support': ['Support Agent', 'Support Lead', 'Support Manager']
        }

        self.resources = {
            'low_sensitivity': [
                'employee_directory', 'public_wiki', 'time_tracker',
                'vacation_calendar', 'company_news', 'learning_portal'
            ],
            'medium_sensitivity': [
                'crm_system', 'project_management', 'analytics_dashboard',
                'source_code_repo', 'marketing_platform', 'sales_tools'
            ],
            'high_sensitivity': [
                'financial_database', 'payroll_system', 'customer_pii',
                'production_database', 'admin_panel', 'security_logs',
                'aws_console', 'payment_processor'
            ]
        }

        self.actions = ['read', 'write', 'delete', 'admin', 'execute', 'download']

        self.locations = [
            'New York, US', 'San Francisco, US', 'London, UK', 'Singapore, SG',
            'Toronto, CA', 'Sydney, AU', 'Berlin, DE', 'Tokyo, JP',
            'Mumbai, IN', 'São Paulo, BR'
        ]

        self.anomalous_locations = [
            'Moscow, RU', 'Beijing, CN', 'Lagos, NG', 'Tehran, IR',
            'Pyongyang, KP'
        ]

    def generate_users(self, num_users: int = 200) -> pd.DataFrame:
        """Generate user profiles."""
        users = []

        for i in range(num_users):
            dept = random.choice(self.departments)
            user = {
                'user_id': f'U{i+1:04d}',
                'username': fake.user_name(),
                'email': fake.email(),
                'department': dept,
                'job_title': random.choice(self.job_titles[dept]),
                'seniority': random.choice(['Junior', 'Mid', 'Senior', 'Lead', 'Manager']),
                'hire_date': fake.date_between(start_date='-5y', end_date='today'),
                'primary_location': random.choice(self.locations),
                'is_contractor': random.random() < 0.1
            }
            users.append(user)

        return pd.DataFrame(users)

    def generate_normal_access_pattern(
        self,
        user: Dict,
        start_date: datetime,
        num_events: int = 50
    ) -> List[Dict]:
        """Generate normal access patterns for a user."""
        events = []

        # Determine typical work hours based on location
        work_start = 9
        work_end = 17

        # Resources accessible based on department
        dept_resources = {
            'Engineering': ['source_code_repo', 'analytics_dashboard', 'project_management'],
            'Finance': ['financial_database', 'analytics_dashboard', 'payroll_system'],
            'HR': ['payroll_system', 'employee_directory', 'crm_system'],
            'Sales': ['crm_system', 'sales_tools', 'analytics_dashboard'],
            'Marketing': ['marketing_platform', 'analytics_dashboard', 'crm_system'],
            'Operations': ['analytics_dashboard', 'project_management', 'time_tracker'],
            'Legal': ['employee_directory', 'public_wiki', 'project_management'],
            'IT': ['admin_panel', 'security_logs', 'aws_console', 'source_code_repo'],
            'Product': ['analytics_dashboard', 'project_management', 'source_code_repo'],
            'Customer Support': ['crm_system', 'employee_directory', 'time_tracker']
        }

        # Add some low sensitivity resources everyone can access
        accessible_resources = (
            dept_resources.get(user['department'], ['employee_directory']) +
            random.sample(self.resources['low_sensitivity'], 2)
        )

        for _ in range(num_events):
            # Generate timestamp during work hours, weekdays
            days_offset = random.randint(0, 90)
            event_date = start_date + timedelta(days=days_offset)

            # Skip weekends for most events
            if event_date.weekday() < 5 or random.random() < 0.1:
                hour = random.randint(work_start, work_end)
                minute = random.randint(0, 59)
                timestamp = event_date.replace(hour=hour, minute=minute, second=random.randint(0, 59))

                resource = random.choice(accessible_resources)

                # Most actions are read
                action_weights = {'read': 0.7, 'write': 0.2, 'execute': 0.08, 'download': 0.02}
                action = random.choices(
                    list(action_weights.keys()),
                    weights=list(action_weights.values())
                )[0]

                event = {
                    'user_id': user['user_id'],
                    'username': user['username'],
                    'department': user['department'],
                    'job_title': user['job_title'],
                    'resource': resource,
                    'action': action,
                    'timestamp': timestamp,
                    'source_ip': f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                    'location': user['primary_location'],
                    'success': True,
                    'is_anomaly': False,
                    'anomaly_type': None
                }
                events.append(event)

        return events

    def generate_anomalous_access_pattern(
        self,
        user: Dict,
        start_date: datetime,
        num_events: int = 5
    ) -> List[Dict]:
        """Generate anomalous access patterns."""
        events = []
        anomaly_types = [
            'unusual_time', 'unusual_location', 'privilege_escalation',
            'impossible_travel', 'unusual_resource', 'excessive_access'
        ]

        for _ in range(num_events):
            anomaly_type = random.choice(anomaly_types)
            days_offset = random.randint(0, 90)
            event_date = start_date + timedelta(days=days_offset)

            if anomaly_type == 'unusual_time':
                # Access at odd hours (2-6 AM)
                hour = random.randint(2, 6)
                timestamp = event_date.replace(hour=hour, minute=random.randint(0, 59))
                resource = random.choice(self.resources['high_sensitivity'])
                location = user['primary_location']

            elif anomaly_type == 'unusual_location':
                # Access from suspicious location
                hour = random.randint(9, 17)
                timestamp = event_date.replace(hour=hour, minute=random.randint(0, 59))
                resource = random.choice(self.resources['medium_sensitivity'] + self.resources['high_sensitivity'])
                location = random.choice(self.anomalous_locations)

            elif anomaly_type == 'privilege_escalation':
                # Access to admin resources
                hour = random.randint(9, 17)
                timestamp = event_date.replace(hour=hour, minute=random.randint(0, 59))
                resource = random.choice(['admin_panel', 'aws_console', 'security_logs'])
                location = user['primary_location']

            elif anomaly_type == 'impossible_travel':
                # Two accesses from far locations in short time
                hour = random.randint(9, 17)
                timestamp = event_date.replace(hour=hour, minute=random.randint(0, 59))
                resource = random.choice(self.resources['medium_sensitivity'])
                location = random.choice([loc for loc in self.locations if loc != user['primary_location']])

            elif anomaly_type == 'unusual_resource':
                # Access to resource not typical for department
                hour = random.randint(9, 17)
                timestamp = event_date.replace(hour=hour, minute=random.randint(0, 59))
                resource = random.choice(self.resources['high_sensitivity'])
                location = user['primary_location']

            else:  # excessive_access
                # Many rapid accesses
                hour = random.randint(9, 17)
                timestamp = event_date.replace(hour=hour, minute=random.randint(0, 59))
                resource = random.choice(self.resources['medium_sensitivity'])
                location = user['primary_location']

            # Anomalous actions more likely to be risky
            action = random.choice(['write', 'delete', 'admin', 'download'])

            event = {
                'user_id': user['user_id'],
                'username': user['username'],
                'department': user['department'],
                'job_title': user['job_title'],
                'resource': resource,
                'action': action,
                'timestamp': timestamp,
                'source_ip': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
                'location': location,
                'success': random.random() < 0.85,  # Some fail
                'is_anomaly': True,
                'anomaly_type': anomaly_type
            }
            events.append(event)

        return events

    def generate_complete_dataset(
        self,
        num_users: int = 200,
        normal_events_per_user: int = 50,
        anomaly_ratio: float = 0.05,
        output_path: str = 'data/sample_iam_logs.csv'
    ) -> pd.DataFrame:
        """
        Generate complete IAM dataset with normal and anomalous patterns.

        Args:
            num_users: Number of users to generate
            normal_events_per_user: Normal events per user
            anomaly_ratio: Ratio of anomalous events (5% default)
            output_path: Path to save CSV file

        Returns:
            DataFrame with all events
        """
        print(f"Generating {num_users} users...")
        users_df = self.generate_users(num_users)

        print("Generating access events...")
        all_events = []
        start_date = datetime.now() - timedelta(days=90)

        for _, user in users_df.iterrows():
            user_dict = user.to_dict()

            # Generate normal events
            normal_events = self.generate_normal_access_pattern(
                user_dict, start_date, normal_events_per_user
            )
            all_events.extend(normal_events)

            # Generate anomalous events for some users
            if random.random() < 0.3:  # 30% of users have anomalies
                num_anomalies = int(normal_events_per_user * anomaly_ratio) + 1
                anomalous_events = self.generate_anomalous_access_pattern(
                    user_dict, start_date, num_anomalies
                )
                all_events.extend(anomalous_events)

        # Create DataFrame
        df = pd.DataFrame(all_events)

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Add event ID
        df.insert(0, 'event_id', [f'E{i+1:06d}' for i in range(len(df))])

        # Save to file
        df.to_csv(output_path, index=False)

        print(f"\nDataset generated successfully!")
        print(f"Total events: {len(df):,}")
        print(f"Normal events: {len(df[~df['is_anomaly']]):,}")
        print(f"Anomalous events: {len(df[df['is_anomaly']]):,}")
        print(f"Anomaly ratio: {len(df[df['is_anomaly']]) / len(df) * 100:.2f}%")
        print(f"Saved to: {output_path}")

        return df


if __name__ == "__main__":
    generator = IAMDataGenerator()
    df = generator.generate_complete_dataset()

    # Display sample
    print("\nSample of generated data:")
    print(df.head(10))

    print("\nAnomaly types distribution:")
    print(df[df['is_anomaly']]['anomaly_type'].value_counts())
