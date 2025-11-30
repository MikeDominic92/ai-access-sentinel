"""
Script to generate sample IAM data.

Run this after installing dependencies:
  pip install -r requirements.txt
  python generate_data.py
"""

from src.data.generators import IAMDataGenerator

if __name__ == "__main__":
    print("Generating AI Access Sentinel sample data...")
    print("=" * 60)

    generator = IAMDataGenerator()

    # Generate main dataset
    df = generator.generate_complete_dataset(
        num_users=200,
        normal_events_per_user=50,
        anomaly_ratio=0.05,
        output_path='data/sample_iam_logs.csv'
    )

    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("\nNext steps:")
    print("1. Explore data: jupyter lab notebooks/01_data_exploration.ipynb")
    print("2. Train models: python -c 'from src.models.anomaly_detector import AnomalyDetector; ...'")
    print("3. Start API: uvicorn src.api.main:app --reload")
    print("4. Launch dashboard: streamlit run dashboard/app.py")
