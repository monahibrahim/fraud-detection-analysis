"""
Simple script to run fraud classification using the FraudClassifier class
"""
from src.fraud_classifier import FraudClassifier

features_to_drop = [
    'is_very_fast_registration',
    'is_smoker', 
    'is_drinker',
    'is_business_hours', 
    'user_id'
]

categorical_features = [
    'email_domain', 
    'gender', 
    'job_title',
    'education_level', 
    'os', 
    'device_type'
]

numerical_features = ['age', 'registration_duration_ms']

classifier = FraudClassifier(data_path='data/Dataset Home Task Data Scientist.csv')

results = classifier.run_full_pipeline(
    features_to_drop=features_to_drop,
    categorical_features=categorical_features,
    numerical_features=numerical_features,
    test_size=0.2,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    save_plots=True
)

print(f"ROC-AUC:{results['metrics']['roc_auc']:.4f}")
print(f"Recall:{results['metrics']['recall']:.4f}")
print(f"Precision:{results['metrics']['precision']:.4f}")
print(f"F1-Score:{results['metrics']['f1_score']:.4f}")
print("pipeline complete!")