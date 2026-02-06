# fraud-detection-analysis
A comprehensive fraud detection system that identifies anomalous time periods and classifies fraudulent user registrations using machine learning

## Project Overview

This project addresses two key fraud detection challenges:

Task 1: Anomaly Detection - Identifies specific time periods with unusually high fraud concentration
Task 2: Fraud Classification - builds a binary classifier to predict fraudulent user registrations

## Project Structure
```
fraud-detection-analysis/
├── README.md                           # This file
├── requirements.txt                    # python dependencies
├── data/
│   └── dataset.csv                     # input dataset(not tracked in git)
├── src/
│   ├── __init__.py
│   ├── utils.py                        # helpers
│   ├── preprocessing.py                # data preprocessing
│   ├── feature_engineering.py          # feature creation and encoding
│   ├── model_trainer.py                # XGBoost model training
│   ├── evaluator.py                    # Model evaluation and visualization
│   └── fraud_classifier.py             # main orchestrator class
    └── run_classifier.py               # run the classifier

├── notebooks/
│   ├── task1_anomaly_detection.ipynb   # Task 1: Anomaly detection analysis
│   └── task2_fraud_classification.ipynb # Task 2: Classification model
└── outputs/
    ├── plots/                          # generated visualizations
```

## Data

The dataset is not included in this repository due to privacy 
Place your dataset.csv (Dataset Home Task Data Scientist.csv) file in the data/ folder

**To run this analysis:**

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

1. Clone the repository**

git clone https://github.com/monahibrahim/fraud-detection-analysis.git
cd fraud-detection-analysis

2. Create and activate virtual environment**

# Create virtual environment
python3.11 -m venv venv

# Ativate virtual environment
source venv/bin/activate

3. Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

### Running the Analysis

#### Option 1: Using Jupyter Notebooks
# Launch Jupyter
jupyter notebook

# Then open and run in order:
# 1. notebooks/task1_anomaly_detection.ipynb
# 2. notebooks/task2_fraud_classification.ipynb

#### Option 2: Using Python Scripts with src/ classes

In your terminal run: python run_classifier.py

## Key Findings

### Task 1: Anomaly Detection

- Anomalous Period Identified: January 11, 2025
- Duration:1 day
- Fraud Rate During Anomaly: ~26% 
- Detection Method: z-score statistical analysis (2.5 std threshold)

### Task 2: Fraud Classification

Model Performance:
- ROC-AUC 0.9586 (perfect discrimination)
- Recall 84% (catching 84% of fraudsters)
- Precision 42% (58% false alarm rate)

Top Predictive Features
- education level, email domain, job title, OS, registration duration

Business impact:
- Successfully identifies 420 out of 500 fraudsters
- 80 fraudsters slip through 
- 577 false alarms require manual review
- Tradeoff acceptable for fraud prevention

## Technical Details

### Data Preprocessing

- Missing values: filled with 'Unknown' category in education_level
- Feature removal: eliminated 4 non-predictive features : smoker, drinker, business hours, very fast registration
- Feature engineering created temporal features: hour, day_of_week, is_weekend, is_night_registration
- Encoding: label encoding for categorical variables

### Model Configuration

- Algorithm XGBoost (Extreme Gradient Boosting)
- Class imbalance handling: scale_pos_weight = 19:1
- Hyperparameters:
  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1
- train/test Split: 80/20 stratified split

### Evaluation Metrics

- Confusion matrix: visual breakdown of predictions
- ROC curve: model discrimination ability
- Precision-Recall curve: performance on imbalanced data
- Feature importance: XGBoost feature rankings

## Outputs

All results are saved in the outputs/ directory:

### Visualizations (outputs/plots/)
- fraud_rate.png - Class balance visualization
- all_features_distribution.png - Feature distributions
- fraud_pattern_analysis.png - Fraud rates by feature
- model_evaluation.png - Complete model evaluation
- daily_fraud_rate_timeline.png - Anomaly detection timeline

## Methodology

### Task 1: Anomaly Detection

Approach:
1. Aggregate fraud rate by day
2. Calculate z-scores for daily fraud rates
3. Flag days exceeding 2.5 std above mean
4. Validate with temporal visualizations

Alternative Methods Discussed:
- Isolation Forest
- Statistical time series monitoring
- Clustering based anomaly detection

### Task 2: Fraud Classification

Pipeline:
1. Data preprocessing handle missing values, remove non-predictive features
2. Feature engineering ereate temporal features, encode categoricals
3. Model training XGBoost with class imbalance handling
4. Evaluation comprehensive metrics and visualizations

## Future Improvements

With More Time:
- Hyperparameter tuning (GridSearchCV/RandomSearchCV)
- Cross-validation for robust performance estimates
- Ensemble methods (stacking multiple models)
- Threshold optimization based on business costs
- Deep learning approaches (LSTM for temporal sequences)

## Requirements
See requirements.txt for complete list. 
Key dependencies:
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost==1.7.6
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0