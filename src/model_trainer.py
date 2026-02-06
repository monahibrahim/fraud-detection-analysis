"""
Model training module for fraud detection.
Handles train-test split and XGBoost model training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import logging
from typing import Tuple, Dict, Any

#Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains XGBoost model for fraud detection
    
    Attributes:
    X_train : pd.DataFrame, training features
    X_test : pd.DataFrame, test features
    y_train : pd.Series, training target
    y_test : pd.Series, test target
    model : XGBClassifier, trained XGBoost model
    scale_pos_weight : float, weight for positive class to handle imbalance
    """
    
    def __init__(self):
        """
        Initialize model trainer.
        """
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scale_pos_weight = None
        logger.info("Initialized ModelTrainer")
    
    def prepare_train_test_split(self, 
                                 X: pd.DataFrame, 
                                 y: pd.Series,
                                 test_size: float = 0.2,
                                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        split data into training and test sets with stratification
        
        Parameters:
        X: pd.DataFrame, eature matrix
        y: pd.Series, target variable
        test_size: float, proportion of data for test set
        random_state : int, random seed for reproducibility
   
        Returns:
        (X_train, X_test, y_train, y_test)
        """
        try:
            logger.info(f"splitting data: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y  
            )
            
            #class imbalance
            train_fraud = self.y_train.sum()
            train_legit = len(self.y_train) - train_fraud
            self.scale_pos_weight = train_legit / train_fraud if train_fraud > 0 else 1.0
            
            logger.info(f"class imbalance ratio: {self.scale_pos_weight:.2f}:1")
            
            return self.X_train, self.X_test, self.y_train, self.y_test
            
        except Exception as e:
            logger.error(f"Error in train-test split: {str(e)}")
            raise
    
    def train_model(self,
                   n_estimators: int = 100,
                   max_depth: int = 6,
                   learning_rate: float = 0.1,
                   random_state: int = 42) -> XGBClassifier:
        """
        train XGBoost classifier
        
        Parameters:
        n_estimators: int, number of boosting rounds
        max_depth: int, maximum tree depth
        learning_rate: float, learning rate
        random_state: int, random seed
            
        Returns:
        XGBClassifier: trained XGBoost model
        """
        try:
            if self.X_train is None or self.y_train is None:
                raise ValueError("must run prepare_train_test_split() first")
            
            logger.info("Training XGBoost model")
            
            self.model = XGBClassifier(
                scale_pos_weight=self.scale_pos_weight,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                eval_metric='logloss'
            )
            
            self.model.fit(self.X_train, self.y_train)            
            logger.info(f"Training complete!")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error training XGBoost: {str(e)}")
            raise
    
    def get_model(self) -> XGBClassifier:
        """
        get the trained model
        Returns:
        XGBClassifier, trained model
        """
        if self.model is None:
            logger.warning("No model has been trained yet")
        return self.model
    
    def get_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        get train and test datasets
        
        Returns:(X_train, X_test, y_train, y_test)
        """
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_training_info(self) -> Dict[str, Any]:
        """
        get training information and statistics
        
        Returns:dict, dictionary with training metadata
        """
        if self.X_train is None:
            return {}
        
        return {
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'n_features': self.X_train.shape[1],
            'scale_pos_weight': self.scale_pos_weight,
            'train_fraud_rate': self.y_train.mean(),
            'test_fraud_rate': self.y_test.mean()
        }