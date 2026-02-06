"""
Feature engineering module for fraud detection.
Handles temporal feature creation, encoding and feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
from typing import List, Dict, Tuple

#Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates and transforms features for fraud detection model
    Attributes:
    data : pd.DataFrame, input dataframe
    label_encoders : dict, dictionary storing label encoders for each categorical feature
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize feature engineer with data.
        
        Parameters:
        data : pd.DataFrame, input dataframe
        """
        self.data = data.copy()
        self.label_encoders = {}
    
    def create_temporal_features(self) -> pd.DataFrame:
        """
        Create temporal features from registration timestamp
        
        Returns:
        pd.DataFrame, dataframe with temporal features added
        """
        try:
            logger.info("Creating temporal features...")
            
            #time features
            self.data['hour'] = self.data['registration_timestamp'].dt.hour
            self.data['day_of_week'] = self.data['registration_timestamp'].dt.dayofweek
            
            #boolean
            self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6]).astype(int)
            self.data['is_night_registration'] = (
                (self.data['hour'] >= 22) | (self.data['hour'] <= 6)
            ).astype(int)
            
            logger.info("created features: hour, day_of_week, is_weekend, is_night_registration")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error creating temporal features: {str(e)}")
            raise
    
    def encode_categorical_features(self, categorical_columns: List[str]) -> pd.DataFrame:
        """
        encode categorical features using Label Encoding

        Parameters:
        categorical_columns : list, --list of categorical column names to encode
            
        Returns:
        pd.DataFrame, dataframe with encoded categorical features
        """
        try:
            logger.info(f"encoding {len(categorical_columns)} categorical features")
            
            for col in categorical_columns:
                if col not in self.data.columns:
                    logger.warning(f"column'{col}' not found in data, skipping")
                    continue
                
                #fit label encoder
                le = LabelEncoder()
                self.data[f'{col}_encoded'] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
                
                n_unique = self.data[col].nunique()
                logger.info(f"encoded '{col}' to '{col}_encoded'")
            
            logger.info(f"Encoding complete")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error encoding categorical features: {str(e)}")
            raise
    
    def select_features(self, 
                       numerical_features: List[str],
                       temporal_features: List[str],
                       categorical_features: List[str],
                       target: str = 'is_fraud') -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        select and prepare final feature set for modeling
        
        params::
        numerical_features : list, list of numerical feature names
        temporal_features : list, list of temporal feature names
        categorical_features : list, list of categorical feature names
        target : str, target variable name
            
        Returns:
        
        (X, y, feature_columns)-- fature matrix, target, and feature names
        """
        try:
            logger.info("Selecting final feature set")
            
            #feature column list
            feature_cols = (
                numerical_features + 
                temporal_features + 
                [f'{col}_encoded' for col in categorical_features]
            )
            
            #verify all features exist
            missing_features = [f for f in feature_cols if f not in self.data.columns]
            if missing_features:
                logger.error(f"Missing features: {missing_features}")
                raise ValueError(f"Features not found in data: {missing_features}")
            
            #prepare X and y
            X = self.data[feature_cols]
            y = self.data[target]
            
            logger.info(f"Total features selected: {len(feature_cols)}")
            
            return X, y, feature_cols
            
        except Exception as e:
            logger.error(f"Error selecting features: {str(e)}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """
        get list of all created feature names
        Returns:
        list --list of feature names in the current dataframe
        """
        return self.data.columns.tolist()
    
    def get_encoders(self) -> Dict[str, LabelEncoder]:
        """
        get dictionary of fitted label encoder
        Returns:
        dict, dictionary 
        """
        return self.label_encoders.copy()
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        get the processed dataframe with all engineered features
       
        Returns:
        pd.DataFrame, processed dataframe
        """
        return self.data.copy()