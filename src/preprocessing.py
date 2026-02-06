"""
Data preprocessing module for fraud detection.
Handles missing values, data cleaning, and initial data preparation.
"""

import pandas as pd
import numpy as np
import logging

#configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessing including missing values and data cleaning
    
    Attributes:
    data : pd.DataFrame
    Input dataframe to preprocess
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize preprocessor with data
        
        Parameters:
        data : pd.DataFrame
        Input dataframe
        """
        self.data = data.copy()
        logger.info(f"Initialized DataPreprocessor with {len(self.data):,} rows")
    
    def handle_missing_values(self, strategy: str = 'unknown') -> pd.DataFrame:
        """
        handle missing values in the dataset
        Parameters:
        strategy : str, strategy for handling missing values ('unknown', 'drop')
            
        Returns:
        pd.DataFrame, dataframe with missing values handled
        """
        try:
            logger.info("Handling missing values")
            
            missing_before = self.data.isnull().sum().sum()
            
            if strategy == 'unknown':
                #fill categorical missing values with 'Unknown'
                for col in self.data.select_dtypes(include=['object']).columns:
                    if self.data[col].isnull().sum() > 0:
                        missing_count = self.data[col].isnull().sum()
                        self.data[col] = self.data[col].fillna('Unknown')
                        logger.info(f"Filled {missing_count} missing values in '{col}' with 'Unknown'")
            
            elif strategy == 'drop':
                original_len = len(self.data)
                self.data = self.data.dropna()
                dropped = original_len - len(self.data)
                logger.info(f"dropped {dropped} rows with missing values")
            
            missing_after = self.data.isnull().sum().sum()
            logger.info(f"issing values: {missing_before} → {missing_after}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def check_data_quality(self) -> dict:
        """
        Data quality checks
        
        Returns:
        dict, dictionary with data quality metrics
        """
        try:
            logger.info("Checking data quality")
            
            quality_report = {
                'total_rows': len(self.data),
                'total_columns': len(self.data.columns),
                'missing_values': self.data.isnull().sum().sum(),
                'duplicate_rows': self.data.duplicated().sum(),
                'dtypes': self.data.dtypes.to_dict()
            }
            
            logger.info(f"Total rows: {quality_report['total_rows']:,}")
            logger.info(f"Total columns: {quality_report['total_columns']}")
            logger.info(f"Missing values: {quality_report['missing_values']}")
            logger.info(f"Duplicate rows: {quality_report['duplicate_rows']}")
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Error in data quality check: {str(e)}")
            raise
    
    def remove_features(self, features_to_drop: list) -> pd.DataFrame:
        """
        remove specified features from the dataset
        
        params:
        features_to_drop : list
        list of column names to drop
            
        Returns:
        pd.DataFrame
        """
        try:            
            existing_features = [f for f in features_to_drop if f in self.data.columns]
            
            if existing_features:
                self.data = self.data.drop(columns=existing_features)
                for feat in existing_features:
                    logger.info(f"Removed: {feat}")
            else:
                logger.warning("No matching features found to remove")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error removing features: {str(e)}")
            raise
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        get the preprocessed dataframe
        returns:
        pd.DataFrame
        """
        return self.data.copy()