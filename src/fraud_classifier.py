"""
Main fraud classifier orchestrator
Coordinates preprocessing, feature engineering, training, and evaluation.
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
import os

from .preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model_trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .utils import load_data, create_output_dirs, save_results

#Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FraudClassifier:
    """
    Main class that orchestrates the entire fraud detection pipeline
    
    This class coordinates:
    - Data preprocessing
    - Feature engineering
    - Model training
    - Model evaluation
    
    Attributes:
    data: pd.DataFrame, raw input data
    preprocessor: DataPreprocessor
    feature_engineer: FeatureEngineer
    trainer: ModelTrainer
    evaluator: ModelEvaluator
    """
    
    def __init__(self, data_path: Optional[str] = None, data: Optional[pd.DataFrame] = None):
        """
        Initialize fraud classifier.
        
        Parameters:
        data_path : str, optional, path to csv file
        data : pd.DataFrame, optional --Dataframe (if already loaded)
        """ 
        #load data
        if data_path:
            logger.info(f"Loading data from: {data_path}")
            self.data = load_data(data_path)
        elif data is not None:
            self.data = data.copy()
        else:
            raise ValueError("Must provide either data_path or data parameter")
        
        #components
        self.preprocessor = None
        self.feature_engineer = None
        self.trainer = None
        self.evaluator = None
        
        self.X = None
        self.y = None
        self.feature_columns = None
        
        #output directories
        create_output_dirs()
        
        logger.info(f"Initialized with {len(self.data):,} rows, {len(self.data.columns)} columns")
    
    def run_preprocessing(self, 
                         features_to_drop: list = None,
                         missing_value_strategy: str = 'unknown') -> pd.DataFrame:
        """
        Run data preprocessing pipeline
        
        Parameters:
        features_to_drop: list, optiona,l ist of features to remove
        missing_value_strategy: str, strategy for handling missing values ('unknown' or 'drop')
            
        Returns:
        pd.DataFrame, preprocessed data
        """
        try:
            
            self.preprocessor = DataPreprocessor(self.data)
            self.preprocessor.check_data_quality()
            self.preprocessor.handle_missing_values(strategy=missing_value_strategy)
            if features_to_drop:
                self.preprocessor.remove_features(features_to_drop)
            processed_data = self.preprocessor.get_processed_data()
            
            logger.info(f"Preprocessing complete")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def run_feature_engineering(self,
                                categorical_features: list,
                                numerical_features: list,
                                target: str = 'is_fraud') -> Tuple[pd.DataFrame, pd.Series, list]:
        """
        Run feature engineering pipeline
        Params: categorical_features, list of categorical feature names
        numerical_features: list of numerical feature names
        target : str, target variable name
            
        Returns:
        (X, y, feature_columns)
        """
        try:
            #preprocessed data
            if self.preprocessor is None:
                raise ValueError("must run preprocessing first")
            
            processed_data = self.preprocessor.get_processed_data()
            self.feature_engineer = FeatureEngineer(processed_data)
            self.feature_engineer.create_temporal_features()
            
            self.feature_engineer.encode_categorical_features(categorical_features)
            
            temporal_features = ['hour', 'day_of_week', 'is_weekend', 'is_night_registration']
            
            self.X, self.y, self.feature_columns = self.feature_engineer.select_features(
                numerical_features=numerical_features,
                temporal_features=temporal_features,
                categorical_features=categorical_features,
                target=target
            )
            
            logger.info("Feature engineering complete")
            
            return self.X, self.y, self.feature_columns
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def run_training(self,
                    test_size: float = 0.2,
                    n_estimators: int = 100,
                    max_depth: int = 6,
                    learning_rate: float = 0.1,
                    random_state: int = 42):
        """
        runmodel training pipeline
        
        Parameters:
        test_size: float, poportion of data for test set
        n_estimators: int, number of boosting rounds
        max_depth: int, maximum tree depth
        learning_rate: float, learning rate
        random_state:int, random seed
            
        Returns:
        Trained model
        """
        try:
            
            if self.X is None or self.y is None:
                raise ValueError("Must run feature engineering first")
            
            self.trainer = ModelTrainer()
            self.trainer.prepare_train_test_split(
                X=self.X,
                y=self.y,
                test_size=test_size,
                random_state=random_state
            )
            
            #train model
            model = self.trainer.train_model(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state
            )
            
            logger.info("training complete")
            
            return model
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise
    
    def run_evaluation(self, 
                      save_plots: bool = True,
                      output_dir: str = 'outputs') -> Dict[str, Any]:
        """
        Run model evaluation pipeline
        
        params:
        save_plots: bool,save plots to files
        output_dir : str, directory to save outputs
            
        Returns:
        dict, dictionary with evaluation metrics
        """
        try:
            
            if self.trainer is None or self.trainer.model is None:
                raise ValueError("Must run training first")
            
            _, X_test, _, y_test = self.trainer.get_training_data()
            
            self.evaluator = ModelEvaluator(
                model=self.trainer.model,
                X_test=X_test,
                y_test=y_test
            )

            metrics = self.evaluator.calculate_metrics()
            
            if save_plots:    
                plot_path = os.path.join(output_dir, 'plots', 'model_evaluation.png')
                self.evaluator.plot_all_evaluations(
                    feature_names=self.feature_columns,
                    save_path=plot_path
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            raise
    
    def run_full_pipeline(self,
                         features_to_drop: list = None,
                         categorical_features: list = None,
                         numerical_features: list = None,
                         test_size: float = 0.2,
                         n_estimators: int = 100,
                         max_depth: int = 6,
                         learning_rate: float = 0.1,
                         random_state: int = 42,
                         save_plots: bool = True) -> Dict[str, Any]:
        """
        run the complete fraud detection pipeline 
        params:
        features_to_drop
        categorical_features
        numerical_features
        test_size
        n_estimators
        max_depth
        learning_rate
        random_state
        save_plot
            
        Returns:
        dictionary containing metrics and results
        """
        try:
            
            self.run_preprocessing(
                features_to_drop=features_to_drop,
                missing_value_strategy='unknown'
            )
            
            self.run_feature_engineering(
                categorical_features=categorical_features,
                numerical_features=numerical_features
            )
            
            self.run_training(
                test_size=test_size,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state
            )
            
            metrics = self.run_evaluation(save_plots=save_plots)
            
            results = {
                'metrics': metrics,
                'feature_columns': self.feature_columns,
                'training_info': self.trainer.get_training_info(),
                'model': self.trainer.get_model()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in full pipeline: {str(e)}")
            raise
    
    def get_model(self):
        """
        Get the trained model
        """
        if self.trainer is None:
            return None
        return self.trainer.get_model()
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get evaluation metrics
        """
        if self.evaluator is None:
            return {}
        return self.evaluator.get_metrics()
