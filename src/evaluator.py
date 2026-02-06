"""
Model evaluation module for fraud detection
Handles predictions, metrics calculation, and visualizations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, average_precision_score
)
import logging
from typing import Dict, Any, Optional, Tuple

#Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates trained fraud detection model
    Attributes:
    model : object -- trained model
    X_test : pd.DataFrame -- test features
    y_test : pd.Series -- test target
    y_pred : np.ndarray -- predicted labels
    y_pred_proba : np.ndarray -- predicted probabilities
    metrics : dict -- dictionary of evaluation metrics
    """
    
    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Params:
        model : object -- trained model
        X_test : pd.DataFrame -- test features
        y_test : pd.Series -- test target
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics = {}
        logger.info(f"Initialized ModelEvaluator with {len(X_test):,} test samples")
    
    def make_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        generate predictions on test data 
        Returns:
        tuple: (y_pred, y_pred_proba) --predicted labels and probabilities
        """
        try:
            logger.info("predicting on on test set")
            
            self.y_pred = self.model.predict(self.X_test)
            self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            logger.info("predictions complete")
            
            return self.y_pred, self.y_pred_proba
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        all evaluation metrics 
        Returns:
        dict -- dictionary containing all metrics
        """
        try:
            if self.y_pred is None or self.y_pred_proba is None:
                self.make_predictions()
            
            logger.info("Calculating metrics")
            
            self.metrics = {
                'accuracy': accuracy_score(self.y_test, self.y_pred),
                'precision': precision_score(self.y_test, self.y_pred),
                'recall': recall_score(self.y_test, self.y_pred),
                'f1_score': f1_score(self.y_test, self.y_pred),
                'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba),
                'average_precision': average_precision_score(self.y_test, self.y_pred_proba)
            }
            
            logger.info("Metrics calculated:")
            logger.info(f"Accuracy: {self.metrics['accuracy']:.4f}")
            logger.info(f"Precision:{self.metrics['precision']:.4f}")
            logger.info(f"Recall:{self.metrics['recall']:.4f}")
            logger.info(f"F1-Score:{self.metrics['f1_score']:.4f}")
            logger.info(f"ROC-AUC:{self.metrics['roc_auc']:.4f}")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> None:
        """
        plot confusion matri
        parameters: --save_path : str, path to save the plot
        """
        try:
            if self.y_pred is None:
                self.make_predictions()
            
            logger.info("Creating confusion matrix plot")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            cm = confusion_matrix(self.y_test, self.y_pred)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, 
                display_labels=['Legitimate', 'Fraudulent']
            )
            disp.plot(ax=ax)
            ax.set_title('Confusion matrix')            
            #add percentages
            for i in range(2):
                for j in range(2):
                    text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                    pct = (cm[i, j] / cm.sum()) * 100
                    ax.text(j, i + 0.2, f'({pct:.1f}%)')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
                logger.info(f"saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise
    
    def plot_roc_curve(self, save_path: Optional[str] = None) -> None:
        """
        lot ROC curve
        Parameters:
        save_path : str, optional -- Path to save the plot
        """
        try:
            if self.y_pred_proba is None:
                self.make_predictions()
            
            logger.info("creating ROC curve")
            
            fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
            roc_auc = self.metrics.get('roc_auc', roc_auc_score(self.y_test, self.y_pred_proba))
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.plot(fpr, tpr, label=f'ROC-curve (AUC = {roc_auc:.4f})')
            ax.plot([0, 1], [0, 1], 'k--',
                   label='random classifier (AUC = 0.5)')
            
            ax.set_xlabel('False positive rate')
            ax.set_ylabel('True positive rate')
            ax.set_title('ROC curve')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {str(e)}")
            raise
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None) -> None:
        """
        plot Precision-Recall curve
        Parameters:
        save_path : str, optional, path to save the plot
        """
        try:
            if self.y_pred_proba is None:
                self.make_predictions()
            
            logger.info("creating Precision-Recall curve")
            
            precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
            avg_precision = self.metrics.get('average_precision', 
                                            average_precision_score(self.y_test, self.y_pred_proba))
            baseline = self.y_test.sum() / len(self.y_test)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.plot(recall, precision, linewidth=2,
                   label=f'PR curve(AP = {avg_precision:.4f})')
            ax.plot([0, 1], [baseline, baseline], 'k--',
                   label=f'Baseline (AP = {baseline:.4f})')
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting PR curve: {str(e)}")
            raise
    
    def plot_feature_importance(self, feature_names: list, 
                                top_n: int = 10,
                                save_path: Optional[str] = None) -> None:
        """
        Plot feature importance 
        Parameters:
        feature_names : list, list of feature names
        top_n : int, number of top features to display
        save_path : str, optional, path to save the plot
        """
        try:
            logger.info(f"creating feature importance plot (top {top_n})")
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.barh(range(len(importance_df)), importance_df['importance'].values)
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'].values)
            ax.set_xlabel('Importance')
            ax.set_title(f'Top {top_n} feature importances',)
            ax.invert_yaxis()            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            raise
    
    def plot_all_evaluations(self, feature_names: list, 
                            save_path: Optional[str] = None) -> None:
        """
        create evaluation plot with all metrics,
        Parameters:
        -----------
        feature_names : list, ist of feature names
        save_path : str, optional, path to save the plot
        """
        try:
            if self.y_pred is None or self.y_pred_proba is None:
                self.make_predictions()
            
            if not self.metrics:
                self.calculate_metrics()
            
            logger.info("creating all evaluation plot")
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            cm = confusion_matrix(self.y_test, self.y_pred)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=['Legitimate', 'Fraudulent']
            )
            disp.plot(ax=axes[0, 0])
            axes[0, 0].set_title('Confusion matrix')
            for i in range(2):
                for j in range(2):
                    text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                    pct = (cm[i, j] / cm.sum()) * 100
                    axes[0, 0].text(j, i + 0.2, f'({pct:.1f}%)')
            
            # ROC
            fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
            axes[0, 1].plot(fpr, tpr,
                           label=f'ROC (AUC={self.metrics["roc_auc"]:.4f})')
            axes[0, 1].plot([0, 1], [0, 1], 'k--', 
                           label='random (AUC=0.5)')
            axes[0, 1].set_xlabel('FPR')
            axes[0, 1].set_ylabel('TPR')
            axes[0, 1].set_title('ROC Curve')
            
            # PR curve
            precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
            baseline = self.y_test.sum() / len(self.y_test)
            axes[1, 0].plot(recall, precision, linewidth=2,
                           label=f'PR (AP={self.metrics["average_precision"]:.4f})')
            axes[1, 0].plot([0, 1], [baseline, baseline], 'k--',
                           label=f'Baseline(AP={baseline:.4f})')
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title('PR Curve')
            
            #Feats importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            axes[1, 1].barh(range(len(importance_df)),
                           importance_df['importance'].values)
            axes[1, 1].set_yticks(range(len(importance_df)))
            axes[1, 1].set_yticklabels(importance_df['feature'].values)
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 10 Feature Importances')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Evaluation plot saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating comprehensive plot: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, float]:
        """
        get dictionary of calculated metrics.
        
        Returns: dict, dictionary of metrics
        """
        return self.metrics.copy()