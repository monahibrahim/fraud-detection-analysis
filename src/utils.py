"""
Utility functions for fraud detection analysis.
Provides data loading, directory management, and common operations.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any


def load_data(filepath: str) -> pd.DataFrame:
    """
    load dataset from csv and parse timestamp
    """
    df = pd.read_csv(filepath)
    df['registration_timestamp'] = pd.to_datetime(df['registration_timestamp'])
    return df


def create_output_dirs() -> None:
    """
    output directories if they dont exist
    """
    directories = [
        'outputs',
        'outputs/plots',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    

def save_results(results: str, filepath: str) -> None:
    """
    save restult
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(results)
    
    print(f"results saved to: {filepath}")

