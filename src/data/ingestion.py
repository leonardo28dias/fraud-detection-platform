import pandas as pd
import logging
from typing import Tuple
from src.data.validation import validate_dataframe

logger = logging.getLogger(__name__)

def load_and_split_data(filepath: str, time_col: str, val_size: float = 0.15, test_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads data and performs temporal splitting to prevent future data leakage.
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Optional schema validation (can be slow on large datasets, omitted for full dataset load but used in API)
    # df = validate_dataframe(df)
    
    df = df.sort_values(by=time_col).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Temporal Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df