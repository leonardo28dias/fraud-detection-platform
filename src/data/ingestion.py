import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def load_and_split_data(
    filepath: str,
    time_col: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data and perform temporal (chronological) splitting.

    Sorting by time before splitting ensures the model never sees
    future data during training or validation — critical for
    time-ordered transaction data like fraud detection.
    """
    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath)
    n_total = len(df)

    df = df.sort_values(by=time_col).reset_index(drop=True)

    n = len(df)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    train_pos = int(train_df["Class"].sum())
    val_pos = int(val_df["Class"].sum())
    test_pos = int(test_df["Class"].sum())

    logger.info(
        "Temporal split (total=%d, fraud=%d): "
        "Train=%d (%d fraud) | Val=%d (%d fraud) | Test=%d (%d fraud)",
        n_total,
        train_pos + val_pos + test_pos,
        len(train_df),
        train_pos,
        len(val_df),
        val_pos,
        len(test_df),
        test_pos,
    )
    return train_df, val_df, test_df
