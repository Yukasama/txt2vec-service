"""Save dataset as CSV file to disk."""

from pathlib import Path

import pandas as pd

from vectorize.config import settings

__all__ = ["_save_dataframe_to_fs"]


def _save_dataframe_to_fs(df: pd.DataFrame, filename: str) -> Path:
    """Persist DataFrame as CSV in upload_dir and return its path.

    Args:
        df: DataFrame to write.
        filename: Target filename (already sanitised).

    Returns:
        Path pointing to the saved CSV file.
    """
    out_path = settings.dataset_upload_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
