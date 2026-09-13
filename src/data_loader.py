from pathlib import Path
import pandas as pd


def load_raw_dataset(path: str) -> pd.DataFrame:
    """
    Load the raw Datafiniti CSV dataset.

    Parameters
    ----------
    path : str
        Path to 245_1.csv.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path.resolve()}")

    df = pd.read_csv(file_path)
    return df
