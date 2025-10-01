"""Data loading utilities for the project.

- load_uci(): load the UCI Online Shoppers Purchasing Intention dataset from data/
- load_retail2(optional): load/aggregate Online Retail II to RFM features (if used)
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_uci(csv_name: str = "online_shoppers_intention.csv") -> pd.DataFrame:
    """Load the UCI Online Shoppers dataset from data/.

Returns a pandas DataFrame.
"""
    path = DATA_DIR / csv_name
    if not path.exists():
        raise FileNotFoundError(f"Expected {path} — please place the CSV under data/.")
    df = pd.read_csv(path)
    return df

def load_retail2(xls_name: str = "online_retail_II.xlsx") -> pd.DataFrame:
    """Optional: load Online Retail II raw Excel file from data/.

You may aggregate to RFM in preprocessing.
"""
    path = DATA_DIR / xls_name
    if not path.exists():
        raise FileNotFoundError(f"Expected {path} — please place the Excel file under data/.")
    # Be mindful of memory; consider loading a subset/sheet.
    df = pd.read_excel(path)
    return df
