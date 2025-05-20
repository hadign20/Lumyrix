# core/utils/io.py

import os
import pandas as pd
import numpy as np
import pickle
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

def save_excel_sheet(
    df: pd.DataFrame,
    filepath: str,
    sheetname: str = "Sheet1",
    index: bool = False,
    mode: str = 'a'
) -> None:
    """
    Save dataframe to Excel sheet, creating file if it doesn't exist
    or appending to existing file.
    
    Args:
        df: DataFrame to save
        filepath: Path to Excel file
        sheetname: Name of sheet
        index: Whether to include index
        mode: File mode ('a' for append, 'w' for write)
    """
    # Check if directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Check if file exists
    file_exists = os.path.exists(filepath)
    
    try:
        # Create file if it does not exist
        if not file_exists:
            df.to_excel(filepath, sheet_name=sheetname, index=index)
            logger.debug(f"Created new Excel file: {filepath}, sheet: {sheetname}")
        
        # Otherwise, append to existing file
        else:
            # Check if sheet already exists
            with pd.ExcelWriter(filepath, engine='openpyxl', mode=mode) as writer:
                df.to_excel(writer, sheet_name=sheetname, index=index)
                logger.debug(f"Saved to existing Excel file: {filepath}, sheet: {sheetname}")
    
    except Exception as e:
        logger.error(f"Error saving Excel sheet {sheetname} to {filepath}: {str(e)}")
        # Try with mode='w' if append fails
        if mode == 'a':
            try:
                df.to_excel(filepath, sheet_name=sheetname, index=index)
                logger.debug(f"Created new Excel file (fallback): {filepath}, sheet: {sheetname}")
            except Exception as e2:
                logger.error(f"Error in fallback save attempt: {str(e2)}")

def load_data(
    filepath: str,
    sheet_name: Optional[str] = None,
    exclude_columns: Optional[List[str]] = None,
    fill_na: Union[float, str, Dict] = 0
) -> pd.DataFrame:
    """
    Load data from file (Excel, CSV, etc).
    
    Args:
        filepath: Path to data file
        sheet_name: Name of Excel sheet (if Excel file)
        exclude_columns: Columns to exclude
        fill_na: Value to fill NAs with
        
    Returns:
        Loaded DataFrame
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return pd.DataFrame()
    
    try:
        # Determine file type from extension
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.xlsx' or file_ext == '.xls':
            # Load Excel file
            if sheet_name:
                df = pd.read_excel(filepath, sheet_name=sheet_name)
            else:
                df = pd.read_excel(filepath)
        
        elif file_ext == '.csv':
            # Load CSV file
            df = pd.read_csv(filepath)
        
        elif file_ext == '.pkl':
            # Load pickle file
            df = pd.read_pickle(filepath)
        
        else:
            logger.error(f"Unsupported file type: {file_ext}")
            return pd.DataFrame()
        
        # Exclude columns if specified
        if exclude_columns:
            df = df.drop(columns=[col for col in exclude_columns if col in df.columns])
        
        # Fill NA values
        df = df.fillna(fill_na)
        
        logger.debug(f"Loaded data from {filepath}, shape: {df.shape}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {str(e)}")
        return pd.DataFrame()

def save_model(
    model: Any,
    filepath: str,
    metadata: Optional[Dict] = None
) -> bool:
    """
    Save model to file with optional metadata.
    
    Args:
        model: Model to save
        filepath: Path to save model
        metadata: Additional metadata to save with model
        
    Returns:
        True if successful, False otherwise
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        # Add metadata if provided
        if metadata:
            # Attach metadata to model if possible
            for key, value in metadata.items():
                setattr(model, key, value)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        logger.debug(f"Model saved to {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving model to {filepath}: {str(e)}")
        return False

def load_model(filepath: str) -> Any:
    """
    Load model from file.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model or None if error
    """
    if not os.path.exists(filepath):
        logger.error(f"Model file not found: {filepath}")
        return None
    
    try:
        # Load model
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.debug(f"Model loaded from {filepath}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {str(e)}")
        return None