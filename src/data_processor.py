import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import torch

warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Class for processing healthcare data for anonymization.
    Handles data loading, preprocessing, and feature engineering.
    """
    
    def __init__(self, output_dir='output'):
        """
        Initialize the data processor.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save processed data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encoders and scalers
        self.encoders = {}
        self.scalers = {}
        
        # Feature categories
        self.direct_identifiers = []
        self.quasi_identifiers = []
        self.clinical_data = []
        
        # Feature metadata
        self.feature_metadata = {}
    
    def load_data(self, file_path, n_records=None):
        """
        Load healthcare data from CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
        n_records : int, optional
            Number of records to load (None for all)
            
        Returns:
        --------
        pandas.DataFrame
            Loaded data
        """
        try:
            print(f"Loading data from {file_path}...")
            
            if n_records:
                print(f"Limiting to first {n_records} records...")
                df = pd.read_csv(file_path, nrows=n_records)
            else:
                df = pd.read_csv(file_path)
            
            print(f"Successfully loaded {len(df)} records")
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {', '.join(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def categorize_features(self, df, direct_identifiers=None, quasi_identifiers=None, clinical_data=None):
        """
        Categorize features into direct identifiers, quasi-identifiers, and clinical data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        direct_identifiers : list, optional
            List of direct identifier columns
        quasi_identifiers : list, optional
            List of quasi-identifier columns
        clinical_data : list, optional
            List of clinical data columns
            
        Returns:
        --------
        dict
            Dictionary with categorized features
        """
        # If not provided, use default categorization
        if direct_identifiers is None:
            # Common direct identifiers in healthcare data
            direct_identifiers = [
                col for col in df.columns if any(id_type in col.upper() for id_type in 
                                               ['ID', 'NAME', 'SSN', 'EMAIL', 'PHONE', 'ADDRESS', 
                                                'PASSPORT', 'LICENSE', 'DRIVERS'])
            ]
        
        if quasi_identifiers is None:
            # Common quasi-identifiers in healthcare data
            quasi_identifiers = [
                col for col in df.columns if any(qi_type in col.upper() for qi_type in 
                                               ['AGE', 'GENDER', 'RACE', 'ETHNICITY', 'BIRTHDATE', 
                                                'DEATHDATE', 'MARITAL', 'LANGUAGE', 'BIRTHPLACE', 
                                                'CITY', 'STATE', 'ZIP', 'LAT', 'LON', 'COUNTY'])
            ]
        
        if clinical_data is None:
            # All remaining columns are considered clinical data
            clinical_data = [col for col in df.columns 
                            if col not in direct_identifiers and col not in quasi_identifiers]
        
        # Store categorized features
        self.direct_identifiers = direct_identifiers
        self.quasi_identifiers = quasi_identifiers
        self.clinical_data = clinical_data
        
        # Create feature metadata
        for col in df.columns:
            self.feature_metadata[col] = {
                'category': 'direct_identifier' if col in direct_identifiers else
                           'quasi_identifier' if col in quasi_identifiers else
                           'clinical_data',
                'dtype': str(df[col].dtype),
                'unique_values': df[col].nunique(),
                'missing_values': df[col].isnull().sum()
            }
        
        print(f"Direct identifiers: {len(direct_identifiers)}")
        print(f"Quasi-identifiers: {len(quasi_identifiers)}")
        print(f"Clinical data: {len(clinical_data)}")
        
        return {
            'direct_identifiers': direct_identifiers,
            'quasi_identifiers': quasi_identifiers,
            'clinical_data': clinical_data
        }
    
    def add_age_range(self, df):
        """
        Add AGE_RANGE column based on BIRTHDATE.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with AGE_RANGE column added
        """
        df_with_age = df.copy()
        
        # Check if BIRTHDATE column exists
        if 'BIRTHDATE' in df_with_age.columns:
            # Convert BIRTHDATE to datetime
            df_with_age['BIRTHDATE'] = pd.to_datetime(df_with_age['BIRTHDATE'], errors='coerce')
            
            # Calculate age in years using a fixed reference date (2023-01-01)
            # This ensures consistent age calculation regardless of when the script runs
            reference_date = pd.Timestamp('2023-01-01')
            df_with_age['AGE'] = ((reference_date - df_with_age['BIRTHDATE']).dt.days / 365.25).astype(int)
            
            # Create more granular age ranges for better variety
            conditions = [
                (df_with_age['AGE'] < 12),
                (df_with_age['AGE'] >= 12) & (df_with_age['AGE'] < 18),
                (df_with_age['AGE'] >= 18) & (df_with_age['AGE'] < 25),
                (df_with_age['AGE'] >= 25) & (df_with_age['AGE'] < 35),
                (df_with_age['AGE'] >= 35) & (df_with_age['AGE'] < 45),
                (df_with_age['AGE'] >= 45) & (df_with_age['AGE'] < 55),
                (df_with_age['AGE'] >= 55) & (df_with_age['AGE'] < 65),
                (df_with_age['AGE'] >= 65) & (df_with_age['AGE'] < 75),
                (df_with_age['AGE'] >= 75)
            ]
            
            values = ['<12', '12-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
            
            df_with_age['AGE_RANGE'] = np.select(conditions, values, default='Unknown')
            
            # Drop temporary AGE column
            df_with_age.drop('AGE', axis=1, inplace=True)
            
            # Ensure AGE_RANGE is in quasi_identifiers if not already there
            if 'AGE_RANGE' not in self.quasi_identifiers:
                self.quasi_identifiers.append('AGE_RANGE')
                print(f"Added AGE_RANGE to quasi_identifiers with {len(values)} distinct ranges")
        
        return df_with_age
    
    def preprocess_data(self, df):
        """
        Preprocess data for anonymization.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        dict
            Dictionary with preprocessed data
        """
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Add AGE_RANGE column
        df_processed = self.add_age_range(df_processed)
        
        # Store original AGE_RANGE values to preserve them
        if 'AGE_RANGE' in df_processed.columns:
            self.original_age_ranges = df_processed['AGE_RANGE'].copy()
            # Remove AGE_RANGE from quasi_identifiers for CGAN processing
            if 'AGE_RANGE' in self.quasi_identifiers:
                self.quasi_identifiers.remove('AGE_RANGE')
                print(f"Removed AGE_RANGE from quasi_identifiers for CGAN processing")
                print(f"Will preserve original AGE_RANGE values in final output")
        
        # Handle missing values
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].fillna('')
            else:
                df_processed[col] = df_processed[col].fillna(0)
        
        # Encode categorical features in quasi-identifiers and clinical data
        for col in self.quasi_identifiers + self.clinical_data:
            if df_processed[col].dtype == 'object':
                # Create and fit encoder
                encoder = LabelEncoder()
                df_processed[col] = encoder.fit_transform(df_processed[col])
                self.encoders[col] = encoder
        
        # Scale numerical features in quasi-identifiers
        for col in self.quasi_identifiers:
            if df_processed[col].dtype != 'object':
                # Create and fit scaler
                scaler = MinMaxScaler(feature_range=(-1, 1))  # For tanh activation
                df_processed[col] = scaler.fit_transform(df_processed[col].values.reshape(-1, 1))
                self.scalers[col] = scaler
        
        # Prepare data for CGAN
        condition_features = df_processed[self.clinical_data].values
        target_features = df_processed[self.quasi_identifiers].values
        
        # Convert to tensors
        condition_tensor = torch.FloatTensor(condition_features)
        target_tensor = torch.FloatTensor(target_features)
        
        # Save preprocessed data
        processed_data = {
            'df_processed': df_processed,
            'condition_features': condition_features,
            'target_features': target_features,
            'condition_tensor': condition_tensor,
            'target_tensor': target_tensor,
            'feature_names': {
                'condition': self.clinical_data,
                'target': self.quasi_identifiers,
                'direct': self.direct_identifiers
            },
            'original_age_ranges': self.original_age_ranges if hasattr(self, 'original_age_ranges') else None
        }
        
        return processed_data
    
    def save_preprocessed_data(self, processed_data, filename='processed_data.pkl'):
        """
        Save preprocessed data to disk.
        
        Parameters:
        -----------
        processed_data : dict
            Dictionary with preprocessed data
        filename : str
            Filename to save data
        """
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed data (excluding tensors)
        save_data = {
            'df_processed': processed_data['df_processed'],
            'condition_features': processed_data['condition_features'],
            'target_features': processed_data['target_features'],
            'feature_names': processed_data['feature_names']
        }
        
        # Save data
        joblib.dump(save_data, self.output_dir / filename)
        
        # Save encoders and scalers
        joblib.dump(self.encoders, self.output_dir / 'encoders.pkl')
        joblib.dump(self.scalers, self.output_dir / 'scalers.pkl')
        
        # Save feature metadata
        with open(self.output_dir / 'feature_metadata.json', 'w') as f:
            json.dump(self.feature_metadata, f, indent=2, default=str)
        
        print(f"Preprocessed data saved to {self.output_dir}")
    
    def load_preprocessed_data(self, filename='processed_data.pkl'):
        """
        Load preprocessed data from disk.
        
        Parameters:
        -----------
        filename : str
            Filename to load data from
            
        Returns:
        --------
        dict
            Dictionary with preprocessed data
        """
        try:
            # Load processed data
            save_data = joblib.load(self.output_dir / filename)
            
            # Load encoders and scalers
            self.encoders = joblib.load(self.output_dir / 'encoders.pkl')
            self.scalers = joblib.load(self.output_dir / 'scalers.pkl')
            
            # Load feature metadata
            with open(self.output_dir / 'feature_metadata.json', 'r') as f:
                self.feature_metadata = json.load(f)
            
            # Extract feature categories
            self.direct_identifiers = save_data['feature_names']['direct']
            self.quasi_identifiers = save_data['feature_names']['target']
            self.clinical_data = save_data['feature_names']['condition']
            
            # Load original age ranges if available
            if 'original_age_ranges' in save_data and save_data['original_age_ranges'] is not None:
                self.original_age_ranges = save_data['original_age_ranges']
                print("Loaded original AGE_RANGE values")
                
                # Remove AGE_RANGE from quasi_identifiers if present
                if 'AGE_RANGE' in self.quasi_identifiers:
                    self.quasi_identifiers.remove('AGE_RANGE')
                    print("Removed AGE_RANGE from quasi_identifiers for CGAN processing")
            
            # Convert to tensors
            condition_tensor = torch.FloatTensor(save_data['condition_features'])
            target_tensor = torch.FloatTensor(save_data['target_features'])
            
            # Reconstruct processed data
            processed_data = {
                'df_processed': save_data['df_processed'],
                'condition_features': save_data['condition_features'],
                'target_features': save_data['target_features'],
                'condition_tensor': condition_tensor,
                'target_tensor': target_tensor,
                'feature_names': save_data['feature_names'],
                'original_age_ranges': save_data.get('original_age_ranges', None)
            }
            
            print(f"Preprocessed data loaded from {self.output_dir}")
            return processed_data
            
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            return None
    
    def inverse_transform_data(self, generated_data, original_df):
        """
        Inverse transform generated data back to original format.
        
        Parameters:
        -----------
        generated_data : numpy.ndarray
            Generated data from CGAN
        original_df : pandas.DataFrame
            Original dataframe with direct identifiers
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with inverse transformed data
        """
        # Create dataframe with generated quasi-identifiers
        df_generated = pd.DataFrame(generated_data, columns=self.quasi_identifiers)
        
        # Inverse transform scaled numerical features
        for col in self.quasi_identifiers:
            if col in self.scalers:
                df_generated[col] = self.scalers[col].inverse_transform(df_generated[col].values.reshape(-1, 1))
        
        # Inverse transform encoded categorical features
        for col in self.quasi_identifiers:
            if col in self.encoders:
                # Handle unseen labels by clipping to valid range
                encoder = self.encoders[col]
                classes = encoder.classes_
                # Convert to int and clip to valid range (0 to len(classes)-1)
                values = df_generated[col].values
                values = np.clip(np.round(values).astype(int), 0, len(classes)-1)
                df_generated[col] = encoder.inverse_transform(values)
        
        # Add clinical data from original dataframe
        df_anonymized = pd.DataFrame()
        
        # Add direct identifiers (these will be pseudonymized later)
        for col in self.direct_identifiers:
            if col in original_df.columns:
                df_anonymized[col] = original_df[col].values
        
        # Add generated quasi-identifiers
        for col in self.quasi_identifiers:
            if col in df_generated.columns:
                df_anonymized[col] = df_generated[col].values
        
        # Add AGE_RANGE from original values if available
        if hasattr(self, 'original_age_ranges'):
            print("Using original AGE_RANGE values in final output")
            df_anonymized['AGE_RANGE'] = self.original_age_ranges.values
        elif 'BIRTHDATE' in df_anonymized.columns:
            # Fallback: calculate AGE_RANGE from BIRTHDATE if original values not available
            print("Calculating AGE_RANGE from BIRTHDATE as fallback")
            birthdate = pd.to_datetime(df_anonymized['BIRTHDATE'], errors='coerce')
            reference_date = pd.Timestamp('2023-01-01')
            age = ((reference_date - birthdate).dt.days / 365.25).astype(int)
            
            conditions = [
                (age < 12),
                (age >= 12) & (age < 18),
                (age >= 18) & (age < 25),
                (age >= 25) & (age < 35),
                (age >= 35) & (age < 45),
                (age >= 45) & (age < 55),
                (age >= 55) & (age < 65),
                (age >= 65) & (age < 75),
                (age >= 75)
            ]
            
            values = ['<12', '12-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
            
            df_anonymized['AGE_RANGE'] = np.select(conditions, values, default='Unknown')
        
        # Add clinical data
        for col in self.clinical_data:
            if col in original_df.columns:
                df_anonymized[col] = original_df[col].values
        
        return df_anonymized