#!/usr/bin/env python3
"""
Data Preprocessing Module for Healthcare Data Anonymization
Handles data loading, cleaning, and preparation for GAN training.
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Data preprocessing class for healthcare data anonymization."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        
    def identify_column_types(self, df):
        """Identify different types of columns in the dataset."""
        # Direct identifiers (to be removed)
        direct_identifiers = [
            'SSN', 'DRIVERS', 'PASSPORT', 'PREFIX', 'FIRST', 'LAST', 'MAIDEN',
            'ADDRESS', 'ZIP', 'LAT', 'LON', 'ADDRESS_payer', 'PHONE_payer',
            'DEATHDATE', 'PROVIDER', 'PHONE', 'NAME_payer'
        ]
        
        # Quasi-identifiers (to be pseudonymized)
        quasi_identifiers = [
            'BIRTHDATE', 'CITY', 'STATE', 'COUNTY', 'ORGANIZATION', 'PATIENT'
        ]
        
        # Sensitive attributes (to be protected)
        sensitive_attributes = [
            'REASONCODE', 'REASONDESCRIPTION', 'PAYER'
        ]
        
        # Date columns
        date_columns = [
            'START', 'STOP', 'BIRTHDATE', 'DEATHDATE'
        ]
        
        # ID columns
        id_columns = [
            'Id_encounter', 'Id_patient', 'Id_payer'
        ]
        
        # Categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        categorical_columns = [col for col in categorical_columns if col not in direct_identifiers]
        
        # Numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_columns = [col for col in numerical_columns if col not in direct_identifiers]
        
        return {
            'direct_identifiers': [col for col in direct_identifiers if col in df.columns],
            'quasi_identifiers': [col for col in quasi_identifiers if col in df.columns],
            'sensitive_attributes': [col for col in sensitive_attributes if col in df.columns],
            'date_columns': [col for col in date_columns if col in df.columns],
            'id_columns': [col for col in id_columns if col in df.columns],
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns
        }
    
    def clean_data(self, df):
        """Clean the dataset by handling missing values and data types."""
        df_clean = df.copy()
        
        # Identify column types
        column_types = self.identify_column_types(df_clean)
        
        # Handle missing values in categorical columns
        categorical_cols = column_types['categorical_columns']
        if categorical_cols:
            df_clean[categorical_cols] = self.categorical_imputer.fit_transform(df_clean[categorical_cols])
        
        # Handle missing values in numerical columns
        numerical_cols = column_types['numerical_columns']
        if numerical_cols:
            df_clean[numerical_cols] = self.imputer.fit_transform(df_clean[numerical_cols])
        
        return df_clean
    
    def encode_categorical_data(self, df):
        """Encode categorical data using label encoding."""
        df_encoded = df.copy()
        column_types = self.identify_column_types(df_encoded)
        
        # Encode categorical columns
        for col in column_types['categorical_columns']:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
        
        return df_encoded
    
    def prepare_gan_data(self, df):
        """Prepare data specifically for GAN training."""
        # Clean the data
        df_clean = self.clean_data(df)
        
        # Remove direct identifiers
        column_types = self.identify_column_types(df_clean)
        columns_to_remove = column_types['direct_identifiers']
        df_gan = df_clean.drop(columns=columns_to_remove, errors='ignore')
        
        # Encode categorical data
        df_encoded = self.encode_categorical_data(df_gan)
        
        # Separate condition and target features
        # Condition features: demographic and organizational data
        condition_features = [
            'CITY', 'STATE', 'COUNTY', 'ORGANIZATION', 'PATIENT',
            'MARITAL', 'RACE', 'ETHNICITY', 'GENDER', 'BIRTHPLACE'
        ]
        
        # Target features: clinical and financial data
        target_features = [
            'ENCOUNTERCLASS', 'CODE', 'DESCRIPTION', 'BASE_ENCOUNTER_COST',
            'TOTAL_CLAIM_COST', 'PAYER_COVERAGE', 'REASONCODE', 'REASONDESCRIPTION',
            'PAYER', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'REVENUE',
            'UTILIZATION_org', 'AMOUNT_COVERED', 'AMOUNT_UNCOVERED', 'REVENUE_payer'
        ]
        
        # Filter to existing columns
        condition_features = [col for col in condition_features if col in df_encoded.columns]
        target_features = [col for col in target_features if col in df_encoded.columns]
        
        # Create condition and target matrices
        condition_data = df_encoded[condition_features].values
        target_data = df_encoded[target_features].values
        
        # Scale the data
        condition_scaled = self.scaler.fit_transform(condition_data)
        target_scaled = self.scaler.fit_transform(target_data)
        
        return condition_scaled, target_scaled, condition_features, target_features
    
    def preprocess_for_gan(self, df):
        """Main preprocessing function for GAN training."""
        print("Preprocessing data for GAN training...")
        
        # Prepare GAN data
        condition_data, target_data, condition_features, target_features = self.prepare_gan_data(df)
        
        # Convert to PyTorch tensors
        condition_tensor = torch.FloatTensor(condition_data)
        target_tensor = torch.FloatTensor(target_data)
        
        print(f"Condition features ({len(condition_features)}): {condition_features}")
        print(f"Target features ({len(target_features)}): {target_features}")
        print(f"Condition tensor shape: {condition_tensor.shape}")
        print(f"Target tensor shape: {target_tensor.shape}")
        
        return condition_tensor, target_tensor, self.scaler, self.encoders
    
    def inverse_transform(self, data, feature_names, scaler, encoders):
        """Inverse transform the data back to original format."""
        # Inverse scale
        data_original = scaler.inverse_transform(data)
        
        # Create DataFrame
        df = pd.DataFrame(data_original, columns=feature_names)
        
        # Inverse encode categorical columns
        for col in feature_names:
            if col in encoders:
                df[col] = encoders[col].inverse_transform(df[col].astype(int))
        
        return df
