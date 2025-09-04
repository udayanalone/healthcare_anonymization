#!/usr/bin/env python3
"""
Healthcare Data Anonymization Pipeline
=====================================

This script implements a comprehensive healthcare data anonymization pipeline
following the specified requirements for removing PII and protecting patient privacy.

Author: Healthcare Anonymization System
Date: 2024
"""

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthcareAnonymizationPipeline:
    """
    Comprehensive healthcare data anonymization pipeline implementing:
    1. Data Collection and Schema Analysis
    2. Preprocessing (Missing Value Handling)
    3. PII Identification and Classification
    4. Anonymization Methods (Column Deletion, Pseudonymization, Generalization, Mapping)
    """
    
    def __init__(self, output_dir="output"):
        """
        Initialize the anonymization pipeline.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save anonymized data and mappings
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize mapping dictionaries
        self.mappings = {
            'encounters': {},
            'patients': {},
            'organizations': {},
            'payers': {},
            'id_payers': {},
            'reason_codes': {}
        }
        
        # Define PII categories
        self.direct_identifiers = [
            'SSN', 'DRIVERS', 'PASSPORT', 'PREFIX', 'FIRST', 'LAST', 'MAIDEN',
            'ADDRESS', 'ZIP', 'LAT', 'LON', 'ADDRESS_payer', 'PHONE_payer',
            'DEATHDATE', 'PROVIDER', 'PHONE', 'NAME_payer'
        ]
        
        self.quasi_identifiers = [
            'BIRTHDATE', 'CITY', 'STATE', 'COUNTY', 'ORGANIZATION', 'PATIENT'
        ]
        
        self.sensitive_attributes = [
            'REASONCODE', 'REASONDESCRIPTION', 'PAYER', 'CONDITION_codes',
            'CONDITION_descriptions', 'MEDICATION_codes', 'MEDICATION_descriptions'
        ]
        
        logger.info("Healthcare Anonymization Pipeline initialized")
    
    def load_data(self, file_path, n_records=None):
        """
        Step 1: Data Collection
        Load and analyze the healthcare dataset.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
        n_records : int, optional
            Number of records to load (None for all)
            
        Returns:
        --------
        pandas.DataFrame
            Loaded dataset
        """
        logger.info("=== STEP 1: DATA COLLECTION ===")
        
        try:
            df = pd.read_csv(file_path, nrows=n_records)
            logger.info(f"Successfully loaded {len(df)} records from {file_path}")
            logger.info(f"Dataset shape: {df.shape}")
            
            # Analyze schema
            self.analyze_schema(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def analyze_schema(self, df):
        """
        Analyze dataset schema to identify PII, quasi-identifiers, and sensitive attributes.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset to analyze
        """
        logger.info("Analyzing dataset schema...")
        
        # Check for direct identifiers
        found_direct_ids = [col for col in self.direct_identifiers if col in df.columns]
        logger.info(f"Direct identifiers found: {found_direct_ids}")
        
        # Check for quasi-identifiers
        found_quasi_ids = [col for col in self.quasi_identifiers if col in df.columns]
        logger.info(f"Quasi-identifiers found: {found_quasi_ids}")
        
        # Check for sensitive attributes
        found_sensitive = [col for col in self.sensitive_attributes if col in df.columns]
        logger.info(f"Sensitive attributes found: {found_sensitive}")
        
        # Data quality analysis
        missing_values = df.isnull().sum()
        total_missing = missing_values.sum()
        logger.info(f"Total missing values: {total_missing}")
        
        # Save schema analysis
        schema_analysis = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'direct_identifiers': found_direct_ids,
            'quasi_identifiers': found_quasi_ids,
            'sensitive_attributes': found_sensitive,
            'missing_values_per_column': missing_values.to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        with open(self.output_dir / 'schema_analysis.json', 'w') as f:
            json.dump(schema_analysis, f, indent=2, default=str)
        
        logger.info("Schema analysis saved to output/schema_analysis.json")
    
    def preprocess_data(self, df):
        """
        Step 2: Preprocessing
        Handle missing values and data cleaning.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset to preprocess
            
        Returns:
        --------
        pandas.DataFrame
            Preprocessed dataset
        """
        logger.info("=== STEP 2: PREPROCESSING ===")
        
        df_processed = df.copy()
        
        # Handle missing values
        missing_before = df_processed.isnull().sum().sum()
        logger.info(f"Missing values before preprocessing: {missing_before}")
        
        # For categorical columns, fill with 'Unknown'
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna('Unknown')
        
        # For numerical columns, fill with median
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        missing_after = df_processed.isnull().sum().sum()
        logger.info(f"Missing values after preprocessing: {missing_after}")
        logger.info(f"Reduced missing values by: {missing_before - missing_after}")
        
        return df_processed
    
    def remove_direct_identifiers(self, df):
        """
        Step 4a: Column Deletion
        Remove direct identifier columns.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset to process
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with direct identifiers removed
        """
        logger.info("=== STEP 4a: COLUMN DELETION ===")
        
        df_cleaned = df.copy()
        
        # Find columns that exist in the dataset
        columns_to_remove = [col for col in self.direct_identifiers if col in df_cleaned.columns]
        
        if columns_to_remove:
            logger.info(f"Removing direct identifier columns: {columns_to_remove}")
            df_cleaned = df_cleaned.drop(columns=columns_to_remove)
            logger.info(f"Removed {len(columns_to_remove)} columns")
        else:
            logger.info("No direct identifier columns found to remove")
        
        logger.info(f"Dataset shape after column removal: {df_cleaned.shape}")
        return df_cleaned
    
    def pseudonymize_encounter_ids(self, df):
        """
        Step 4b: Encounter ID Pseudonymization
        Convert Id_encounter to EN00001, EN00002, etc.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset to process
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with pseudonymized encounter IDs
        """
        logger.info("=== STEP 4b: ENCOUNTER ID PSEUDONYMIZATION ===")
        
        if 'Id_encounter' not in df.columns:
            logger.warning("Id_encounter column not found")
            return df
        
        df_pseudo = df.copy()
        
        def get_encounter_pseudonym(encounter_id):
            if pd.isna(encounter_id) or encounter_id == '':
                return encounter_id
            
            if encounter_id in self.mappings['encounters']:
                return self.mappings['encounters'][encounter_id]
            
            next_num = len(self.mappings['encounters']) + 1
            pseudonym = f"EN{next_num:05d}"
            self.mappings['encounters'][encounter_id] = pseudonym
            return pseudonym
        
        df_pseudo['Id_encounter'] = df_pseudo['Id_encounter'].apply(get_encounter_pseudonym)
        logger.info(f"Pseudonymized {len(self.mappings['encounters'])} encounter IDs")
        
        return df_pseudo
    
    def generalize_dates(self, df):
        """
        Step 4c: Date Generalization
        Convert dates to year-only format.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset to process
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with generalized dates
        """
        logger.info("=== STEP 4c: DATE GENERALIZATION ===")
        
        df_generalized = df.copy()
        
        # Date columns to generalize
        date_mappings = {
            'START': 'start_year',
            'STOP': 'stop_year',
            'BIRTHDATE': 'birth_year',
            'condition_dates': 'condition_year'
        }
        
        for original_col, new_col in date_mappings.items():
            if original_col in df_generalized.columns:
                logger.info(f"Generalizing {original_col} to {new_col}")
                
                def extract_year(date_str):
                    if pd.isna(date_str) or date_str == '':
                        return date_str
                    
                    # Handle different date formats
                    if isinstance(date_str, str):
                        # Try to extract year from various formats
                        year_match = re.search(r'(\d{4})', str(date_str))
                        if year_match:
                            return int(year_match.group(1))
                    
                    return date_str
                
                # Apply year extraction
                df_generalized[new_col] = df_generalized[original_col].apply(extract_year)
                
                # Remove original column
                df_generalized = df_generalized.drop(columns=[original_col])
        
        return df_generalized
    
    def pseudonymize_organizations(self, df):
        """
        Step 4d: Organization Grouping
        Map same ORGANIZATION to Org_001, Org_002, etc.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset to process
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with pseudonymized organizations
        """
        logger.info("=== STEP 4d: ORGANIZATION GROUPING ===")
        
        if 'ORGANIZATION' not in df.columns:
            logger.warning("ORGANIZATION column not found")
            return df
        
        df_pseudo = df.copy()
        
        def get_org_pseudonym(org_id):
            if pd.isna(org_id) or org_id == '':
                return org_id
            
            if org_id in self.mappings['organizations']:
                return self.mappings['organizations'][org_id]
            
            next_num = len(self.mappings['organizations']) + 1
            pseudonym = f"Org_{next_num:03d}"
            self.mappings['organizations'][org_id] = pseudonym
            return pseudonym
        
        df_pseudo['ORGANIZATION'] = df_pseudo['ORGANIZATION'].apply(get_org_pseudonym)
        logger.info(f"Pseudonymized {len(self.mappings['organizations'])} organizations")
        
        return df_pseudo
    
    def pseudonymize_patients(self, df):
        """
        Step 4e: Patient Pseudonymization
        Convert Patient to Patient_0001, Patient_0002, etc.
        Also anonymize Id_patient.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset to process
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with pseudonymized patients
        """
        logger.info("=== STEP 4e: PATIENT PSEUDONYMIZATION ===")
        
        df_pseudo = df.copy()
        
        def get_patient_pseudonym(patient_id):
            if pd.isna(patient_id) or patient_id == '':
                return patient_id
            
            if patient_id in self.mappings['patients']:
                return self.mappings['patients'][patient_id]
            
            next_num = len(self.mappings['patients']) + 1
            pseudonym = f"Patient_{next_num:04d}"
            self.mappings['patients'][patient_id] = pseudonym
            return pseudonym
        
        # Pseudonymize PATIENT column
        if 'PATIENT' in df_pseudo.columns:
            df_pseudo['PATIENT'] = df_pseudo['PATIENT'].apply(get_patient_pseudonym)
        
        # Pseudonymize Id_patient column
        if 'Id_patient' in df_pseudo.columns:
            df_pseudo['Id_patient'] = df_pseudo['Id_patient'].apply(get_patient_pseudonym)
        
        logger.info(f"Pseudonymized {len(self.mappings['patients'])} patients")
        
        return df_pseudo
    
    def pseudonymize_payers(self, df):
        """
        Step 4f: Payer Mapping
        Map payer codes to Payer_0001, Payer_0002, etc.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset to process
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with pseudonymized payers
        """
        logger.info("=== STEP 4f: PAYER MAPPING ===")
        
        if 'PAYER' not in df.columns:
            logger.warning("PAYER column not found")
            return df
        
        df_pseudo = df.copy()
        
        def get_payer_pseudonym(payer_id):
            if pd.isna(payer_id) or payer_id == '':
                return payer_id
            
            if payer_id in self.mappings['payers']:
                return self.mappings['payers'][payer_id]
            
            next_num = len(self.mappings['payers']) + 1
            pseudonym = f"Payer_{next_num:04d}"
            self.mappings['payers'][payer_id] = pseudonym
            return pseudonym
        
        df_pseudo['PAYER'] = df_pseudo['PAYER'].apply(get_payer_pseudonym)
        logger.info(f"Pseudonymized {len(self.mappings['payers'])} payers")
        
        return df_pseudo
    
    def pseudonymize_id_payers(self, df):
        """
        Step 4f2: Id_payer Mapping
        Map Id_payer to Payer_0001, Payer_0002, etc.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset to process
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with pseudonymized Id_payers
        """
        logger.info("=== STEP 4f2: ID_PAYER MAPPING ===")
        
        if 'Id_payer' not in df.columns:
            logger.warning("Id_payer column not found")
            return df
        
        df_pseudo = df.copy()
        
        def get_id_payer_pseudonym(id_payer):
            if pd.isna(id_payer) or id_payer == '':
                return id_payer
            
            if id_payer in self.mappings['id_payers']:
                return self.mappings['id_payers'][id_payer]
            
            next_num = len(self.mappings['id_payers']) + 1
            pseudonym = f"Payer_{next_num:04d}"
            self.mappings['id_payers'][id_payer] = pseudonym
            return pseudonym
        
        df_pseudo['Id_payer'] = df_pseudo['Id_payer'].apply(get_id_payer_pseudonym)
        logger.info(f"Pseudonymized {len(self.mappings['id_payers'])} Id_payers")
        
        return df_pseudo
    
    def pseudonymize_reason_codes(self, df):
        """
        Step 4g: Reason Code Mapping
        Map reason_code uniquely.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset to process
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with pseudonymized reason codes
        """
        logger.info("=== STEP 4g: REASON CODE MAPPING ===")
        
        if 'REASONCODE' not in df.columns:
            logger.warning("REASONCODE column not found")
            return df
        
        df_pseudo = df.copy()
        
        def get_reason_code_pseudonym(reason_code):
            if pd.isna(reason_code) or reason_code == '':
                return reason_code
            
            if reason_code in self.mappings['reason_codes']:
                return self.mappings['reason_codes'][reason_code]
            
            next_num = len(self.mappings['reason_codes']) + 1
            pseudonym = f"RC_{next_num:04d}"
            self.mappings['reason_codes'][reason_code] = pseudonym
            return pseudonym
        
        df_pseudo['REASONCODE'] = df_pseudo['REASONCODE'].apply(get_reason_code_pseudonym)
        logger.info(f"Pseudonymized {len(self.mappings['reason_codes'])} reason codes")
        
        return df_pseudo
    
    def save_mappings(self):
        """Save all pseudonymization mappings."""
        mapping_file = self.output_dir / 'pseudonymization_mappings.json'
        
        with open(mapping_file, 'w') as f:
            json.dump(self.mappings, f, indent=2)
        
        logger.info(f"Saved pseudonymization mappings to {mapping_file}")
    
    def run_anonymization_pipeline(self, input_file, n_records=None):
        """
        Run the complete anonymization pipeline.
        
        Parameters:
        -----------
        input_file : str
            Path to input CSV file
        n_records : int, optional
            Number of records to process (None for all)
            
        Returns:
        --------
        pandas.DataFrame
            Fully anonymized dataset
        """
        logger.info("=== HEALTHCARE DATA ANONYMIZATION PIPELINE ===")
        
        # Step 1: Data Collection
        df = self.load_data(input_file, n_records)
        if df is None:
            logger.error("Failed to load data. Exiting.")
            return None
        
        # Step 2: Preprocessing
        df = self.preprocess_data(df)
        
        # Step 3: PII Identification (already done in analyze_schema)
        logger.info("=== STEP 3: PII IDENTIFICATION ===")
        logger.info("PII identification completed during schema analysis")
        
        # Step 4: Anonymization Methods
        logger.info("=== STEP 4: ANONYMIZATION METHODS ===")
        
        # 4a. Column Deletion
        df = self.remove_direct_identifiers(df)
        
        # 4b. Encounter ID Pseudonymization
        df = self.pseudonymize_encounter_ids(df)
        
        # 4c. Date Generalization
        df = self.generalize_dates(df)
        
        # 4d. Organization Grouping
        df = self.pseudonymize_organizations(df)
        
        # 4e. Patient Pseudonymization
        df = self.pseudonymize_patients(df)
        
        # 4f. Payer Mapping
        df = self.pseudonymize_payers(df)
        
        # 4f2. Id_payer Mapping
        df = self.pseudonymize_id_payers(df)
        
        # 4g. Reason Code Mapping
        df = self.pseudonymize_reason_codes(df)
        
        # Save mappings
        self.save_mappings()
        
        # Save anonymized data
        output_file = self.output_dir / 'anonymized_final.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Anonymized data saved to {output_file}")
        
        # Generate summary report
        self.generate_summary_report(df)
        
        logger.info("=== ANONYMIZATION PIPELINE COMPLETED ===")
        return df
    
    def generate_summary_report(self, df):
        """Generate a summary report of the anonymization process."""
        report = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'mappings_generated': {
                'encounters': len(self.mappings['encounters']),
                'patients': len(self.mappings['patients']),
                'organizations': len(self.mappings['organizations']),
                'payers': len(self.mappings['payers']),
                'id_payers': len(self.mappings['id_payers']),
                'reason_codes': len(self.mappings['reason_codes'])
            },
            'columns_removed': len(self.direct_identifiers),
            'date_columns_generalized': 4,
            'anonymization_methods_applied': [
                'Column Deletion',
                'Encounter ID Pseudonymization',
                'Date Generalization',
                'Organization Grouping',
                'Patient Pseudonymization',
                'Payer Mapping',
                'Id_payer Mapping',
                'Reason Code Mapping'
            ]
        }
        
        report_file = self.output_dir / 'anonymization_summary.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved to {report_file}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("ANONYMIZATION SUMMARY REPORT")
        print("="*60)
        print(f"Total records processed: {report['total_records']:,}")
        print(f"Total columns in final dataset: {report['total_columns']}")
        print(f"Columns removed: {report['columns_removed']}")
        print(f"Date columns generalized: {report['date_columns_generalized']}")
        print("\nMappings generated:")
        for key, value in report['mappings_generated'].items():
            print(f"  {key}: {value:,}")
        print("\nAnonymization methods applied:")
        for method in report['anonymization_methods_applied']:
            print(f"  ✓ {method}")
        print("="*60)

def main():
    """Main function to run the anonymization pipeline."""
    parser = argparse.ArgumentParser(description='Healthcare Data Anonymization Pipeline')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for anonymized data and reports')
    parser.add_argument('--records', type=int, default=None,
                        help='Number of records to process (None for all)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = HealthcareAnonymizationPipeline(output_dir=args.output_dir)
    
    # Run anonymization
    anonymized_df = pipeline.run_anonymization_pipeline(args.input, args.records)
    
    if anonymized_df is not None:
        print(f"\n✅ Anonymization completed successfully!")
        print(f"📊 Processed {len(anonymized_df):,} records")
        print(f"📁 Output saved to: {args.output_dir}")
    else:
        print("❌ Anonymization failed!")

if __name__ == "__main__":
    main()
