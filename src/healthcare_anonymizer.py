import pandas as pd
import numpy as np
import re
from datetime import datetime
from pathlib import Path
import json

class HealthcareAnonymizer:
    """
    Comprehensive healthcare data anonymization class implementing specific requirements:
    1. Column removal/deletion
    2. ID and encounter pseudonymization
    3. Date generalization
    4. Organization grouping
    5. Patient pseudonymization
    6. Payer mapping
    7. Reason code mapping
    """
    
    def __init__(self, mapping_file=None):
        """
        Initialize the healthcare anonymizer.
        
        Parameters:
        -----------
        mapping_file : str, optional
            Path to save/load the anonymization mapping
        """
        self.mapping = {
            'encounters': {},
            'organizations': {},
            'patients': {},
            'payers': {},
            'reason_codes': {}
        }
        self.mapping_file = mapping_file or "output/anonymization_mapping.json"
        self.load_mapping()
    
    def load_mapping(self):
        """Load existing anonymization mapping if available."""
        try:
            if Path(self.mapping_file).exists():
                with open(self.mapping_file, 'r') as f:
                    loaded_mapping = json.load(f)
                    # Ensure all required keys exist
                    self.mapping = {
                        'encounters': loaded_mapping.get('encounters', {}),
                        'organizations': loaded_mapping.get('organizations', {}),
                        'patients': loaded_mapping.get('patients', {}),
                        'payers': loaded_mapping.get('payers', {}),
                        'reason_codes': loaded_mapping.get('reason_codes', {})
                    }
                print(f"Loaded existing mapping with {sum(len(v) for v in self.mapping.values())} entries")
        except Exception as e:
            print(f"Could not load existing mapping: {e}")
            self.mapping = {
                'encounters': {},
                'organizations': {},
                'patients': {},
                'payers': {},
                'reason_codes': {}
            }
    
    def save_mapping(self):
        """Save the current anonymization mapping."""
        try:
            Path(self.mapping_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.mapping_file, 'w') as f:
                json.dump(self.mapping, f, indent=2)
            print(f"Saved mapping to {self.mapping_file}")
        except Exception as e:
            print(f"Error saving mapping: {e}")
    
    def remove_columns(self, df):
        """
        Remove specified columns that contain direct identifiers.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with specified columns removed
        """
        columns_to_remove = [
            'SSN', 'DRIVERS', 'PASSPORT',
            'PREFIX', 'FIRST', 'LAST', 'MAIDEN',
            'ADDRESS', 'ZIP', 'LAT', 'LON', 'ADDRESS_payer', 'PHONE_payer',
            'DEATHDATE', 'PROVIDER'
        ]
        
        # Find columns that exist in the dataframe
        existing_columns = [col for col in columns_to_remove if col in df.columns]
        
        if existing_columns:
            print(f"Removing columns: {existing_columns}")
            df_cleaned = df.drop(columns=existing_columns)
        else:
            print("No specified columns found to remove")
            df_cleaned = df.copy()
        
        return df_cleaned
    
    def pseudonymize_encounter_id(self, encounter_id):
        """
        Transform Id_encounter to EN00001, EN00002, etc. format.
        
        Parameters:
        -----------
        encounter_id : str
            Original encounter ID
            
        Returns:
        --------
        str
            Pseudonymized encounter ID
        """
        if pd.isna(encounter_id) or encounter_id == '':
            return encounter_id
        
        if encounter_id in self.mapping['encounters']:
            return self.mapping['encounters'][encounter_id]
        
        # Generate new pseudonymized ID
        next_num = len(self.mapping['encounters']) + 1
        pseudonymized_id = f"EN{next_num:05d}"
        self.mapping['encounters'][encounter_id] = pseudonymized_id
        return pseudonymized_id
    
    def generalize_dates(self, df):
        """
        Generalize date columns to year only.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with generalized dates
        """
        df_generalized = df.copy()
        
        # Date columns to generalize
        date_columns = {
            'START': 'start_year',
            'STOP': 'stop_year',
            'BIRTHDATE': 'birth_year',
            'condition_dates': 'condition_year'
        }
        
        for original_col, new_col in date_columns.items():
            if original_col in df_generalized.columns:
                print(f"Generalizing {original_col} to {new_col}")
                
                # Extract year from date strings
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
    
    def pseudonymize_organization(self, org_id):
        """
        Map ORGANIZATION to Org_001, Org_002, etc. format.
        
        Parameters:
        -----------
        org_id : str
            Original organization ID
            
        Returns:
        --------
        str
            Pseudonymized organization ID
        """
        if pd.isna(org_id) or org_id == '':
            return org_id
        
        if org_id in self.mapping['organizations']:
            return self.mapping['organizations'][org_id]
        
        # Generate new pseudonymized ID
        next_num = len(self.mapping['organizations']) + 1
        pseudonymized_id = f"Org_{next_num:03d}"
        self.mapping['organizations'][org_id] = pseudonymized_id
        return pseudonymized_id
    
    def pseudonymize_patient(self, patient_id):
        """
        Transform Patient to Patient_0001, Patient_0002, etc. format.
        
        Parameters:
        -----------
        patient_id : str
            Original patient ID
            
        Returns:
        --------
        str
            Pseudonymized patient ID
        """
        if pd.isna(patient_id) or patient_id == '':
            return patient_id
        
        if patient_id in self.mapping['patients']:
            return self.mapping['patients'][patient_id]
        
        # Generate new pseudonymized ID
        next_num = len(self.mapping['patients']) + 1
        pseudonymized_id = f"Patient_{next_num:04d}"
        self.mapping['patients'][patient_id] = pseudonymized_id
        return pseudonymized_id
    
    def pseudonymize_id_patient(self, id_patient):
        """
        Transform Id_patient to IDP_0001, IDP_0002, etc. format.
        
        Parameters:
        -----------
        id_patient : str
            Original Id_patient
            
        Returns:
        --------
        str
            Pseudonymized Id_patient
        """
        if pd.isna(id_patient) or id_patient == '':
            return id_patient
        
        if id_patient in self.mapping['patients']:
            return self.mapping['patients'][id_patient]
        
        # Generate new pseudonymized ID
        next_num = len(self.mapping['patients']) + 1
        pseudonymized_id = f"IDP_{next_num:04d}"
        self.mapping['patients'][id_patient] = pseudonymized_id
        return pseudonymized_id
    
    def pseudonymize_payer(self, payer_id):
        """
        Map payer codes to Payer_0001, Payer_0002, etc. format.
        
        Parameters:
        -----------
        payer_id : str
            Original payer ID
            
        Returns:
        --------
        str
            Pseudonymized payer ID
        """
        if pd.isna(payer_id) or payer_id == '':
            return payer_id
        
        if payer_id in self.mapping['payers']:
            return self.mapping['payers'][payer_id]
        
        # Generate new pseudonymized ID
        next_num = len(self.mapping['payers']) + 1
        pseudonymized_id = f"Payer_{next_num:04d}"
        self.mapping['payers'][payer_id] = pseudonymized_id
        return pseudonymized_id
    
    def pseudonymize_reason_code(self, reason_code):
        """
        Map reason_code uniquely while preventing exposure of diagnostic codes.
        
        Parameters:
        -----------
        reason_code : str
            Original reason code
            
        Returns:
        --------
        str
            Pseudonymized reason code
        """
        if pd.isna(reason_code) or reason_code == '':
            return reason_code
        
        if reason_code in self.mapping['reason_codes']:
            return self.mapping['reason_codes'][reason_code]
        
        # Generate new pseudonymized code
        next_num = len(self.mapping['reason_codes']) + 1
        pseudonymized_code = f"RC_{next_num:04d}"
        self.mapping['reason_codes'][reason_code] = pseudonymized_code
        return pseudonymized_code
    
    def anonymize_dataframe(self, df):
        """
        Apply all anonymization transformations to the dataframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Fully anonymized dataframe
        """
        print("Starting healthcare data anonymization...")
        
        # Step 1: Remove specified columns
        print("\n1. Removing direct identifier columns...")
        df_anonymized = self.remove_columns(df)
        
        # Step 2: Pseudonymize encounter IDs
        print("\n2. Pseudonymizing encounter IDs...")
        if 'Id_encounter' in df_anonymized.columns:
            df_anonymized['Id_encounter'] = df_anonymized['Id_encounter'].apply(
                self.pseudonymize_encounter_id
            )
        
        # Step 3: Generalize dates
        print("\n3. Generalizing dates...")
        df_anonymized = self.generalize_dates(df_anonymized)
        
        # Step 4: Pseudonymize organizations
        print("\n4. Pseudonymizing organizations...")
        if 'ORGANIZATION' in df_anonymized.columns:
            df_anonymized['ORGANIZATION'] = df_anonymized['ORGANIZATION'].apply(
                self.pseudonymize_organization
            )
        
        # Step 5: Pseudonymize patients
        print("\n5. Pseudonymizing patients...")
        if 'PATIENT' in df_anonymized.columns:
            df_anonymized['PATIENT'] = df_anonymized['PATIENT'].apply(
                self.pseudonymize_patient
            )
        
        # Step 5b: Pseudonymize Id_patient
        if 'Id_patient' in df_anonymized.columns:
            df_anonymized['Id_patient'] = df_anonymized['Id_patient'].apply(
                self.pseudonymize_id_patient
            )
        
        # Step 6: Pseudonymize payers
        print("\n6. Pseudonymizing payers...")
        if 'PAYER' in df_anonymized.columns:
            df_anonymized['PAYER'] = df_anonymized['PAYER'].apply(
                self.pseudonymize_payer
            )
        
        # Step 7: Pseudonymize reason codes
        print("\n7. Pseudonymizing reason codes...")
        if 'REASONCODE' in df_anonymized.columns:
            df_anonymized['REASONCODE'] = df_anonymized['REASONCODE'].apply(
                self.pseudonymize_reason_code
            )
        
        # Save mapping
        self.save_mapping()
        
        print(f"\nAnonymization complete!")
        print(f"Processed {len(df_anonymized)} records")
        print(f"Generated mappings for:")
        print(f"  - {len(self.mapping['encounters'])} encounters")
        print(f"  - {len(self.mapping['organizations'])} organizations")
        print(f"  - {len(self.mapping['patients'])} patients")
        print(f"  - {len(self.mapping['payers'])} payers")
        print(f"  - {len(self.mapping['reason_codes'])} reason codes")
        
        return df_anonymized
    
    def get_anonymization_summary(self):
        """
        Get a summary of the anonymization mappings.
        
        Returns:
        --------
        dict
            Summary of anonymization mappings
        """
        return {
            'total_mappings': sum(len(v) for v in self.mapping.values()),
            'encounters': len(self.mapping['encounters']),
            'organizations': len(self.mapping['organizations']),
            'patients': len(self.mapping['patients']),
            'payers': len(self.mapping['payers']),
            'reason_codes': len(self.mapping['reason_codes'])
        }
