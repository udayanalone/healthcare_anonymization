import pandas as pd
import numpy as np
import uuid
import hashlib
import json
from pathlib import Path

class Pseudonymizer:
    """
    Class for pseudonymizing direct identifiers while maintaining referential integrity.
    """
    
    def __init__(self, mapping_file=None):
        """
        Initialize the pseudonymizer.
        
        Parameters:
        -----------
        mapping_file : str, optional
            Path to save/load the pseudonymization mapping
        """
        self.mapping = {}
        self.mapping_file = mapping_file or "pseudonymization_mapping.json"
        self.load_mapping()
    
    def load_mapping(self):
        """Load existing pseudonymization mapping if available."""
        try:
            if Path(self.mapping_file).exists():
                with open(self.mapping_file, 'r') as f:
                    self.mapping = json.load(f)
                print(f"Loaded existing mapping with {len(self.mapping)} entries")
        except Exception as e:
            print(f"Could not load existing mapping: {e}")
            self.mapping = {}
    
    def save_mapping(self):
        """Save the current pseudonymization mapping."""
        try:
            with open(self.mapping_file, 'w') as f:
                json.dump(self.mapping, f, indent=2)
            print(f"Saved mapping to {self.mapping_file}")
        except Exception as e:
            print(f"Error saving mapping: {e}")
    
    def pseudonymize_id(self, original_id):
        """
        Pseudonymize a patient ID using UUID.
        
        Parameters:
        -----------
        original_id : str
            Original patient ID
            
        Returns:
        --------
        str
            Pseudonymized ID
        """
        if original_id in self.mapping:
            return self.mapping[original_id]
        
        # Generate new pseudonymized ID
        pseudonymized_id = str(uuid.uuid4())
        self.mapping[original_id] = pseudonymized_id
        return pseudonymized_id
    
    def pseudonymize_ssn(self, ssn):
        """
        Pseudonymize SSN by hashing.
        
        Parameters:
        -----------
        ssn : str
            Original SSN
            
        Returns:
        --------
        str
            Pseudonymized SSN
        """
        if pd.isna(ssn) or ssn == '':
            return ssn
        
        # Hash the SSN to maintain consistency
        hashed_ssn = hashlib.sha256(str(ssn).encode()).hexdigest()[:9]
        # Format as XXX-XX-XXXX
        return f"{hashed_ssn[:3]}-{hashed_ssn[3:5]}-{hashed_ssn[5:9]}"
    
    def pseudonymize_name(self, name):
        """
        Pseudonymize name by replacing with generic identifier.
        
        Parameters:
        -----------
        name : str
            Original name
            
        Returns:
        --------
        str
            Pseudonymized name
        """
        if pd.isna(name) or name == '':
            return name
        
        # Create a consistent hash for the name
        name_hash = hashlib.md5(str(name).encode()).hexdigest()
        
        # Use first 8 characters of hash as a suffix
        suffix = name_hash[:8]
        
        # Return pseudonymized name
        return f"Person_{suffix}"
    
    def pseudonymize_address(self, address):
        """
        Pseudonymize address by replacing with generic identifier.
        
        Parameters:
        -----------
        address : str
            Original address
            
        Returns:
        --------
        str
            Pseudonymized address
        """
        if pd.isna(address) or address == '':
            return address
        
        # Create a consistent hash for the address
        addr_hash = hashlib.md5(str(address).encode()).hexdigest()
        
        # Use first 8 characters of hash as a suffix
        suffix = addr_hash[:8]
        
        # Return pseudonymized address
        return f"Address_{suffix}"
    
    def pseudonymize_direct_identifiers(self, df, id_columns=None, name_columns=None, 
                                      address_columns=None, ssn_columns=None):
        """
        Pseudonymize all direct identifiers in a dataframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe containing direct identifiers
        id_columns : list, optional
            List of ID columns to pseudonymize
        name_columns : list, optional
            List of name columns to pseudonymize
        address_columns : list, optional
            List of address columns to pseudonymize
        ssn_columns : list, optional
            List of SSN columns to pseudonymize
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with pseudonymized direct identifiers
        """
        # Make a copy to avoid modifying the original
        df_pseudo = df.copy()
        
        # Pseudonymize IDs
        if id_columns:
            for col in id_columns:
                if col in df_pseudo.columns:
                    df_pseudo[col] = df_pseudo[col].apply(self.pseudonymize_id)
        
        # Pseudonymize names
        if name_columns:
            for col in name_columns:
                if col in df_pseudo.columns:
                    df_pseudo[col] = df_pseudo[col].apply(self.pseudonymize_name)
        
        # Pseudonymize addresses
        if address_columns:
            for col in address_columns:
                if col in df_pseudo.columns:
                    df_pseudo[col] = df_pseudo[col].apply(self.pseudonymize_address)
        
        # Pseudonymize SSNs
        if ssn_columns:
            for col in ssn_columns:
                if col in df_pseudo.columns:
                    df_pseudo[col] = df_pseudo[col].apply(self.pseudonymize_ssn)
        
        # Save the mapping
        self.save_mapping()
        
        return df_pseudo