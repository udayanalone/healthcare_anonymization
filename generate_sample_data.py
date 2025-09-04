#!/usr/bin/env python3
"""
Generate sample anonymized data using the trained CGAN model.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('src')
from gan_model import ConditionalGAN
from data_preprocessing import DataPreprocessor
import joblib

def main():
    """Generate sample data using the trained CGAN."""
    print("=" * 60)
    print("GENERATING SAMPLE ANONYMIZED DATA")
    print("=" * 60)
    
    # Load the trained model
    model_dir = Path('models')
    cgan = ConditionalGAN(noise_dim=100, device='cpu')
    cgan.build_models(condition_dim=10, target_dim=16, hidden_dims=[128, 256, 128])
    
    # Load the final trained models
    print("Loading trained models...")
    cgan.generator.load_state_dict(torch.load(model_dir / 'cgan_epoch_500' / 'generator.pth', map_location='cpu'))
    cgan.discriminator.load_state_dict(torch.load(model_dir / 'cgan_epoch_500' / 'discriminator.pth', map_location='cpu'))
    
    # Load scaler and encoder
    scaler = joblib.load(model_dir / 'scaler.pkl')
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    df = pd.read_csv('data/unified_5000_records.csv').head(1000)
    condition_tensor, target_tensor, _, _ = preprocessor.preprocess_for_gan(df)
    
    # Generate sample data
    print("Generating sample anonymized data...")
    with torch.no_grad():
        sample_conditions = condition_tensor[:100]
        generated_data = cgan.generate_samples(sample_conditions, num_samples=100)
        
        # Convert back to original scale
        generated_np = generated_data.cpu().numpy()
        generated_original = scaler.inverse_transform(generated_np)
        
        # Define target features
        target_features = [
            'ENCOUNTERCLASS', 'CODE', 'DESCRIPTION', 'BASE_ENCOUNTER_COST',
            'TOTAL_CLAIM_COST', 'PAYER_COVERAGE', 'REASONCODE', 'REASONDESCRIPTION',
            'PAYER', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'REVENUE',
            'UTILIZATION_org', 'AMOUNT_COVERED', 'AMOUNT_UNCOVERED', 'REVENUE_payer'
        ]
        
        # Create DataFrame
        sample_df = pd.DataFrame(generated_original, columns=target_features)
        sample_df.to_csv(model_dir / 'sample_generated_data.csv', index=False)
        
        print(f"✅ Sample generated data saved to {model_dir / 'sample_generated_data.csv'}")
        print(f"📊 Generated {len(sample_df)} records with {len(sample_df.columns)} features")
        print("\nSample data preview:")
        print(sample_df.head())
        
        print("\n" + "=" * 60)
        print("SAMPLE DATA GENERATION COMPLETED!")
        print("=" * 60)

if __name__ == "__main__":
    main()
