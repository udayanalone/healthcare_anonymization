import argparse
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
import json
import joblib

from src.pseudonymization import Pseudonymizer
from src.gan_model import ConditionalGAN
from src.data_processor import DataProcessor

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Healthcare Data Anonymization')
    
    # Input/output arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='output/anonymized_data.csv',
                        help='Path to output anonymized CSV file')
    parser.add_argument('--mapping', type=str, default='output/pseudonymization_mapping.json',
                        help='Path to save pseudonymization mapping')
    
    # Processing arguments
    parser.add_argument('--records', type=int, default=None,
                        help='Number of records to process (None for all)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    
    # Model arguments
    parser.add_argument('--train', action='store_true',
                        help='Train CGAN model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--noise-dim', type=int, default=100,
                        help='Dimension of noise vector')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save/load models')
    parser.add_argument('--model-name', type=str, default='latest',
                        help='Name of model to load (latest or specific epoch)')
    
    return parser.parse_args()

def load_config(config_path):
    """
    Load configuration from JSON file.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def main():
    """
    Main function to run the anonymization process.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    output_dir = Path(os.path.dirname(args.output))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Healthcare Data Anonymization ===\n")
    
    # Step 1: Load and process data
    print("\n--- Step 1: Loading and Processing Data ---")
    processor = DataProcessor(output_dir=output_dir)
    
    # Load data
    df = processor.load_data(args.input, n_records=args.records)
    if df is None:
        print("Error loading data. Exiting.")
        return
    
    # Get feature categories from config or use default
    direct_identifiers = config.get('direct_identifiers', None)
    quasi_identifiers = config.get('quasi_identifiers', None)
    clinical_data = config.get('clinical_data', None)
    
    # Categorize features
    feature_categories = processor.categorize_features(
        df, direct_identifiers, quasi_identifiers, clinical_data
    )
    
    # Preprocess data
    processed_data = processor.preprocess_data(df)
    
    # Save preprocessed data
    processor.save_preprocessed_data(processed_data)
    
    # Step 2: Pseudonymize direct identifiers
    print("\n--- Step 2: Pseudonymizing Direct Identifiers ---")
    pseudonymizer = Pseudonymizer(mapping_file=args.mapping)
    
    # Get direct identifiers from feature categories
    direct_ids = feature_categories['direct_identifiers']
    
    # Identify specific types of direct identifiers
    id_columns = [col for col in direct_ids if 'ID' in col.upper()]
    name_columns = [col for col in direct_ids if 'NAME' in col.upper() or 'FIRST' in col.upper() or 'LAST' in col.upper()]
    address_columns = [col for col in direct_ids if 'ADDRESS' in col.upper()]
    ssn_columns = [col for col in direct_ids if 'SSN' in col.upper()]
    
    # Pseudonymize direct identifiers
    df_pseudo = pseudonymizer.pseudonymize_direct_identifiers(
        df, id_columns, name_columns, address_columns, ssn_columns
    )
    
    print(f"Pseudonymized {len(direct_ids)} direct identifiers")
    
    # Step 3: Anonymize quasi-identifiers with CGAN
    print("\n--- Step 3: Anonymizing Quasi-identifiers with CGAN ---")
    
    # Initialize CGAN
    cgan = ConditionalGAN(noise_dim=args.noise_dim)
    
    # Get dimensions
    condition_dim = processed_data['condition_features'].shape[1]
    target_dim = processed_data['target_features'].shape[1]
    
    # Build models
    cgan.build_models(condition_dim, target_dim)
    
    # Train or load model
    if args.train:
        print(f"Training CGAN for {args.epochs} epochs...")
        history = cgan.train(
            processed_data['condition_tensor'],
            processed_data['target_tensor'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=args.model_dir
        )
        
        # Plot training history
        cgan.plot_training_history(save_path=output_dir / 'training_history.png')
    else:
        print(f"Loading trained CGAN model: {args.model_name}")
        if args.model_name == 'latest':
            # Find the latest model
            model_dirs = list(model_dir.glob('cgan_epoch_*'))
            if not model_dirs:
                print("No trained models found. Please train a model first.")
                return
            model_path = max(model_dirs, key=lambda x: int(x.name.split('_')[-1]))
        else:
            model_path = model_dir / args.model_name
        
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return
        
        cgan.load_models(model_path)
    
    # Generate synthetic quasi-identifiers
    print("Generating synthetic quasi-identifiers...")
    synthetic_data = cgan.generate_samples(
        processed_data['condition_tensor']
    ).numpy()
    
    # Step 4: Combine pseudonymized direct identifiers and synthetic quasi-identifiers
    print("\n--- Step 4: Creating Final Anonymized Dataset ---")
    
    # Inverse transform synthetic data
    df_anonymized = processor.inverse_transform_data(synthetic_data, df_pseudo)
    
    # Save anonymized data
    df_anonymized.to_csv(args.output, index=False)
    print(f"Anonymized data saved to {args.output}")
    
    print("\n=== Anonymization Complete ===\n")

if __name__ == "__main__":
    main()