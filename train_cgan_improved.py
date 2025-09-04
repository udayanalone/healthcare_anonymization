#!/usr/bin/env python3
"""
Improved CGAN Training Script for Healthcare Data Anonymization
This script trains a Conditional GAN with enhanced features for better performance.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# Add src directory to path
sys.path.append('src')
from gan_model import ConditionalGAN
from data_preprocessing import DataPreprocessor

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train improved CGAN for healthcare data anonymization')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--records', type=int, default=1000, help='Number of records to process')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--learning-rate', type=float, default=0.0002, help='Learning rate for optimizers')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    return parser.parse_args()

def setup_device(use_cuda=True):
    """Setup computing device."""
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    return device

def load_and_preprocess_data(file_path, num_records=None):
    """Load and preprocess the healthcare data."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    if num_records:
        df = df.head(num_records)
        print(f"Using first {num_records} records")
    
    print(f"Dataset shape: {df.shape}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess the data
    print("Preprocessing data...")
    condition_tensor, target_tensor, scaler, encoder = preprocessor.preprocess_for_gan(df)
    
    print(f"Condition tensor shape: {condition_tensor.shape}")
    print(f"Target tensor shape: {target_tensor.shape}")
    
    return condition_tensor, target_tensor, scaler, encoder, df

def create_model_directories(model_dir):
    """Create necessary directories for model saving."""
    model_path = Path(model_dir)
    model_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    (model_path / 'generator').mkdir(exist_ok=True)
    (model_path / 'discriminator').mkdir(exist_ok=True)
    (model_path / 'plots').mkdir(exist_ok=True)
    (model_path / 'config').mkdir(exist_ok=True)
    
    return model_path

def save_training_config(args, model_dir, condition_dim, target_dim):
    """Save training configuration."""
    config = {
        'input_file': args.input,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'records_processed': args.records,
        'learning_rate': args.learning_rate,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'condition_dim': condition_dim,
        'target_dim': target_dim,
        'timestamp': datetime.now().isoformat(),
        'device': str(torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'))
    }
    
    config_path = Path(model_dir) / 'config' / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training configuration saved to {config_path}")

def plot_training_history(history, model_dir):
    """Plot and save training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['epochs'], history['g_losses'], label='Generator Loss', alpha=0.7)
    axes[0, 0].plot(history['epochs'], history['d_losses'], label='Discriminator Loss', alpha=0.7)
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Learning rate curves
    axes[0, 1].plot(history['epochs'], history['g_lr'], label='Generator LR', alpha=0.7)
    axes[0, 1].plot(history['epochs'], history['d_lr'], label='Discriminator LR', alpha=0.7)
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Loss ratio
    loss_ratio = [g/d if d > 0 else 0 for g, d in zip(history['g_losses'], history['d_losses'])]
    axes[1, 0].plot(history['epochs'], loss_ratio, label='G/D Loss Ratio', alpha=0.7)
    axes[1, 0].set_title('Generator/Discriminator Loss Ratio')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss difference
    loss_diff = [abs(g - d) for g, d in zip(history['g_losses'], history['d_losses'])]
    axes[1, 1].plot(history['epochs'], loss_diff, label='|G Loss - D Loss|', alpha=0.7)
    axes[1, 1].set_title('Loss Difference')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Absolute Difference')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(model_dir) / 'plots' / 'training_history.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {plot_path}")
    
    plt.close()

def main():
    """Main training function."""
    args = parse_arguments()
    
    print("=" * 60)
    print("IMPROVED CGAN TRAINING FOR HEALTHCARE DATA ANONYMIZATION")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Records: {args.records}")
    print(f"Model directory: {args.model_dir}")
    print("=" * 60)
    
    # Setup device
    device = setup_device(not args.no_cuda)
    
    # Create model directories
    model_dir = create_model_directories(args.model_dir)
    
    try:
        # Load and preprocess data
        condition_tensor, target_tensor, scaler, encoder, df = load_and_preprocess_data(
            args.input, args.records
        )
        
        # Move tensors to device
        condition_tensor = condition_tensor.to(device)
        target_tensor = target_tensor.to(device)
        
        # Save training configuration
        save_training_config(args, model_dir, condition_tensor.shape[1], target_tensor.shape[1])
        
        # Initialize CGAN
        print("\nInitializing Conditional GAN...")
        cgan = ConditionalGAN(
            noise_dim=100,
            device=device
        )
        
        # Build the networks with proper dimensions
        cgan.build_models(
            condition_dim=condition_tensor.shape[1],
            target_dim=target_tensor.shape[1],
            hidden_dims=[128, 256, 128]
        )
        
        # Update optimizers with custom learning rates
        cgan.g_optimizer = optim.Adam(cgan.generator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
        cgan.d_optimizer = optim.Adam(cgan.discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
        
        print(f"Generator parameters: {sum(p.numel() for p in cgan.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in cgan.discriminator.parameters()):,}")
        
        # Train the model
        print(f"\nStarting training for {args.epochs} epochs...")
        history = cgan.train(
            condition_tensor=condition_tensor,
            target_tensor=target_tensor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=model_dir
        )
        
        # Save final models
        print("\nSaving final models...")
        cgan.save_models(model_dir, args.epochs, history)
        
        # Save scaler and encoder
        import joblib
        joblib.dump(scaler, model_dir / 'scaler.pkl')
        joblib.dump(encoder, model_dir / 'encoder.pkl')
        
        # Plot training history
        plot_training_history(history, model_dir)
        
        # Generate sample data
        print("\nGenerating sample anonymized data...")
        with torch.no_grad():
            sample_conditions = condition_tensor[:100]  # First 100 samples
            generated_data = cgan.generate_samples(sample_conditions, num_samples=100)
            
            # Convert back to original scale
            generated_np = generated_data.cpu().numpy()
            generated_original = scaler.inverse_transform(generated_np)
            
            # Define target features (clinical and financial data)
            target_features = [
                'ENCOUNTERCLASS', 'CODE', 'DESCRIPTION', 'BASE_ENCOUNTER_COST',
                'TOTAL_CLAIM_COST', 'PAYER_COVERAGE', 'REASONCODE', 'REASONDESCRIPTION',
                'PAYER', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'REVENUE',
                'UTILIZATION_org', 'AMOUNT_COVERED', 'AMOUNT_UNCOVERED', 'REVENUE_payer'
            ]
            
            # Create DataFrame with sample data using only target features
            sample_df = pd.DataFrame(generated_original, columns=target_features)
            sample_df.to_csv(model_dir / 'sample_generated_data.csv', index=False)
            print(f"Sample generated data saved to {model_dir / 'sample_generated_data.csv'}")
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Final Generator Loss: {history['g_losses'][-1]:.4f}")
        print(f"Final Discriminator Loss: {history['d_losses'][-1]:.4f}")
        print(f"Models saved to: {model_dir}")
        print(f"Training plots saved to: {model_dir / 'plots'}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
