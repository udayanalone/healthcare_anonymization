#!/usr/bin/env python3
"""
CGAN Training Monitor
Monitors the progress of CGAN training and displays real-time statistics.
"""

import os
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def check_training_status(model_dir="models"):
    """Check the current status of CGAN training."""
    model_path = Path(model_dir)
    
    print("=" * 60)
    print("CGAN TRAINING MONITOR")
    print("=" * 60)
    
    # Check if model directory exists
    if not model_path.exists():
        print("❌ Model directory not found. Training may not have started.")
        return False
    
    # Check for training configuration
    config_file = model_path / 'config' / 'training_config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"📋 Training Configuration:")
        print(f"   Epochs: {config.get('epochs', 'Unknown')}")
        print(f"   Batch Size: {config.get('batch_size', 'Unknown')}")
        print(f"   Records: {config.get('records_processed', 'Unknown')}")
        print(f"   Learning Rate: {config.get('learning_rate', 'Unknown')}")
        print(f"   Started: {config.get('timestamp', 'Unknown')}")
        print()
    
    # Check for model files
    generator_dir = model_path / 'generator'
    discriminator_dir = model_path / 'discriminator'
    
    generator_files = list(generator_dir.glob('*.pth')) if generator_dir.exists() else []
    discriminator_files = list(discriminator_dir.glob('*.pth')) if discriminator_dir.exists() else []
    
    print(f"🤖 Model Files:")
    print(f"   Generator checkpoints: {len(generator_files)}")
    print(f"   Discriminator checkpoints: {len(discriminator_files)}")
    
    if generator_files:
        latest_gen = max(generator_files, key=os.path.getmtime)
        print(f"   Latest generator: {latest_gen.name}")
    
    if discriminator_files:
        latest_disc = max(discriminator_files, key=os.path.getmtime)
        print(f"   Latest discriminator: {latest_disc.name}")
    
    # Check for plots
    plots_dir = model_path / 'plots'
    plot_files = list(plots_dir.glob('*.png')) if plots_dir.exists() else []
    print(f"📊 Training plots: {len(plot_files)}")
    
    # Check for sample data
    sample_file = model_path / 'sample_generated_data.csv'
    if sample_file.exists():
        print(f"📄 Sample generated data: Available")
        try:
            sample_df = pd.read_csv(sample_file)
            print(f"   Sample size: {len(sample_df)} records")
        except:
            print(f"   Sample data: Error reading file")
    else:
        print(f"📄 Sample generated data: Not available yet")
    
    # Check for training history
    history_files = list(model_path.glob('*history*.json'))
    if history_files:
        print(f"📈 Training history files: {len(history_files)}")
        try:
            with open(history_files[0], 'r') as f:
                history = json.load(f)
            if 'epochs' in history and history['epochs']:
                current_epoch = max(history['epochs'])
                print(f"   Current epoch: {current_epoch}")
                if 'g_losses' in history and history['g_losses']:
                    print(f"   Latest G loss: {history['g_losses'][-1]:.4f}")
                if 'd_losses' in history and history['d_losses']:
                    print(f"   Latest D loss: {history['d_losses'][-1]:.4f}")
        except:
            print(f"   Training history: Error reading file")
    
    print("=" * 60)
    return True

def monitor_continuously(model_dir="models", interval=30):
    """Continuously monitor training progress."""
    print(f"Starting continuous monitoring (checking every {interval} seconds)...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            check_training_status(model_dir)
            print(f"\n⏰ Next check in {interval} seconds...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor CGAN training progress')
    parser.add_argument('--model-dir', type=str, default='models', help='Model directory to monitor')
    parser.add_argument('--continuous', action='store_true', help='Monitor continuously')
    parser.add_argument('--interval', type=int, default=30, help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    if args.continuous:
        monitor_continuously(args.model_dir, args.interval)
    else:
        check_training_status(args.model_dir)
