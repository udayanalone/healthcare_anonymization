import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class Generator(nn.Module):
    """
    Generator network for Conditional GAN.
    Takes noise vector + condition vector as input and generates synthetic quasi-identifiers.
    """
    
    def __init__(self, noise_dim, condition_dim, target_dim, hidden_dims=[128, 256, 128]):
        """
        Initialize the Generator.
        
        Parameters:
        -----------
        noise_dim : int
            Dimension of noise vector
        condition_dim : int
            Dimension of condition vector (clinical features)
        target_dim : int
            Dimension of target vector (quasi-identifiers to generate)
        hidden_dims : list
            List of hidden layer dimensions
        """
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.target_dim = target_dim
        
        # Input layer: noise + condition
        input_dim = noise_dim + condition_dim
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, target_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1] range
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, noise, condition):
        """
        Forward pass through the generator.
        
        Parameters:
        -----------
        noise : torch.Tensor
            Random noise vector
        condition : torch.Tensor
            Condition vector (clinical features)
            
        Returns:
        --------
        torch.Tensor
            Generated synthetic quasi-identifiers
        """
        # Concatenate noise and condition
        x = torch.cat([noise, condition], dim=1)
        return self.network(x)

class Discriminator(nn.Module):
    """
    Discriminator network for Conditional GAN.
    Takes real/fake quasi-identifiers + condition vector as input and outputs probability.
    """
    
    def __init__(self, target_dim, condition_dim, hidden_dims=[128, 256, 128]):
        """
        Initialize the Discriminator.
        
        Parameters:
        -----------
        target_dim : int
            Dimension of target vector (quasi-identifiers)
        condition_dim : int
            Dimension of condition vector (clinical features)
        hidden_dims : list
            List of hidden layer dimensions
        """
        super(Discriminator, self).__init__()
        
        self.target_dim = target_dim
        self.condition_dim = condition_dim
        
        # Input layer: target + condition
        input_dim = target_dim + condition_dim
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output probability [0, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, target, condition):
        """
        Forward pass through the discriminator.
        
        Parameters:
        -----------
        target : torch.Tensor
            Target vector (real or fake quasi-identifiers)
        condition : torch.Tensor
            Condition vector (clinical features)
            
        Returns:
        --------
        torch.Tensor
            Probability that input is real
        """
        # Concatenate target and condition
        x = torch.cat([target, condition], dim=1)
        return self.network(x)

class ConditionalGAN:
    """
    Conditional GAN for synthetic data generation.
    """
    
    def __init__(self, noise_dim=100, device='auto'):
        """
        Initialize the Conditional GAN.
        
        Parameters:
        -----------
        noise_dim : int
            Dimension of noise vector
        device : str
            Device to use ('auto', 'cpu', 'cuda')
        """
        self.noise_dim = noise_dim
        self.device = self._get_device(device)
        
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        
        self.g_losses = []
        self.d_losses = []
        
        print(f"Using device: {self.device}")
    
    def _get_device(self, device):
        """Determine the best device to use."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)
    
    def build_models(self, condition_dim, target_dim, hidden_dims=[128, 256, 128]):
        """
        Build generator and discriminator models.
        
        Parameters:
        -----------
        condition_dim : int
            Dimension of condition vector (clinical features)
        target_dim : int
            Dimension of target vector (quasi-identifiers)
        hidden_dims : list
            List of hidden layer dimensions
        """
        # Build generator
        self.generator = Generator(
            noise_dim=self.noise_dim,
            condition_dim=condition_dim,
            target_dim=target_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # Build discriminator
        self.discriminator = Discriminator(
            target_dim=target_dim,
            condition_dim=condition_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # Setup optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def train(self, condition_tensor, target_tensor, epochs=500, batch_size=64, save_dir=None):
        """
        Train the Conditional GAN.
        
        Parameters:
        -----------
        condition_tensor : torch.Tensor
            Tensor of condition vectors (clinical features)
        target_tensor : torch.Tensor
            Tensor of target vectors (quasi-identifiers)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        save_dir : str, optional
            Directory to save models and training history
        """
        if self.generator is None or self.discriminator is None:
            raise ValueError("Models not built. Call build_models() first.")
        
        # Create dataset and dataloader
        dataset = TensorDataset(condition_tensor, target_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Binary cross entropy loss
        criterion = nn.BCELoss()
        
        # Learning rate schedulers for better convergence with more epochs
        g_scheduler = optim.lr_scheduler.StepLR(self.g_optimizer, step_size=100, gamma=0.8)
        d_scheduler = optim.lr_scheduler.StepLR(self.d_optimizer, step_size=100, gamma=0.8)
        
        # Training history
        history = {
            'g_losses': [],
            'd_losses': [],
            'epochs': [],
            'g_lr': [],
            'd_lr': []
        }
        
        # Training loop with improved configurations for longer training
        print(f"Starting CGAN training for {epochs} epochs...")
        print(f"Batch size: {batch_size}, Data samples: {len(condition_tensor)}")
        
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            
            for batch_idx, (condition_batch, target_batch) in enumerate(dataloader):
                batch_size = condition_batch.size(0)
                
                # Move data to device
                real_target = target_batch.to(self.device)
                condition = condition_batch.to(self.device)
                
                # Create labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # ---------------------
                # Train Discriminator
                # ---------------------
                self.d_optimizer.zero_grad()
                
                # Train with real data
                real_output = self.discriminator(real_target, condition)
                d_loss_real = criterion(real_output, real_labels)
                
                # Train with fake data
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_target = self.generator(noise, condition)
                fake_output = self.discriminator(fake_target.detach(), condition)
                d_loss_fake = criterion(fake_output, fake_labels)
                
                # Combine losses and update
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()
                
                # ---------------------
                # Train Generator
                # ---------------------
                self.g_optimizer.zero_grad()
                
                # Generate fake data and try to fool discriminator
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_target = self.generator(noise, condition)
                fake_output = self.discriminator(fake_target, condition)
                
                # Generator wants discriminator to think its output is real
                g_loss = criterion(fake_output, real_labels)
                g_loss.backward()
                self.g_optimizer.step()
                
                # Record losses
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            # Calculate average losses for this epoch
            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)
            
            # Update learning rate schedulers
            g_scheduler.step()
            d_scheduler.step()
            
            # Get current learning rates
            current_g_lr = self.g_optimizer.param_groups[0]['lr']
            current_d_lr = self.d_optimizer.param_groups[0]['lr']
            
            # Update history
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            history['g_losses'].append(avg_g_loss)
            history['d_losses'].append(avg_d_loss)
            history['epochs'].append(epoch + 1)
            history['g_lr'].append(current_g_lr)
            history['d_lr'].append(current_d_lr)
            
            # Print progress with more frequent updates for longer training
            if (epoch + 1) % 25 == 0 or epoch == 0 or (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f} | G LR: {current_g_lr:.6f} | D LR: {current_d_lr:.6f}")
            
            # Save models periodically (more frequent saves for longer training)
            if save_dir and ((epoch + 1) % 25 == 0 or epoch == 0):
                self.save_models(save_dir, epoch + 1, history)
        
        # Save final models
        if save_dir:
            self.save_models(save_dir, epochs, history)
        
        return history
    
    def save_models(self, save_dir, epoch, history=None):
        """
        Save trained models and training history.
        
        Parameters:
        -----------
        save_dir : str
            Directory to save models
        epoch : int
            Current epoch number
        history : dict, optional
            Training history
        """
        save_dir = Path(save_dir) / f"cgan_epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save generator
        torch.save(self.generator.state_dict(), save_dir / "generator.pth")
        
        # Save discriminator
        torch.save(self.discriminator.state_dict(), save_dir / "discriminator.pth")
        
        # Save training history
        if history:
            with open(save_dir / "training_history.json", 'w') as f:
                json.dump(history, f, indent=2)
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, load_dir):
        """
        Load trained models.
        
        Parameters:
        -----------
        load_dir : str
            Directory to load models from
        """
        load_dir = Path(load_dir)
        
        # Load generator
        self.generator.load_state_dict(torch.load(load_dir / "generator.pth", map_location=self.device))
        
        # Load discriminator
        self.discriminator.load_state_dict(torch.load(load_dir / "discriminator.pth", map_location=self.device))
        
        print(f"Models loaded from {load_dir}")
    
    def generate_samples(self, condition_tensor, num_samples=None):
        """
        Generate synthetic samples using trained generator.
        
        Parameters:
        -----------
        condition_tensor : torch.Tensor
            Tensor of condition vectors (clinical features)
        num_samples : int, optional
            Number of samples to generate (defaults to condition tensor size)
            
        Returns:
        --------
        torch.Tensor
            Generated synthetic quasi-identifiers
        """
        if self.generator is None:
            raise ValueError("Generator not built or trained. Call build_models() and train() first.")
        
        # Set generator to evaluation mode
        self.generator.eval()
        
        # Determine number of samples
        if num_samples is None:
            num_samples = condition_tensor.size(0)
        else:
            # If num_samples > condition_tensor size, repeat condition tensor
            if num_samples > condition_tensor.size(0):
                repeats = int(np.ceil(num_samples / condition_tensor.size(0)))
                condition_tensor = condition_tensor.repeat(repeats, 1)
            condition_tensor = condition_tensor[:num_samples]
        
        # Move condition to device
        condition = condition_tensor.to(self.device)
        
        # Generate noise
        noise = torch.randn(num_samples, self.noise_dim).to(self.device)
        
        # Generate samples
        with torch.no_grad():
            generated_samples = self.generator(noise, condition)
        
        return generated_samples.cpu()
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if not self.g_losses or not self.d_losses:
            print("No training history to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('CGAN Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        
        plt.show()