# Conditional GAN for Healthcare Data Anonymization

This document provides detailed technical documentation for the Conditional Generative Adversarial Network (CGAN) implementation used in the healthcare data anonymization system.

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Workflow](#workflow)
- [Generation Conditions](#generation-conditions)
- [Implementation Details](#implementation-details)
- [Training Process](#training-process)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

## Overview

The Conditional GAN (CGAN) in this system is designed to generate synthetic quasi-identifiers while preserving statistical relationships with clinical data. Unlike traditional anonymization techniques that simply generalize or perturb data, the CGAN learns the underlying distribution of the data and generates new samples that maintain utility for analysis while protecting individual privacy.

## Model Architecture

### Generator

The Generator network takes two inputs:
1. A random noise vector (z)
2. Condition vector (clinical data features)

**Architecture:**
- Input Layer: Concatenated noise vector and condition vector
- Hidden Layers: Multiple fully connected layers with batch normalization and LeakyReLU activation
- Output Layer: Fully connected layer with tanh activation (scaled to feature range)

```python
class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim, output_dim, hidden_dim=128):
        super(Generator, self).__init__()
        self.input_dim = noise_dim + condition_dim
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Tanh()
        )
    
    def forward(self, noise, condition):
        # Concatenate noise and condition
        x = torch.cat([noise, condition], dim=1)
        return self.model(x)
```

### Discriminator

The Discriminator network takes two inputs:
1. Real or generated quasi-identifier features
2. Condition vector (clinical data features)

**Architecture:**
- Input Layer: Concatenated feature vector and condition vector
- Hidden Layers: Multiple fully connected layers with dropout and LeakyReLU activation
- Output Layer: Single neuron with sigmoid activation for binary classification

```python
class Discriminator(nn.Module):
    def __init__(self, feature_dim, condition_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.input_dim = feature_dim + condition_dim
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, condition):
        # Concatenate features and condition
        x = torch.cat([features, condition], dim=1)
        return self.model(x)
```

## Workflow

The CGAN workflow consists of the following steps:

### 1. Data Preprocessing

- **Feature Categorization**: Separate features into quasi-identifiers (to be generated) and clinical data (conditions)
- **Encoding**: Convert categorical features to numerical using label encoding
- **Scaling**: Normalize numerical features to [-1, 1] range using MinMaxScaler
- **Tensor Conversion**: Convert data to PyTorch tensors

### 2. Model Training

- **Initialization**: Create Generator and Discriminator networks
- **Training Loop**: For each epoch:
  - Sample random noise vectors
  - Generate synthetic quasi-identifiers using clinical data as conditions
  - Train Discriminator to distinguish real from generated data
  - Train Generator to produce data that fools the Discriminator
- **Model Saving**: Save trained models for later use

### 3. Synthetic Data Generation

- **Load Model**: Load trained Generator model
- **Generate Data**: For each record:
  - Sample random noise vector
  - Use clinical data as condition
  - Generate synthetic quasi-identifiers
- **Inverse Transform**: Convert generated data back to original format

### 4. Final Dataset Creation

- **Combine Data**: Merge pseudonymized direct identifiers, synthetic quasi-identifiers, and original clinical data
- **Post-processing**: Apply any necessary formatting or constraints

## Generation Conditions

The CGAN uses clinical data as conditions to generate appropriate quasi-identifiers. This ensures that the generated data maintains the statistical relationships between clinical and demographic features.

### Condition Types

1. **Categorical Clinical Conditions**

   These include diagnosis codes, procedure codes, and other categorical clinical features.

   **Example:**
   ```
   Condition: DIAGNOSIS_CODE = "I25.10" (Coronary artery disease)
   Generated: AGE_RANGE = "70+", GENDER = "M", ZIP = "02108"
   ```

   In this example, the CGAN has learned that coronary artery disease is more common in older males, and generates appropriate demographic data.

2. **Numerical Clinical Conditions**

   These include lab values, vital signs, and other numerical clinical features.

   **Example:**
   ```
   Condition: BLOOD_PRESSURE_SYSTOLIC = 160, BLOOD_PRESSURE_DIASTOLIC = 95
   Generated: AGE_RANGE = "65-74", GENDER = "F", BMI = 32.4
   ```

   Here, the CGAN has learned the correlation between high blood pressure, age, gender, and BMI.

3. **Multiple Clinical Conditions**

   The CGAN can handle multiple clinical conditions simultaneously.

   **Example:**
   ```
   Conditions: DIAGNOSIS_CODE = "E11.9" (Type 2 diabetes), MEDICATION = "Metformin"
   Generated: AGE_RANGE = "55-64", GENDER = "F", RACE = "White", INCOME = "Medium"
   ```

   This demonstrates how the CGAN captures complex relationships between multiple clinical features and demographics.

### Condition Workflow for Each Feature Type

#### Age Range Generation

1. **Preprocessing**: Original BIRTHDATE is converted to AGE_RANGE categories (e.g., "0-17", "18-34", "35-44", "45-54", "55-64", "65-74", "75+")
2. **Condition Input**: Clinical features (diagnoses, procedures, medications)
3. **Generation Process**:
   - The Generator receives clinical conditions and random noise
   - Outputs a probability distribution across age ranges
   - The most probable age range is selected

**Example:**
```python
# Original data
original_data = {
    "BIRTHDATE": "1945-03-15",
    "DIAGNOSIS_CODE": "I25.10",
    "PROCEDURE_CODE": "33533"
}

# Preprocessing
age = 2023 - 1945 = 78
age_range = "75+"

# CGAN generation with clinical conditions
conditions = [encoded_diagnosis, encoded_procedure]
# Generated output: AGE_RANGE = "75+"
```

#### Gender Generation

1. **Preprocessing**: GENDER is encoded (e.g., "M" = 0, "F" = 1)
2. **Condition Input**: Clinical features and other generated quasi-identifiers
3. **Generation Process**:
   - The Generator produces a probability for each gender
   - The gender with highest probability is selected

**Example:**
```python
# Original data
original_data = {
    "GENDER": "M",
    "DIAGNOSIS_CODE": "I25.10"
}

# CGAN generation with clinical conditions
conditions = [encoded_diagnosis]
# Generated output: GENDER = "M"
```

#### Geographic Data Generation

1. **Preprocessing**: ZIP codes are encoded and normalized
2. **Condition Input**: Clinical features, age range, gender
3. **Generation Process**:
   - The Generator produces synthetic ZIP values
   - Values are transformed back to valid ZIP codes
   - Optional: Generalization to first 3 digits for enhanced privacy

**Example:**
```python
# Original data
original_data = {
    "ZIP": "02108",
    "DIAGNOSIS_CODE": "I25.10",
    "AGE_RANGE": "75+",
    "GENDER": "M"
}

# CGAN generation with clinical conditions
conditions = [encoded_diagnosis, encoded_age_range, encoded_gender]
# Generated output: ZIP = "02110" (nearby ZIP in same area)
```

## Implementation Details

### ConditionalGAN Class

The `ConditionalGAN` class encapsulates the entire CGAN functionality:

```python
class ConditionalGAN:
    def __init__(self, noise_dim=100, hidden_dim=128, lr=0.0002, beta1=0.5, device=None):
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.beta1 = beta1
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.criterion = nn.BCELoss()
        
    def initialize_networks(self, feature_dim, condition_dim):
        # Initialize Generator and Discriminator
        self.generator = Generator(self.noise_dim, condition_dim, feature_dim, self.hidden_dim).to(self.device)
        self.discriminator = Discriminator(feature_dim, condition_dim, self.hidden_dim).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
    
    def train(self, dataloader, epochs, feature_dim, condition_dim):
        # Training implementation
        
    def generate(self, conditions, num_samples=1):
        # Generation implementation
        
    def save_model(self, path):
        # Save model implementation
        
    def load_model(self, path):
        # Load model implementation
```

### Key Methods

#### Training Method

```python
def train(self, dataloader, epochs, feature_dim, condition_dim):
    self.initialize_networks(feature_dim, condition_dim)
    
    for epoch in range(epochs):
        for i, (real_features, conditions) in enumerate(dataloader):
            batch_size = real_features.size(0)
            real_features = real_features.to(self.device)
            conditions = conditions.to(self.device)
            
            # Labels
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            
            # Train Discriminator
            self.d_optimizer.zero_grad()
            
            # Real data
            d_real_output = self.discriminator(real_features, conditions)
            d_real_loss = self.criterion(d_real_output, real_labels)
            
            # Fake data
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_features = self.generator(noise, conditions)
            d_fake_output = self.discriminator(fake_features.detach(), conditions)
            d_fake_loss = self.criterion(d_fake_output, fake_labels)
            
            # Combined loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.d_optimizer.step()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            g_output = self.discriminator(fake_features, conditions)
            g_loss = self.criterion(g_output, real_labels)
            g_loss.backward()
            self.g_optimizer.step()
            
            # Print progress
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], '
                      f'd_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
```

#### Generation Method

```python
def generate(self, conditions, num_samples=1):
    self.generator.eval()
    with torch.no_grad():
        conditions = conditions.to(self.device)
        batch_size = conditions.size(0)
        
        # Repeat conditions if generating multiple samples per condition
        if num_samples > 1:
            conditions = conditions.repeat(num_samples, 1)
            batch_size = conditions.size(0)
        
        # Generate random noise
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        
        # Generate synthetic features
        generated_features = self.generator(noise, conditions)
        
    return generated_features.cpu().numpy()
```

## Training Process

### Data Preparation

```python
# Prepare data for CGAN training
def prepare_data_for_cgan(self, data):
    # Separate features and conditions
    quasi_identifiers = data[self.quasi_identifiers].copy()
    clinical_data = data[self.clinical_data].copy()
    
    # Encode categorical features
    for col in quasi_identifiers.select_dtypes(include=['object']).columns:
        encoder = LabelEncoder()
        quasi_identifiers[col] = encoder.fit_transform(quasi_identifiers[col])
        self.encoders[col] = encoder
    
    for col in clinical_data.select_dtypes(include=['object']).columns:
        encoder = LabelEncoder()
        clinical_data[col] = encoder.fit_transform(clinical_data[col])
        self.encoders[col] = encoder
    
    # Scale numerical features to [-1, 1]
    for col in quasi_identifiers.select_dtypes(include=['number']).columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        quasi_identifiers[col] = scaler.fit_transform(quasi_identifiers[col].values.reshape(-1, 1))
        self.scalers[col] = scaler
    
    for col in clinical_data.select_dtypes(include=['number']).columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        clinical_data[col] = scaler.fit_transform(clinical_data[col].values.reshape(-1, 1))
        self.scalers[col] = scaler
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(quasi_identifiers.values)
    conditions_tensor = torch.FloatTensor(clinical_data.values)
    
    # Create dataset and dataloader
    dataset = TensorDataset(features_tensor, conditions_tensor)
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    return dataloader, quasi_identifiers.shape[1], clinical_data.shape[1]
```

### Training Execution

```python
# Train the CGAN model
def train_cgan(self, data):
    print("Preparing data for CGAN training...")
    dataloader, feature_dim, condition_dim = self.prepare_data_for_cgan(data)
    
    print(f"Initializing CGAN with feature_dim={feature_dim}, condition_dim={condition_dim}")
    self.cgan = ConditionalGAN(noise_dim=self.noise_dim, hidden_dim=self.hidden_dim)
    
    print(f"Training CGAN for {self.epochs} epochs...")
    self.cgan.train(dataloader, self.epochs, feature_dim, condition_dim)
    
    # Save the trained model
    os.makedirs(self.model_dir, exist_ok=True)
    model_path = os.path.join(self.model_dir, f"cgan_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
    self.cgan.save_model(model_path)
    print(f"CGAN model saved to {model_path}")
    
    return model_path
```

## Hyperparameter Tuning

The CGAN model's performance is sensitive to hyperparameter settings. Here are the key hyperparameters and their effects:

| Hyperparameter | Description | Default | Effect |
|----------------|-------------|---------|--------|
| `noise_dim` | Dimension of noise vector | 100 | Higher values increase model capacity but require more data |
| `hidden_dim` | Size of hidden layers | 128 | Larger values increase model capacity but may lead to overfitting |
| `lr` | Learning rate | 0.0002 | Controls step size during optimization |
| `beta1` | Adam optimizer beta1 | 0.5 | Controls exponential decay rate for first moment estimates |
| `epochs` | Number of training epochs | 100 | More epochs improve results but may lead to overfitting |
| `batch_size` | Batch size for training | 64 | Larger batches provide more stable gradients but require more memory |

### Recommended Settings for Different Scenarios

| Scenario | noise_dim | hidden_dim | epochs | batch_size |
|----------|-----------|------------|--------|------------|
| Small dataset (<1000 records) | 50 | 64 | 200 | 32 |
| Medium dataset (1000-10000) | 100 | 128 | 100 | 64 |
| Large dataset (>10000) | 200 | 256 | 50 | 128 |
| High privacy requirements | 150 | 192 | 150 | 64 |
| High utility requirements | 100 | 128 | 200 | 64 |

## Evaluation Metrics

The CGAN model's performance is evaluated using several metrics:

### 1. Data Utility

Measures how well the synthetic data preserves statistical properties of the original data.

```python
def evaluate_data_utility(self):
    # Calculate statistical similarity between original and synthetic quasi-identifiers
    utility_scores = {}
    
    for col in self.quasi_identifiers:
        if col in self.original_data.columns and col in self.anonymized_data.columns:
            if pd.api.types.is_numeric_dtype(self.original_data[col]):
                # For numerical features, compare distributions using KS test
                ks_stat, p_value = ks_2samp(self.original_data[col].dropna(), self.anonymized_data[col].dropna())
                utility_scores[col] = 1 - ks_stat  # Higher score = more similar distributions
            else:
                # For categorical features, compare distributions using chi-squared test
                orig_counts = self.original_data[col].value_counts(normalize=True)
                anon_counts = self.anonymized_data[col].value_counts(normalize=True)
                
                # Align the categories
                categories = list(set(orig_counts.index) | set(anon_counts.index))
                orig_dist = np.array([orig_counts.get(cat, 0) for cat in categories])
                anon_dist = np.array([anon_counts.get(cat, 0) for cat in categories])
                
                # Calculate Jensen-Shannon divergence
                js_div = jensenshannon(orig_dist, anon_dist)
                utility_scores[col] = 1 - js_div  # Higher score = more similar distributions
    
    # Overall utility score (average across features)
    overall_utility = np.mean(list(utility_scores.values()))
    
    return {
        "overall_utility_score": overall_utility,
        "feature_utility_scores": utility_scores
    }
```

### 2. Privacy Protection

Assesses the risk of re-identification using k-anonymity.

```python
def evaluate_privacy_protection(self):
    # Calculate k-anonymity for the anonymized data
    # k-anonymity is the minimum number of records that share the same combination of quasi-identifiers
    
    # Group by all quasi-identifiers
    grouped = self.anonymized_data.groupby(self.quasi_identifiers).size().reset_index(name='count')
    
    # Find the minimum group size (k)
    k = grouped['count'].min()
    
    # Count groups with different k values
    k_counts = grouped['count'].value_counts().sort_index()
    
    return {
        "k_anonymity": k,
        "k_distribution": k_counts.to_dict()
    }
```

### 3. Information Preservation

Measures how well the relationships between quasi-identifiers and clinical data are preserved.

```python
def evaluate_information_preservation(self):
    # Calculate mutual information between quasi-identifiers and clinical data
    # in both original and anonymized datasets
    
    mi_scores_original = {}
    mi_scores_anonymized = {}
    
    for qi_col in self.quasi_identifiers:
        for cd_col in self.clinical_data:
            if qi_col in self.original_data.columns and cd_col in self.original_data.columns:
                # Calculate mutual information in original data
                mi_original = mutual_info_score(
                    self.original_data[qi_col].astype(str),
                    self.original_data[cd_col].astype(str)
                )
                mi_scores_original[(qi_col, cd_col)] = mi_original
            
            if qi_col in self.anonymized_data.columns and cd_col in self.anonymized_data.columns:
                # Calculate mutual information in anonymized data
                mi_anonymized = mutual_info_score(
                    self.anonymized_data[qi_col].astype(str),
                    self.anonymized_data[cd_col].astype(str)
                )
                mi_scores_anonymized[(qi_col, cd_col)] = mi_anonymized
    
    # Calculate preservation ratio for each pair
    preservation_ratios = {}
    for pair in mi_scores_original.keys():
        if pair in mi_scores_anonymized and mi_scores_original[pair] > 0:
            preservation_ratios[pair] = mi_scores_anonymized[pair] / mi_scores_original[pair]
    
    # Overall information preservation score
    overall_preservation = np.mean(list(preservation_ratios.values())) if preservation_ratios else 0
    
    return {
        "overall_information_preservation": overall_preservation,
        "pair_preservation_ratios": preservation_ratios
    }
```

## Usage Examples

### Example 1: Basic Usage with Default Settings

```python
from src.data_processor import DataProcessor
from src.gan_model import ConditionalGAN

# Load and process data
data_processor = DataProcessor(config_path="config/default_config.json")
data = data_processor.load_data("data/healthcare_data.csv")
processed_data = data_processor.preprocess_data(data)

# Train CGAN model
cgan = ConditionalGAN()
dataloader, feature_dim, condition_dim = data_processor.prepare_data_for_cgan(processed_data)
cgan.train(dataloader, epochs=100, feature_dim=feature_dim, condition_dim=condition_dim)

# Generate synthetic quasi-identifiers
conditions = data_processor.get_conditions(processed_data)
synthetic_features = cgan.generate(conditions)

# Create anonymized dataset
anonymized_data = data_processor.create_anonymized_dataset(
    original_data=data,
    synthetic_quasi_identifiers=synthetic_features,
    conditions=conditions
)

# Save anonymized data
anonymized_data.to_csv("output/anonymized_data.csv", index=False)
```

### Example 2: Custom Configuration for Higher Privacy

```python
from src.data_processor import DataProcessor
from src.gan_model import ConditionalGAN

# Custom configuration with more quasi-identifiers for higher privacy
config_path = "config/high_privacy_config.json"

# Load and process data
data_processor = DataProcessor(config_path=config_path)
data = data_processor.load_data("data/healthcare_data.csv")
processed_data = data_processor.preprocess_data(data)

# Train CGAN model with settings for higher privacy
cgan = ConditionalGAN(noise_dim=150, hidden_dim=192)
dataloader, feature_dim, condition_dim = data_processor.prepare_data_for_cgan(processed_data)
cgan.train(dataloader, epochs=150, feature_dim=feature_dim, condition_dim=condition_dim)

# Generate synthetic quasi-identifiers
conditions = data_processor.get_conditions(processed_data)
synthetic_features = cgan.generate(conditions)

# Create anonymized dataset
anonymized_data = data_processor.create_anonymized_dataset(
    original_data=data,
    synthetic_quasi_identifiers=synthetic_features,
    conditions=conditions
)

# Save anonymized data
anonymized_data.to_csv("output/high_privacy_anonymized_data.csv", index=False)
```

### Example 3: Evaluating Anonymization Quality

```python
from src.evaluate_anonymization import AnonymizationEvaluator

# Initialize evaluator
evaluator = AnonymizationEvaluator(
    original_data_path="data/healthcare_data.csv",
    anonymized_data_path="output/anonymized_data.csv",
    config_path="config/default_config.json"
)

# Run comprehensive evaluation
evaluation_results = evaluator.evaluate()

# Print results
print(f"Clinical Data Preservation: {evaluation_results['clinical_data_preservation']['overall_score']:.4f}")
print(f"Data Utility: {evaluation_results['data_utility']['overall_utility_score']:.4f}")
print(f"Privacy Protection (k-anonymity): {evaluation_results['privacy_protection']['k_anonymity']}")
print(f"Information Preservation: {evaluation_results['information_preservation']['overall_information_preservation']:.4f}")

# Save detailed results
import json
with open("evaluation_results/comprehensive_report.json", "w") as f:
    json.dump(evaluation_results, f, indent=2)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Mode Collapse

**Symptoms**: The Generator produces limited variety of outputs, often the same or very similar samples.

**Solutions**:
- Increase the noise dimension (`noise_dim`)
- Add dropout to the Generator
- Implement mini-batch discrimination
- Reduce learning rate
- Use Wasserstein GAN with gradient penalty

#### 2. Training Instability

**Symptoms**: Loss values oscillate wildly or don't converge.

**Solutions**:
- Reduce learning rate
- Use Adam optimizer with beta1=0.5
- Implement gradient clipping
- Balance Discriminator and Generator training (e.g., train D more than G)

#### 3. Poor Quality Synthetic Data

**Symptoms**: Generated data doesn't maintain statistical properties of original data.

**Solutions**:
- Increase model capacity (larger hidden dimensions)
- Train for more epochs
- Ensure proper data preprocessing
- Add more conditions to guide generation
- Implement feature matching loss

#### 4. Memory Issues

**Symptoms**: Out of memory errors during training.

**Solutions**:
- Reduce batch size
- Simplify model architecture
- Use gradient accumulation
- Train on subset of data first

### Debugging Tips

1. **Visualize Losses**: Plot Generator and Discriminator losses to identify training issues
2. **Compare Distributions**: Compare distributions of original and synthetic data for each feature
3. **Incremental Development**: Start with a subset of features and gradually add more
4. **Hyperparameter Search**: Systematically test different hyperparameter combinations
5. **Validation**: Use a validation set to monitor overfitting