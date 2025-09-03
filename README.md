# Healthcare Data Anonymization System

A comprehensive system for anonymizing healthcare data while preserving clinical utility. This project implements two complementary approaches: pseudonymization for direct identifiers and Conditional Generative Adversarial Networks (CGAN) for quasi-identifiers, while preserving original clinical data.

## Table of Contents

- [Overview](#overview)
- [Anonymization Approaches](#anonymization-approaches)
  - [Approach 1: Pseudonymization](#approach-1-pseudonymization)
  - [Approach 2: CGAN-based Anonymization](#approach-2-cgan-based-anonymization)
  - [Comparison of Approaches](#comparison-of-approaches)
- [Data Categories](#data-categories)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Privacy Considerations](#privacy-considerations)

## Overview

Healthcare data anonymization is critical for enabling research and analysis while protecting patient privacy. This system provides a balanced approach that:

1. **Protects patient privacy** by removing or transforming identifying information
2. **Preserves clinical utility** by maintaining the integrity of medical data
3. **Supports data analysis** by generating statistically similar synthetic data

## Anonymization Approaches

This project implements two complementary approaches to healthcare data anonymization:

### Approach 1: Pseudonymization

Pseudonymization replaces direct identifiers with artificial identifiers while maintaining referential integrity across the dataset.

**Implementation Details:**

- **ID Pseudonymization**: Patient IDs and other identifiers are replaced with UUIDs
- **Name Pseudonymization**: Names are replaced with generic identifiers (e.g., "Person_a1b2c3d4")
- **Address Pseudonymization**: Addresses are replaced with generic identifiers (e.g., "Address_a1b2c3d4")
- **SSN Pseudonymization**: SSNs are hashed and formatted to maintain the XXX-XX-XXXX pattern

**Justification:**
- Maintains referential integrity across the dataset
- Allows for authorized re-identification if needed (with secure mapping)
- Simple and efficient for direct identifiers
- Preserves the structure and format of the original data

### Approach 2: CGAN-based Anonymization

Conditional Generative Adversarial Networks (CGANs) are used to generate synthetic quasi-identifiers that preserve statistical properties while protecting individual privacy.

**Implementation Details:**

- **Conditional Generation**: Uses clinical data as conditions to generate appropriate quasi-identifiers
- **Statistical Preservation**: Maintains statistical relationships between variables
- **Age Range Transformation**: Converts exact birthdates to age ranges (e.g., "65-74", "75+")
- **Geographical Generalization**: Reduces precision of geographical data

**Justification:**
- Preserves statistical relationships between variables
- Generates realistic synthetic data that maintains utility for analysis
- Provides stronger privacy protection than simple generalization
- Allows for customization of the privacy-utility tradeoff

### Comparison of Approaches

| Feature | Pseudonymization | CGAN-based Anonymization |
|---------|-----------------|---------------------------|
| **Target Data** | Direct identifiers | Quasi-identifiers |
| **Privacy Level** | Moderate | High |
| **Data Utility** | High | Moderate |
| **Implementation Complexity** | Low | High |
| **Computational Cost** | Low | High |
| **Re-identification Risk** | Moderate (with mapping) | Low |
| **Best For** | Medical records, IDs, names | Demographics, dates, locations |

**Recommended Approach:**

Based on evaluation metrics, the combined approach (pseudonymization for direct identifiers + CGAN for quasi-identifiers) provides the best balance of privacy protection and data utility:

- **Clinical Data Preservation Score**: 0.83 (Good)
- **Data Utility Score**: 0.46 (Moderate)
- **Privacy Protection**: k-anonymity of 1 (needs improvement)
- **Information Preservation**: 0.06 (Low)

**Improvement Recommendations:**
1. Increase k-anonymity to at least 5 by adjusting CGAN parameters
2. Improve information preservation while maintaining privacy
3. Implement differential privacy techniques for sensitive fields

## Data Categories

The system categorizes healthcare data fields into three types:

### Direct Identifiers

Fields that directly identify an individual:

| Category | Examples | Anonymization Method |
|----------|----------|----------------------|
| **Patient IDs** | PATIENT, ID, ENCOUNTER | UUID replacement |
| **Names** | FIRST, LAST, NAME | Generic identifier |
| **Contact Information** | EMAIL, PHONE, FAX | Generic identifier |
| **Government IDs** | SSN, PASSPORT, LICENSE | Hashing with formatting |
| **Addresses** | ADDRESS, STREET, LOCATION | Generic identifier |

### Quasi-identifiers

Fields that could potentially identify an individual when combined:

| Category | Examples | Anonymization Method |
|----------|----------|----------------------|
| **Demographics** | GENDER, RACE, ETHNICITY | CGAN-generated values |
| **Dates** | BIRTHDATE, DEATHDATE | Age ranges + CGAN |
| **Geography** | ZIP, CITY, STATE, LAT, LON | CGAN-generated values |
| **Socioeconomic** | INCOME, EDUCATION, MARITAL | CGAN-generated values |

### Clinical Data

Fields containing medical information that are preserved in their original form:

| Category | Examples | Treatment |
|----------|----------|-----------|  
| **Diagnoses** | CODE, DESCRIPTION, REASONCODE | Preserved as-is |
| **Procedures** | PROCEDURE, REASONDESCRIPTION | Preserved as-is |
| **Medications** | MEDICATION, DOSAGE, STRENGTH | Preserved as-is |
| **Lab Results** | RESULTS, VALUES, OBSERVATIONS | Preserved as-is |

## Project Structure

```
healthcare_anonymization/
├── anonymize.py         # Main application script
├── config/              # Configuration files
│   └── default_config.json  # Default configuration for feature categories
├── data/                # Directory for input data
├── models/              # Directory for trained models
├── output/              # Directory for output files
├── evaluation_results/  # Evaluation metrics and visualizations
├── src/                 # Source code
│   ├── data_processor.py    # Data processing pipeline
│   ├── gan_model.py         # CGAN model implementation
│   ├── pseudonymization.py  # Pseudonymization for direct identifiers
│   └── utils/               # Utility functions
├── evaluate_anonymization.py  # Evaluation script
├── visualize_evaluation.py    # Visualization script
├── README.md                  # Main documentation
└── CGAN_README.md            # Detailed CGAN documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/healthcare_anonymization.git
cd healthcare_anonymization

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python anonymize.py --input /path/to/input.csv --output /path/to/output.csv
```

### Training a New CGAN Model

```bash
python anonymize.py --input /path/to/input.csv --train --epochs 200 --batch-size 128
```

### Using a Custom Configuration

```bash
python anonymize.py --input /path/to/input.csv --config /path/to/config.json
```

### Command Line Arguments

- `--input`: Path to input CSV file (required)
- `--output`: Path to output anonymized CSV file (default: output/anonymized_data.csv)
- `--mapping`: Path to save pseudonymization mapping (default: output/pseudonymization_mapping.json)
- `--records`: Number of records to process (default: all)
- `--config`: Path to configuration file
- `--train`: Train CGAN model (flag)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 64)
- `--noise-dim`: Dimension of noise vector (default: 100)
- `--model-dir`: Directory to save/load models (default: models)
- `--model-name`: Name of model to load (default: latest)

## Evaluation

The system includes comprehensive evaluation tools to assess the effectiveness of the anonymization process:

```bash
python evaluate_anonymization.py
```

This will generate evaluation metrics and visualizations in the `evaluation_results` directory.

### Evaluation Metrics

- **Data Structure Analysis**: Compares the structure of original and anonymized data
- **Clinical Data Preservation**: Measures how well clinical data is preserved
- **Data Utility**: Evaluates the statistical utility of anonymized data
- **Privacy Protection**: Assesses re-identification risk using k-anonymity
- **Information Loss**: Measures information loss using mutual information

## Privacy Considerations

- **Referential Integrity**: The system maintains referential integrity across the dataset
- **Secure Mapping**: Pseudonymization mappings are stored securely for authorized re-identification
- **Statistical Privacy**: The CGAN model is trained to preserve statistical properties while protecting individual privacy
- **Regulatory Compliance**: The system is designed to help meet HIPAA and GDPR requirements

For detailed information about the CGAN implementation, see [CGAN_README.md](CGAN_README.md).