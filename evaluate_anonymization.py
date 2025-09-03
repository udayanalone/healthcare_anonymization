import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
import os

class AnonymizationEvaluator:
    def __init__(self, original_data_path, anonymized_data_path, config_path):
        self.original_data_path = original_data_path
        self.anonymized_data_path = anonymized_data_path
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Load datasets
        self.original_data = pd.read_csv(original_data_path)
        self.anonymized_data = pd.read_csv(anonymized_data_path)
        
        # Extract column categories
        self.direct_identifiers = self.config['direct_identifiers']
        self.quasi_identifiers = self.config['quasi_identifiers']
        self.clinical_data = self.config['clinical_data']
        
        # Create output directory if it doesn't exist
        os.makedirs('evaluation_results', exist_ok=True)
        
    def analyze_data_structure(self):
        """Analyze the structure of original and anonymized data"""
        original_columns = set(self.original_data.columns)
        anonymized_columns = set(self.anonymized_data.columns)
        
        # Check for missing columns
        missing_columns = original_columns - anonymized_columns
        extra_columns = anonymized_columns - original_columns
        
        # Check column types
        original_dtypes = self.original_data.dtypes
        anonymized_dtypes = self.anonymized_data.dtypes
        
        dtype_changes = {}
        for col in original_columns.intersection(anonymized_columns):
            if original_dtypes[col] != anonymized_dtypes[col]:
                dtype_changes[col] = (original_dtypes[col], anonymized_dtypes[col])
        
        # Generate report
        report = {
            'original_column_count': len(original_columns),
            'anonymized_column_count': len(anonymized_columns),
            'missing_columns': list(missing_columns),
            'extra_columns': list(extra_columns),
            'dtype_changes': {k: (str(v[0]), str(v[1])) for k, v in dtype_changes.items()}
        }
        
        return report
    
    def verify_clinical_data_preservation(self):
        """Verify that clinical data is preserved in the anonymized dataset"""
        # Filter for clinical data columns that exist in both datasets
        clinical_cols = [col for col in self.clinical_data if col in self.original_data.columns and col in self.anonymized_data.columns]
        
        # Calculate preservation metrics for each clinical column
        preservation_metrics = {}
        for col in clinical_cols:
            # Skip columns with object dtype that are likely to be identifiers
            if self.original_data[col].dtype == 'object' and ';' not in str(self.original_data[col].iloc[0]):
                # For categorical data, check value distributions
                orig_value_counts = self.original_data[col].value_counts(normalize=True)
                anon_value_counts = self.anonymized_data[col].value_counts(normalize=True)
                
                # Calculate Jensen-Shannon divergence or other similarity measure
                # For simplicity, we'll use a basic overlap coefficient
                common_values = set(orig_value_counts.index).intersection(set(anon_value_counts.index))
                if len(common_values) > 0:
                    preservation_score = sum(min(orig_value_counts.get(val, 0), anon_value_counts.get(val, 0)) for val in common_values)
                else:
                    preservation_score = 0.0
            else:
                # For numerical data or complex strings (like lists), check basic statistics
                try:
                    # Try to convert to numeric
                    orig_col = pd.to_numeric(self.original_data[col], errors='coerce')
                    anon_col = pd.to_numeric(self.anonymized_data[col], errors='coerce')
                    
                    # Calculate statistics
                    orig_stats = {'mean': orig_col.mean(), 'std': orig_col.std(), 'min': orig_col.min(), 'max': orig_col.max()}
                    anon_stats = {'mean': anon_col.mean(), 'std': anon_col.std(), 'min': anon_col.min(), 'max': anon_col.max()}
                    
                    # Calculate relative differences
                    rel_diff = {}
                    for stat in orig_stats:
                        if orig_stats[stat] != 0:
                            rel_diff[stat] = abs(orig_stats[stat] - anon_stats[stat]) / abs(orig_stats[stat])
                        else:
                            rel_diff[stat] = abs(orig_stats[stat] - anon_stats[stat]) if anon_stats[stat] != 0 else 0
                    
                    # Overall preservation score (lower is better)
                    # Ensure preservation score is not negative
                    preservation_score = max(0.0, 1.0 - np.mean(list(rel_diff.values())))
                except:
                    # For complex strings, check exact match percentage
                    exact_matches = sum(self.original_data[col] == self.anonymized_data[col])
                    preservation_score = exact_matches / len(self.original_data)
            
            preservation_metrics[col] = preservation_score
        
        # Calculate overall preservation score
        valid_scores = [score for score in preservation_metrics.values() if not np.isnan(score) and score >= 0.0]
        overall_score = np.mean(valid_scores) if valid_scores else 0.0
        
        # Generate report
        report = {
            'clinical_columns_analyzed': len(clinical_cols),
            'overall_preservation_score': overall_score,
            'column_preservation_scores': preservation_metrics
        }
        
        return report
    
    def evaluate_data_utility(self):
        """Evaluate data utility by comparing statistical distributions"""
        # Focus on quasi-identifiers for utility analysis
        quasi_cols = [col for col in self.quasi_identifiers if col in self.original_data.columns and col in self.anonymized_data.columns]
        
        utility_metrics = {}
        for col in quasi_cols:
            try:
                # For numerical columns
                if self.original_data[col].dtype in [np.int64, np.float64] or pd.to_numeric(self.original_data[col], errors='coerce').notna().sum() > 0.5 * len(self.original_data):
                    orig_col = pd.to_numeric(self.original_data[col], errors='coerce')
                    anon_col = pd.to_numeric(self.anonymized_data[col], errors='coerce')
                    
                    # Calculate KL divergence or other distribution similarity
                    # For simplicity, we'll use basic statistical comparison
                    orig_stats = {'mean': orig_col.mean(), 'std': orig_col.std(), 'median': orig_col.median()}
                    anon_stats = {'mean': anon_col.mean(), 'std': anon_col.std(), 'median': anon_col.median()}
                    
                    # Calculate relative differences
                    rel_diff = {}
                    for stat in orig_stats:
                        if not pd.isna(orig_stats[stat]) and orig_stats[stat] != 0:
                            rel_diff[stat] = abs(orig_stats[stat] - anon_stats[stat]) / abs(orig_stats[stat])
                        else:
                            rel_diff[stat] = 1.0 if not pd.isna(anon_stats[stat]) and anon_stats[stat] != 0 else 0.0
                    
                    # Overall utility score (higher is better)
                    utility_score = 1.0 - np.mean(list(rel_diff.values()))
                    utility_metrics[col] = {'type': 'numerical', 'score': utility_score, 'details': rel_diff}
                    
                    # Create distribution plots
                    plt.figure(figsize=(10, 6))
                    plt.subplot(1, 2, 1)
                    sns.histplot(orig_col.dropna(), kde=True, color='blue')
                    plt.title(f'Original {col} Distribution')
                    plt.subplot(1, 2, 2)
                    sns.histplot(anon_col.dropna(), kde=True, color='red')
                    plt.title(f'Anonymized {col} Distribution')
                    plt.tight_layout()
                    plt.savefig(f'evaluation_results/{col}_distribution.png')
                    plt.close()
                    
                else:  # For categorical columns
                    orig_counts = self.original_data[col].value_counts(normalize=True)
                    anon_counts = self.anonymized_data[col].value_counts(normalize=True)
                    
                    # Calculate Earth Mover's Distance or other categorical similarity
                    # For simplicity, we'll use a basic overlap coefficient
                    common_values = set(orig_counts.index).intersection(set(anon_counts.index))
                    if len(common_values) > 0:
                        overlap_score = sum(min(orig_counts.get(val, 0), anon_counts.get(val, 0)) for val in common_values)
                    else:
                        overlap_score = 0.0
                    
                    utility_metrics[col] = {'type': 'categorical', 'score': overlap_score}
                    
                    # Create bar plots for top categories
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    orig_counts.head(10).plot(kind='bar', color='blue')
                    plt.title(f'Original {col} Top Categories')
                    plt.subplot(1, 2, 2)
                    anon_counts.head(10).plot(kind='bar', color='red')
                    plt.title(f'Anonymized {col} Top Categories')
                    plt.tight_layout()
                    plt.savefig(f'evaluation_results/{col}_categories.png')
                    plt.close()
            except Exception as e:
                utility_metrics[col] = {'type': 'error', 'error': str(e)}
        
        # Calculate overall utility score
        valid_scores = [m['score'] for m in utility_metrics.values() if 'score' in m and not np.isnan(m['score'])]
        overall_score = np.mean(valid_scores) if valid_scores else 0.0
        
        # Generate report
        report = {
            'quasi_columns_analyzed': len(quasi_cols),
            'overall_utility_score': overall_score,
            'column_utility_scores': utility_metrics
        }
        
        return report
    
    def assess_privacy_protection(self):
        """Assess privacy protection using re-identification risk metrics"""
        # Focus on direct identifiers and quasi-identifiers
        direct_cols = [col for col in self.direct_identifiers if col in self.original_data.columns and col in self.anonymized_data.columns]
        quasi_cols = [col for col in self.quasi_identifiers if col in self.original_data.columns and col in self.anonymized_data.columns]
        
        # Check pseudonymization of direct identifiers
        pseudonymization_metrics = {}
        for col in direct_cols:
            # Calculate uniqueness in original and anonymized data
            orig_unique = self.original_data[col].nunique()
            anon_unique = self.anonymized_data[col].nunique()
            
            # Calculate overlap between original and anonymized values
            orig_values = set(self.original_data[col].dropna().unique())
            anon_values = set(self.anonymized_data[col].dropna().unique())
            overlap = len(orig_values.intersection(anon_values))
            
            # Calculate preservation of uniqueness
            if orig_unique > 0:
                uniqueness_preservation = anon_unique / orig_unique
            else:
                uniqueness_preservation = 1.0 if anon_unique == 0 else float('inf')
            
            pseudonymization_metrics[col] = {
                'original_unique_values': orig_unique,
                'anonymized_unique_values': anon_unique,
                'value_overlap': overlap,
                'uniqueness_preservation': uniqueness_preservation
            }
        
        # Calculate k-anonymity for quasi-identifiers
        # Group by all quasi-identifiers and count occurrences
        if quasi_cols:
            try:
                grouped = self.anonymized_data.groupby(quasi_cols).size().reset_index(name='count')
                k_anonymity = grouped['count'].min()
                k_anonymity_avg = grouped['count'].mean()
                k_anonymity_distribution = grouped['count'].value_counts().to_dict()
            except Exception as e:
                k_anonymity = 'Error: ' + str(e)
                k_anonymity_avg = 'Error'
                k_anonymity_distribution = {}
        else:
            k_anonymity = 'N/A - No quasi-identifiers found'
            k_anonymity_avg = 'N/A'
            k_anonymity_distribution = {}
        
        # Calculate l-diversity for sensitive attributes (using clinical data as sensitive)
        l_diversity_metrics = {}
        if quasi_cols and self.clinical_data:
            sensitive_cols = [col for col in self.clinical_data if col in self.anonymized_data.columns]
            for sensitive_col in sensitive_cols[:5]:  # Limit to first 5 sensitive columns for efficiency
                try:
                    # Group by quasi-identifiers and count distinct values of sensitive attribute
                    l_diversity_groups = self.anonymized_data.groupby(quasi_cols)[sensitive_col].nunique().reset_index(name='l_value')
                    l_diversity = l_diversity_groups['l_value'].min()
                    l_diversity_avg = l_diversity_groups['l_value'].mean()
                    l_diversity_metrics[sensitive_col] = {
                        'min_l_diversity': l_diversity,
                        'avg_l_diversity': l_diversity_avg
                    }
                except Exception as e:
                    l_diversity_metrics[sensitive_col] = {'error': str(e)}
        
        # Generate report
        report = {
            'direct_identifiers_analyzed': len(direct_cols),
            'quasi_identifiers_analyzed': len(quasi_cols),
            'pseudonymization_metrics': pseudonymization_metrics,
            'k_anonymity': k_anonymity,
            'k_anonymity_avg': k_anonymity_avg,
            'l_diversity_metrics': l_diversity_metrics
        }
        
        return report
    
    def test_information_loss(self):
        """Test information loss using standard metrics"""
        # Calculate information loss metrics for quasi-identifiers
        quasi_cols = [col for col in self.quasi_identifiers if col in self.original_data.columns and col in self.anonymized_data.columns]
        
        # Initialize metrics
        information_loss_metrics = {}
        
        # Calculate normalized variance for numerical attributes
        for col in quasi_cols:
            try:
                # Try to convert to numeric
                orig_col = pd.to_numeric(self.original_data[col], errors='coerce')
                anon_col = pd.to_numeric(self.anonymized_data[col], errors='coerce')
                
                # Skip if mostly non-numeric
                if orig_col.isna().sum() > 0.5 * len(orig_col) or anon_col.isna().sum() > 0.5 * len(anon_col):
                    continue
                
                # Calculate normalized variance
                orig_var = orig_col.var()
                if orig_var > 0:
                    anon_var = anon_col.var()
                    nv = 1.0 - min(anon_var / orig_var, orig_var / anon_var)
                else:
                    nv = 1.0 if anon_col.var() > 0 else 0.0
                
                # Calculate mutual information
                # Bin the data for mutual information calculation
                orig_binned = pd.qcut(orig_col.dropna(), q=10, labels=False, duplicates='drop')
                anon_binned = pd.qcut(anon_col.dropna(), q=10, labels=False, duplicates='drop')
                
                # Align the indices
                common_idx = orig_binned.index.intersection(anon_binned.index)
                if len(common_idx) > 0:
                    mi = mutual_info_score(orig_binned.loc[common_idx], anon_binned.loc[common_idx])
                    # Normalize by entropy
                    entropy = pd.Series(orig_binned.loc[common_idx]).value_counts(normalize=True)
                    entropy = -np.sum(entropy * np.log2(entropy))
                    nmi = mi / entropy if entropy > 0 else 0.0
                else:
                    mi = 0.0
                    nmi = 0.0
                
                information_loss_metrics[col] = {
                    'type': 'numerical',
                    'normalized_variance': nv,
                    'mutual_information': mi,
                    'normalized_mutual_information': nmi,
                    'information_preservation': nmi  # Higher is better
                }
            except Exception as e:
                # For categorical attributes
                try:
                    # Calculate categorical similarity
                    orig_counts = self.original_data[col].value_counts(normalize=True)
                    anon_counts = self.anonymized_data[col].value_counts(normalize=True)
                    
                    # Calculate Jensen-Shannon divergence
                    # For simplicity, we'll use a basic overlap coefficient
                    common_values = set(orig_counts.index).intersection(set(anon_counts.index))
                    if len(common_values) > 0:
                        overlap = sum(min(orig_counts.get(val, 0), anon_counts.get(val, 0)) for val in common_values)
                    else:
                        overlap = 0.0
                    
                    information_loss_metrics[col] = {
                        'type': 'categorical',
                        'distribution_overlap': overlap,
                        'information_preservation': overlap  # Higher is better
                    }
                except Exception as e2:
                    information_loss_metrics[col] = {'type': 'error', 'error': str(e2)}
        
        # Calculate overall information preservation score
        valid_scores = [m.get('information_preservation', 0) for m in information_loss_metrics.values() if 'information_preservation' in m]
        overall_score = np.mean(valid_scores) if valid_scores else 0.0
        
        # Generate report
        report = {
            'quasi_columns_analyzed': len(quasi_cols),
            'overall_information_preservation': overall_score,
            'column_information_metrics': information_loss_metrics
        }
        
        return report
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive evaluation report"""
        # Run all evaluations
        structure_report = self.analyze_data_structure()
        clinical_report = self.verify_clinical_data_preservation()
        utility_report = self.evaluate_data_utility()
        privacy_report = self.assess_privacy_protection()
        info_loss_report = self.test_information_loss()
        
        # Combine into comprehensive report
        comprehensive_report = {
            'data_structure_analysis': structure_report,
            'clinical_data_preservation': clinical_report,
            'data_utility_evaluation': utility_report,
            'privacy_protection_assessment': privacy_report,
            'information_loss_metrics': info_loss_report,
            'summary': {
                'clinical_data_preservation_score': clinical_report['overall_preservation_score'],
                'data_utility_score': utility_report['overall_utility_score'],
                'k_anonymity': privacy_report['k_anonymity'],
                'information_preservation_score': info_loss_report['overall_information_preservation'],
                'composite_metrics': {
                    'avg_kl_divergence': np.mean([c.get('kl_divergence', 0) for c in info_loss_report['column_information_metrics'].values()]),
                    'avg_wasserstein': np.mean([c.get('normalized_variance', 0) for c in info_loss_report['column_information_metrics'].values()]),
                    'correlation_metrics': calculate_correlations(evaluator.original_data, evaluator.anonymized_data, evaluator.quasi_identifiers)
                },
                'quality_score': max(0, 100 - (
                    np.mean([c.get('kl_divergence', 0) for c in info_loss_report['column_information_metrics'].values()]) * 10 +
                    np.mean([c.get('normalized_variance', 0) for c in info_loss_report['column_information_metrics'].values()]) * 5 +
                    calculate_correlations(evaluator.original_data, evaluator.anonymized_data, evaluator.quasi_identifiers)['correlation_mae'] * 20
                ))
            }
        }
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            else:
                return obj
        
        serializable_report = convert_to_serializable(comprehensive_report)
        
        # Save report to file
        with open('evaluation_results/comprehensive_report.json', 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        return comprehensive_report

# Main execution
def calculate_distribution_similarity(orig_series, anon_series):
    """Calculate KL divergence and Wasserstein distance between distributions"""
    if orig_series.nunique() > 20:
        # Numerical feature metrics
        return {
            'wasserstein': wasserstein_distance(orig_series, anon_series),
            'ks_stat': ks_2samp(orig_series, anon_series).statistic
        }
    else:
        # Categorical feature metrics
        orig_counts = orig_series.value_counts(normalize=True)
        anon_counts = anon_series.value_counts(normalize=True)
        kl_div = entropy(orig_counts, anon_counts)
        return {'kl_divergence': kl_div, 'js_distance': jensenshannon(orig_series, anon_series)}

def calculate_uniqueness(orig_series, anon_series):
    """Calculate uniqueness preservation metrics"""
    return {
        'original_unique': orig_series.nunique(),
        'anonymized_unique': anon_series.nunique(),
        'uniqueness_ratio': anon_series.nunique() / orig_series.nunique()
    }

def calculate_correlations(df_orig, df_anon, features):
    """Compare correlation matrices"""
    corr_orig = df_orig[features].corr().values
    corr_anon = df_anon[features].corr().values
    return {
        'correlation_mae': np.mean(np.abs(corr_orig - corr_anon)),
        'correlation_rmse': np.sqrt(np.mean((corr_orig - corr_anon)**2))
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate anonymization effectiveness')
    parser.add_argument('--original', default='data/unified_5000_records.csv', help='Path to original data')
    parser.add_argument('--anonymized', default='output/anonymized_data_final.csv', help='Path to anonymized data')
    parser.add_argument('--config', default='config/default_config.json', help='Path to configuration file')
    
    args = parser.parse_args()
    
    evaluator = AnonymizationEvaluator(args.original, args.anonymized, args.config)
    
    print("\n=== Analyzing Data Structure ===")
    structure_report = evaluator.analyze_data_structure()
    print(f"Original columns: {structure_report['original_column_count']}")
    print(f"Anonymized columns: {structure_report['anonymized_column_count']}")
    print(f"Missing columns: {len(structure_report['missing_columns'])}")
    print(f"Extra columns: {len(structure_report['extra_columns'])}")
    print(f"Data type changes: {len(structure_report['dtype_changes'])}")
    
    print("\n=== Verifying Clinical Data Preservation ===")
    clinical_report = evaluator.verify_clinical_data_preservation()
    print(f"Clinical columns analyzed: {clinical_report['clinical_columns_analyzed']}")
    print(f"Overall preservation score: {clinical_report['overall_preservation_score']:.4f}")
    
    print("\n=== Evaluating Data Utility ===")
    utility_report = evaluator.evaluate_data_utility()
    print(f"Quasi-identifier columns analyzed: {utility_report['quasi_columns_analyzed']}")
    print(f"Overall utility score: {utility_report['overall_utility_score']:.4f}")
    
    print("\n=== Assessing Privacy Protection ===")
    privacy_report = evaluator.assess_privacy_protection()
    print(f"Direct identifiers analyzed: {privacy_report['direct_identifiers_analyzed']}")
    print(f"Quasi-identifiers analyzed: {privacy_report['quasi_identifiers_analyzed']}")
    print(f"K-anonymity: {privacy_report['k_anonymity']}")
    
    print("\n=== Testing Information Loss ===")
    info_loss_report = evaluator.test_information_loss()
    print(f"Quasi-identifier columns analyzed: {info_loss_report['quasi_columns_analyzed']}")
    print(f"Overall information preservation: {info_loss_report['overall_information_preservation']:.4f}")
    
    print("\n=== Generating Comprehensive Report ===")
    comprehensive_report = evaluator.generate_comprehensive_report()
    print(f"Report saved to evaluation_results/comprehensive_report.json")
    
    print("\n=== Summary ===")
    print(f"Clinical Data Preservation Score: {comprehensive_report['summary']['clinical_data_preservation_score']:.4f}")
    print(f"Data Utility Score: {comprehensive_report['summary']['data_utility_score']:.4f}")
    print(f"K-anonymity: {comprehensive_report['summary']['k_anonymity']}")
    print(f"Information Preservation Score: {comprehensive_report['summary']['information_preservation_score']:.4f}")