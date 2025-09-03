import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.gridspec import GridSpec

def load_evaluation_results(file_path='evaluation_results/comprehensive_report.json'):
    """Load evaluation results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_output_directory(directory='evaluation_results/visualizations'):
    """Create directory for visualizations if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)
    return directory

def plot_summary_metrics(results, output_dir):
    """Create a summary dashboard of key metrics"""
    summary = results['summary']
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Anonymization Evaluation Summary', fontsize=16)
    
    # Create grid for subplots
    gs = GridSpec(2, 2, figure=fig)
    
    # Clinical Data Preservation
    ax1 = fig.add_subplot(gs[0, 0])
    clinical_score = summary['clinical_data_preservation_score']
    ax1.bar(['Clinical Data Preservation'], [clinical_score], color='green')
    ax1.set_ylim(0, 1)
    ax1.set_title('Clinical Data Preservation')
    ax1.text(0, clinical_score/2, f'{clinical_score:.2f}', 
             ha='center', va='center', color='white', fontweight='bold')
    
    # Data Utility
    ax2 = fig.add_subplot(gs[0, 1])
    utility_score = summary['data_utility_score']
    ax2.bar(['Data Utility'], [utility_score], color='blue')
    ax2.set_ylim(0, 1)
    ax2.set_title('Data Utility Score')
    ax2.text(0, utility_score/2, f'{utility_score:.2f}', 
             ha='center', va='center', color='white', fontweight='bold')
    
    # K-anonymity
    ax3 = fig.add_subplot(gs[1, 0])
    k_anonymity = summary['k_anonymity']
    ax3.bar(['K-anonymity'], [k_anonymity], color='red')
    ax3.set_title('K-anonymity')
    ax3.text(0, k_anonymity/2, f'{k_anonymity}', 
             ha='center', va='center', color='white', fontweight='bold')
    
    # Information Preservation
    ax4 = fig.add_subplot(gs[1, 1])
    info_score = summary['information_preservation_score']
    ax4.bar(['Information Preservation'], [info_score], color='purple')
    ax4.set_ylim(0, 1)
    ax4.set_title('Information Preservation Score')
    ax4.text(0, info_score/2, f'{info_score:.2f}', 
             ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{output_dir}/summary_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_clinical_data_preservation(results, output_dir):
    """Plot clinical data preservation metrics"""
    # Check if the expected structure exists
    if 'clinical_data_preservation' not in results or 'preservation_metrics' not in results['clinical_data_preservation']:
        print("Warning: Clinical data preservation metrics not found in expected format")
        # Create a simple plot with the overall score
        plt.figure(figsize=(8, 6))
        score = results['summary']['clinical_data_preservation_score']
        plt.bar(['Clinical Data Preservation'], [score], color='green')
        plt.title('Clinical Data Preservation Score')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/clinical_preservation.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    clinical_data = results['clinical_data_preservation']
    metrics = clinical_data['preservation_metrics']
    
    # Get top 10 and bottom 10 preserved columns
    scores = [(col, score) for col, score in metrics.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    top_10 = scores[:min(10, len(scores))]
    bottom_10 = scores[-min(10, len(scores)):]
    
    # Plot top 10
    plt.figure(figsize=(12, 6))
    cols, vals = zip(*top_10) if top_10 else ([], [])
    if cols and vals:
        plt.bar(cols, vals, color='green')
        plt.title('Top 10 Best Preserved Clinical Data Columns')
        plt.ylabel('Preservation Score')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/top10_clinical_preservation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot bottom 10
    plt.figure(figsize=(12, 6))
    cols, vals = zip(*bottom_10) if bottom_10 else ([], [])
    if cols and vals:
        plt.bar(cols, vals, color='red')
        plt.title('Bottom 10 Least Preserved Clinical Data Columns')
        plt.ylabel('Preservation Score')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/bottom10_clinical_preservation.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_data_utility(results, output_dir):
    """Plot data utility metrics"""
    # Check if the expected structure exists
    if 'data_utility' not in results or 'utility_metrics' not in results['data_utility']:
        print("Warning: Data utility metrics not found in expected format")
        # Create a simple plot with the overall score
        plt.figure(figsize=(8, 6))
        score = results['summary']['data_utility_score']
        plt.bar(['Data Utility'], [score], color='blue')
        plt.title('Data Utility Score')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/data_utility.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    utility_data = results['data_utility']
    metrics = utility_data['utility_metrics']
    
    # Extract scores for each column
    scores = []
    for col, data in metrics.items():
        if 'score' in data and not np.isnan(data['score']):
            scores.append((col, data['score']))
    
    if not scores:
        print("Warning: No valid data utility scores found")
        return
    
    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Plot all columns
    plt.figure(figsize=(14, 8))
    cols, vals = zip(*scores)
    plt.bar(cols, vals, color='blue')
    plt.title('Data Utility Scores by Column')
    plt.ylabel('Utility Score')
    plt.xticks(rotation=90)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/data_utility_by_column.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_information_preservation(results, output_dir):
    """Plot information preservation metrics"""
    # Check if the expected structure exists
    if 'information_loss' not in results or 'column_metrics' not in results['information_loss']:
        print("Warning: Information preservation metrics not found in expected format")
        # Create a simple plot with the overall score
        plt.figure(figsize=(8, 6))
        score = results['summary']['information_preservation_score']
        plt.bar(['Information Preservation'], [score], color='purple')
        plt.title('Information Preservation Score')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/information_preservation.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    info_data = results['information_loss']
    metrics = info_data['column_metrics']
    
    # Extract information preservation for each column
    scores = []
    for col, data in metrics.items():
        if 'information_preservation' in data and not np.isnan(data['information_preservation']):
            scores.append((col, data['information_preservation']))
    
    if not scores:
        print("Warning: No valid information preservation scores found")
        return
    
    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Plot top 10
    top_10 = scores[:min(10, len(scores))]
    if not top_10:
        return
        
    plt.figure(figsize=(12, 6))
    cols, vals = zip(*top_10)
    plt.bar(cols, vals, color='purple')
    plt.title('Top 10 Columns with Highest Information Preservation')
    plt.ylabel('Information Preservation Score')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(vals) * 1.1 if vals else 1.0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top10_information_preservation.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_evaluation_report(results, output_dir):
    """Create a comprehensive evaluation report in HTML format"""
    summary = results['summary']
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Healthcare Data Anonymization Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; }}
            .metric {{ margin-bottom: 10px; }}
            .metric-name {{ font-weight: bold; }}
            .score {{ font-size: 1.2em; }}
            .good {{ color: green; }}
            .moderate {{ color: orange; }}
            .poor {{ color: red; }}
            .section {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .dashboard {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
            .dashboard-item {{ width: 48%; margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .dashboard-item h3 {{ margin-top: 0; color: #2c3e50; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Healthcare Data Anonymization Evaluation Report</h1>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="dashboard">
                <div class="dashboard-item">
                    <h3>Clinical Data Preservation</h3>
                    <div class="metric">
                        <span class="metric-name">Score:</span>
                        <span class="score {{'good' if summary['clinical_data_preservation_score'] > 0.8 else 'moderate' if summary['clinical_data_preservation_score'] > 0.5 else 'poor'}}">
                            {summary['clinical_data_preservation_score']:.4f}
                        </span>
                    </div>
                    <p>This score indicates how well the clinical data has been preserved during anonymization. A higher score means better preservation.</p>
                </div>
                
                <div class="dashboard-item">
                    <h3>Data Utility</h3>
                    <div class="metric">
                        <span class="metric-name">Score:</span>
                        <span class="score {{'good' if summary['data_utility_score'] > 0.8 else 'moderate' if summary['data_utility_score'] > 0.5 else 'poor'}}">
                            {summary['data_utility_score']:.4f}
                        </span>
                    </div>
                    <p>This score measures how useful the anonymized data remains for analysis. Higher scores indicate better utility preservation.</p>
                </div>
                
                <div class="dashboard-item">
                    <h3>Privacy Protection</h3>
                    <div class="metric">
                        <span class="metric-name">K-anonymity:</span>
                        <span class="score {{'good' if summary['k_anonymity'] >= 5 else 'moderate' if summary['k_anonymity'] >= 2 else 'poor'}}">
                            {summary['k_anonymity']}
                        </span>
                    </div>
                    <p>K-anonymity measures privacy protection. A value of k means each record is indistinguishable from at least k-1 other records. Higher values provide better privacy.</p>
                </div>
                
                <div class="dashboard-item">
                    <h3>Information Preservation</h3>
                    <div class="metric">
                        <span class="metric-name">Score:</span>
                        <span class="score {{'good' if summary['information_preservation_score'] > 0.8 else 'moderate' if summary['information_preservation_score'] > 0.5 else 'poor'}}">
                            {summary['information_preservation_score']:.4f}
                        </span>
                    </div>
                    <p>This score indicates how much of the original information content is preserved in the anonymized data. Higher scores mean better information preservation.</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Visualizations</h2>
            <div class="dashboard">
                <div class="dashboard-item">
                    <h3>Summary Metrics</h3>
                    <img src="summary_metrics.png" alt="Summary Metrics">
                </div>
                
                <div class="dashboard-item">
                    <h3>Top 10 Best Preserved Clinical Data Columns</h3>
                    <img src="top10_clinical_preservation.png" alt="Top 10 Clinical Preservation">
                </div>
                
                <div class="dashboard-item">
                    <h3>Data Utility by Column</h3>
                    <img src="data_utility_by_column.png" alt="Data Utility by Column">
                </div>
                
                <div class="dashboard-item">
                    <h3>Top 10 Columns with Highest Information Preservation</h3>
                    <img src="top10_information_preservation.png" alt="Top 10 Information Preservation">
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Conclusion and Recommendations</h2>
            <p>
                Based on the evaluation results, the anonymization process has achieved:
                <ul>
                    <li><strong>Clinical Data Preservation:</strong> {"Good" if summary['clinical_data_preservation_score'] > 0.8 else "Moderate" if summary['clinical_data_preservation_score'] > 0.5 else "Poor"} preservation of clinical data with a score of {summary['clinical_data_preservation_score']:.4f}.</li>
                    <li><strong>Data Utility:</strong> {"Good" if summary['data_utility_score'] > 0.8 else "Moderate" if summary['data_utility_score'] > 0.5 else "Poor"} utility preservation with a score of {summary['data_utility_score']:.4f}.</li>
                    <li><strong>Privacy Protection:</strong> {"Good" if summary['k_anonymity'] >= 5 else "Moderate" if summary['k_anonymity'] >= 2 else "Poor"} privacy protection with a k-anonymity of {summary['k_anonymity']}.</li>
                    <li><strong>Information Preservation:</strong> {"Good" if summary['information_preservation_score'] > 0.8 else "Moderate" if summary['information_preservation_score'] > 0.5 else "Poor"} information preservation with a score of {summary['information_preservation_score']:.4f}.</li>
                </ul>
            </p>
            
            <p><strong>Recommendations:</strong></p>
            <ul>
                <li>{"Continue with the current anonymization approach as it provides good balance between utility and privacy." if summary['clinical_data_preservation_score'] > 0.7 and summary['data_utility_score'] > 0.4 else "Consider adjusting the anonymization parameters to improve clinical data preservation."}</li>
                <li>{"The k-anonymity level is sufficient for basic privacy protection." if summary['k_anonymity'] >= 5 else "Increase the k-anonymity level to at least 5 for better privacy protection."}</li>
                <li>{"The information preservation is adequate for most analytical purposes." if summary['information_preservation_score'] > 0.5 else "Improve information preservation by refining the anonymization techniques."}</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(f'{output_dir}/evaluation_report.html', 'w') as f:
        f.write(html_content)

def main():
    # Load evaluation results
    results = load_evaluation_results()
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Create visualizations
    plot_summary_metrics(results, output_dir)
    plot_clinical_data_preservation(results, output_dir)
    plot_data_utility(results, output_dir)
    plot_information_preservation(results, output_dir)
    
    # Create HTML report
    create_evaluation_report(results, output_dir)
    
    print(f"Visualizations and report saved to {output_dir}")

if __name__ == "__main__":
    main()