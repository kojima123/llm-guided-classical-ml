#!/usr/bin/env python3
"""
Generate all figures for the LLM-Guided Classical ML paper.
Reads experimental data and creates publication-ready plots.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_latest_results():
    """Load the most recent experimental results."""
    results_dir = Path("results/data")
    
    # Find the latest experiment file
    experiment_files = list(results_dir.glob("experiments_*.json"))
    if not experiment_files:
        raise FileNotFoundError("No experiment results found. Run experiments first.")
    
    latest_file = max(experiment_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded results from: {latest_file}")
    return results

def create_comparison_barplot(data, title, ylabel, filename, show_ci=True):
    """Create a comparison bar plot with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(data.keys())
    means = [data[method]['mean'] for method in methods]
    stds = [data[method]['std'] for method in methods]
    
    bars = ax.bar(methods, means, yerr=stds if show_ci else None, 
                  capsize=5, alpha=0.8, color=['#1f77b4', '#ff7f0e'])
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std/2,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Improve method labels
    ax.set_xticklabels([method.replace('_', ' ').title() for method in methods])
    
    plt.tight_layout()
    plt.savefig(f"results/figures/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated: results/figures/{filename}")

def create_learning_curves(results, task_type):
    """Create learning curves if trial data is available."""
    if task_type not in results or not results[task_type]:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance over trials
    for i, (method, data) in enumerate(results[task_type].items()):
        if method == 'metadata' or 'stats' not in data:
            continue
            
        if task_type == 'mnist':
            values = data.get('accuracies', [])
            ylabel = 'Accuracy'
        else:
            values = data.get('performances', [])
            ylabel = 'Performance Score'
        
        if values:
            trials = range(1, len(values) + 1)
            axes[0].plot(trials, values, 'o-', label=method.replace('_', ' ').title(), 
                        linewidth=2, markersize=6)
    
    axes[0].set_title(f'{task_type.upper()} - Performance Over Trials', fontweight='bold')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Distribution comparison
    all_values = []
    all_methods = []
    
    for method, data in results[task_type].items():
        if method == 'metadata' or 'stats' not in data:
            continue
            
        if task_type == 'mnist':
            values = data.get('accuracies', [])
        else:
            values = data.get('performances', [])
        
        if values:
            all_values.extend(values)
            all_methods.extend([method.replace('_', ' ').title()] * len(values))
    
    if all_values:
        df = pd.DataFrame({'Method': all_methods, 'Value': all_values})
        sns.boxplot(data=df, x='Method', y='Value', ax=axes[1])
        axes[1].set_title(f'{task_type.upper()} - Distribution Comparison', fontweight='bold')
        axes[1].set_ylabel(ylabel)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"results/figures/{task_type}_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated: results/figures/{task_type}_analysis.png")

def create_improvement_analysis(results):
    """Create improvement analysis figure."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    improvements = {}
    
    # Calculate improvements for each task
    for task in ['mnist', 'robot']:
        if (task in results and 
            'traditional_elm' in results[task] and 
            'llm_guided_elm' in results[task] and
            'stats' in results[task]['traditional_elm'] and
            'stats' in results[task]['llm_guided_elm']):
            
            if task == 'mnist':
                baseline = results[task]['traditional_elm']['stats']['accuracy_mean']
                guided = results[task]['llm_guided_elm']['stats']['accuracy_mean']
                metric = 'Accuracy'
            else:
                baseline = results[task]['traditional_elm']['stats']['performance_mean']
                guided = results[task]['llm_guided_elm']['stats']['performance_mean']
                metric = 'Performance'
            
            absolute_improvement = guided - baseline
            relative_improvement = (absolute_improvement / baseline) * 100
            
            improvements[task] = {
                'absolute': absolute_improvement,
                'relative': relative_improvement,
                'baseline': baseline,
                'guided': guided,
                'metric': metric
            }
    
    # Absolute improvement plot
    if improvements:
        tasks = list(improvements.keys())
        abs_improvements = [improvements[task]['absolute'] for task in tasks]
        colors = ['red' if x < 0 else 'green' for x in abs_improvements]
        
        bars = axes[0].bar(tasks, abs_improvements, color=colors, alpha=0.7)
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0].set_title('Absolute Improvement with LLM Guidance', fontweight='bold')
        axes[0].set_ylabel('Improvement')
        axes[0].set_xlabel('Task')
        
        # Add value labels
        for bar, improvement in zip(bars, abs_improvements):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., 
                        height + (0.01 if height >= 0 else -0.01),
                        f'{improvement:+.3f}',
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontweight='bold')
        
        # Relative improvement plot
        rel_improvements = [improvements[task]['relative'] for task in tasks]
        colors = ['red' if x < 0 else 'green' for x in rel_improvements]
        
        bars = axes[1].bar(tasks, rel_improvements, color=colors, alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_title('Relative Improvement with LLM Guidance', fontweight='bold')
        axes[1].set_ylabel('Improvement (%)')
        axes[1].set_xlabel('Task')
        
        # Add value labels
        for bar, improvement in zip(bars, rel_improvements):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., 
                        height + (1 if height >= 0 else -1),
                        f'{improvement:+.1f}%',
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("results/figures/improvement_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: results/figures/improvement_analysis.png")

def create_summary_table_figure(results):
    """Create a summary table as a figure."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Task', 'Method', 'Mean ± Std', '95% CI', 'Improvement']
    
    for task in ['mnist', 'robot']:
        if task not in results:
            continue
            
        task_name = task.upper()
        baseline_stats = None
        
        for method in ['traditional_elm', 'llm_guided_elm']:
            if method not in results[task] or 'stats' not in results[task][method]:
                continue
                
            stats = results[task][method]['stats']
            method_name = method.replace('_', ' ').title()
            
            if task == 'mnist':
                mean = stats['accuracy_mean']
                std = stats['accuracy_std']
                ci = stats['accuracy_ci95']
                metric = 'Accuracy'
            else:
                mean = stats['performance_mean']
                std = stats['performance_std']
                ci = stats['performance_ci95']
                metric = 'Performance'
            
            mean_std = f"{mean:.3f} ± {std:.3f}"
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            
            if method == 'traditional_elm':
                baseline_stats = mean
                improvement = "Baseline"
            else:
                if baseline_stats is not None:
                    abs_imp = mean - baseline_stats
                    rel_imp = (abs_imp / baseline_stats) * 100
                    improvement = f"{abs_imp:+.3f} ({rel_imp:+.1f}%)"
                else:
                    improvement = "N/A"
            
            table_data.append([task_name, method_name, mean_std, ci_str, improvement])
            task_name = ""  # Only show task name for first row
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color improvement cells
    for i, row in enumerate(table_data, 1):
        if 'Baseline' in row[4]:
            table[(i, 4)].set_facecolor('#E3F2FD')
        elif '+' in row[4]:
            table[(i, 4)].set_facecolor('#C8E6C9')
        elif '-' in row[4]:
            table[(i, 4)].set_facecolor('#FFCDD2')
    
    plt.title('Experimental Results Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig("results/figures/results_summary_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: results/figures/results_summary_table.png")

def generate_all_figures(results):
    """Generate all figures from experimental results."""
    print("Generating publication-ready figures...")
    
    # Create figures directory
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    
    # MNIST comparison
    if 'mnist' in results and results['mnist']:
        mnist_data = {}
        for method in ['traditional_elm', 'llm_guided_elm']:
            if method in results['mnist'] and 'stats' in results['mnist'][method]:
                stats = results['mnist'][method]['stats']
                mnist_data[method] = {
                    'mean': stats['accuracy_mean'],
                    'std': stats['accuracy_std']
                }
        
        if mnist_data:
            create_comparison_barplot(
                mnist_data, 
                'MNIST Classification Accuracy Comparison',
                'Accuracy',
                'mnist_comparison.png'
            )
    
    # Robot comparison
    if 'robot' in results and results['robot']:
        robot_data = {}
        for method in ['traditional_elm', 'llm_guided_elm']:
            if method in results['robot'] and 'stats' in results['robot'][method]:
                stats = results['robot'][method]['stats']
                robot_data[method] = {
                    'mean': stats['performance_mean'],
                    'std': stats['performance_std']
                }
        
        if robot_data:
            create_comparison_barplot(
                robot_data,
                'Robot Control Performance Comparison',
                'Performance Score',
                'robot_comparison.png'
            )
    
    # Learning curves and distributions
    create_learning_curves(results, 'mnist')
    create_learning_curves(results, 'robot')
    
    # Improvement analysis
    create_improvement_analysis(results)
    
    # Summary table
    create_summary_table_figure(results)
    
    print("\n✅ All figures generated successfully!")
    print("Figures saved in results/figures/")

def main():
    parser = argparse.ArgumentParser(description="Generate figures for LLM-Guided Classical ML")
    parser.add_argument("--results-file", type=str, help="Specific results file to use")
    
    args = parser.parse_args()
    
    try:
        if args.results_file:
            with open(args.results_file, 'r') as f:
                results = json.load(f)
        else:
            results = load_latest_results()
        
        generate_all_figures(results)
        
    except Exception as e:
        print(f"Error generating figures: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
