#!/usr/bin/env python3
"""
Run all experiments for the LLM-Guided Classical ML paper.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from mnist_experiments import run_mnist_comparison
from robot_experiments import run_robot_comparison
import json
from datetime import datetime

def run_all_experiments():
    """Run all experiments and save results."""
    
    print("Starting LLM-Guided Classical ML Experiments")
    print("=" * 50)
    
    results = {}
    
    # MNIST Experiments
    print("\n1. Running MNIST Classification Experiments...")
    try:
        mnist_results = run_mnist_comparison(
            n_samples=1000,
            test_samples=200,
            use_llm_teacher=True
        )
        results['mnist'] = mnist_results
        print("✓ MNIST experiments completed")
    except Exception as e:
        print(f"✗ MNIST experiments failed: {e}")
        results['mnist'] = {'error': str(e)}
    
    # Robot Control Experiments
    print("\n2. Running Robot Control Experiments...")
    try:
        robot_results = run_robot_comparison(
            n_episodes=10,
            steps_per_episode=100,
            use_llm_teacher=True
        )
        results['robot'] = robot_results
        print("✓ Robot control experiments completed")
    except Exception as e:
        print(f"✗ Robot control experiments failed: {e}")
        results['robot'] = {'error': str(e)}
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"../results/data/all_experiments_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ All results saved to {results_file}")
    
    # Generate summary
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    
    if 'mnist' in results and 'error' not in results['mnist']:
        print("\nMNIST Classification:")
        mnist = results['mnist']
        print(f"  Traditional ELM: {mnist.get('traditional_accuracy', 'N/A'):.2%}")
        print(f"  LLM-Guided ELM: {mnist.get('llm_guided_accuracy', 'N/A'):.2%}")
    
    if 'robot' in results and 'error' not in results['robot']:
        print("\nRobot Control:")
        robot = results['robot']
        print(f"  Traditional ELM: {robot.get('traditional_performance', 'N/A'):.3f}")
        print(f"  LLM-Guided ELM: {robot.get('llm_guided_performance', 'N/A'):.3f}")
    
    print("\n✓ All experiments completed!")
    return results

if __name__ == "__main__":
    run_all_experiments()
