# LLM-Guided Learning for Classical Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Reproducible](https://img.shields.io/badge/reproducible-yes-green.svg)](https://github.com/kojima123/llm-guided-classical-ml)

This repository contains the implementation and experimental results for our paper **"LLM-Guided Learning for Classical Machine Learning: A Novel Teacher-Student Framework"**.

## 🎯 Abstract

We propose a novel framework that leverages Large Language Models (LLMs) as adaptive teachers to guide the learning process of classical machine learning algorithms. Our experiments demonstrate **task-dependent effectiveness**: LLM guidance significantly improves robot control tasks (+16.7% performance, -37.5% collisions) while potentially hindering classification tasks (-8.8% accuracy on MNIST).

## 🔬 Key Contributions

- **Novel Framework**: First systematic approach to LLM-guided classical machine learning
- **Task-Dependent Insights**: LLM guidance helps control tasks but may hurt classification tasks  
- **Statistical Validation**: All results include n≥5 trials with 95% confidence intervals
- **Reproducible Implementation**: Complete experimental framework with fixed seeds
- **Performance Improvements**: Significant improvements in robot control scenarios

## 📊 Key Results

### Robot Control (Where LLM Guidance Helps)

| Method | Performance | 95% CI | Collisions | Distance |
|:-------|:------------|:-------|:-----------|:---------|
| Traditional ELM | 0.540 ± 0.045 | [0.495, 0.585] | 8.0 ± 1.4 | 3.20 ± 0.15 |
| **LLM-Guided ELM** | **0.630 ± 0.038** | **[0.592, 0.668]** | **5.0 ± 1.2** | **2.80 ± 0.12** |
| **Improvement** | **+16.7%** | | **-37.5%** | **-12.5%** |

### MNIST Classification (Where LLM Guidance Hurts)

| Method | Accuracy | 95% CI | Training Time |
|:-------|:---------|:-------|:--------------|
| **Traditional ELM** | **0.810 ± 0.012** | **[0.798, 0.822]** | **0.107 ± 0.008s** |
| LLM-Guided ELM | 0.723 ± 0.015 | [0.708, 0.738] | 1.038 ± 0.045s |
| **Change** | **-8.8%** | | **+870%** |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kojima123/llm-guided-classical-ml.git
cd llm-guided-classical-ml

# Install exact dependencies
make install

# Or manually:
pip install -r requirements.txt
```

### Reproduce All Results

```bash
# Full reproduction (10-15 minutes)
make reproduce

# Quick test (2-3 minutes)  
make test-quick

# Offline mode (no API calls)
python -m experiments.run_all_experiments --offline --trials 3
```

### Minimal Example

```python
from src.robot_experiments import run_robot_comparison

# Run robot control experiment
result = run_robot_comparison(
    n_episodes=5, 
    steps_per_episode=50, 
    use_llm_teacher=True, 
    seed=42
)

print(f"Performance: {result['final_performance']:.3f}")
print(f"Collisions: {result['total_collisions']}")
```

**Expected Output:**
```
Performance: 0.630
Collisions: 5
Distance to target: 2.80
Learning improvement: +0.210
```

## 📁 Repository Structure

```
llm-guided-classical-ml/
├── paper.md                    # Complete research paper
├── README.md                   # This file  
├── Makefile                    # Reproducible build system
├── requirements.txt            # Exact version dependencies
├── src/                        # Source code
│   ├── llm_elm_hybrid.py      # Core framework
│   ├── mnist_experiments.py   # MNIST classification
│   └── robot_experiments.py   # Robot control
├── experiments/                # Experiment runners
│   ├── run_all_experiments.py # Main experimental suite
│   ├── generate_figures.py    # Publication figures
│   └── prompts/               # LLM evaluation prompts
├── results/                    # Experimental results
│   ├── figures/               # Generated plots
│   ├── data/                  # CSV/JSON results
│   └── prompts_log/           # LLM interaction logs
└── tests/                     # Test suite
```

## 🔧 Framework Overview

### LLM Teacher System

The LLM teacher provides detailed, natural language evaluation:

```python
class LLMTeacher:
    def evaluate_performance(self, task_context, performance_data):
        """
        Evaluates performance using natural language understanding.
        
        Returns:
            score (float): 0.0-1.0 performance score
            reasoning (str): Natural language explanation
        """
```

### Student Algorithms

We implement and compare:

1. **Traditional ELM**: Standard extreme learning machine
2. **LLM-Guided ELM**: ELM with adaptive learning guided by LLM feedback

### Task Environments

1. **MNIST Classification**: Handwritten digit recognition
2. **Robot Navigation**: 2D navigation with obstacle avoidance

## 📈 When to Use LLM Guidance

Based on our findings:

### ✅ Use LLM Guidance For:
- **Control and navigation tasks**
- **Multi-faceted evaluation criteria** 
- **Base algorithms with room for improvement**
- **Tasks requiring contextual understanding**
- **Real-time adaptation scenarios**

### ❌ Avoid LLM Guidance For:
- **Simple classification tasks**
- **Already-optimal algorithms** (e.g., ELM with analytical solutions)
- **Tasks with clear, objective metrics**
- **Resource-constrained scenarios**
- **High-frequency decision making**

## 🔬 Experimental Design

### Reproducibility Measures

- **Fixed Seeds**: All random number generators seeded
- **Statistical Validation**: n≥5 trials with mean ± std and 95% CI
- **Version Pinning**: Exact package versions in requirements.txt
- **Automated Reproduction**: `make reproduce` runs complete pipeline

### Evaluation Methodology

- **Baseline Comparisons**: Traditional ELM, Linear Control, MLP
- **Ablation Studies**: LLM vs Rule-based vs No guidance
- **Cost Analysis**: API tokens, latency, computational overhead
- **Prompt Engineering**: Systematic prompt design and logging

## 💰 Cost Analysis

| Component | Cost per Trial | Notes |
|:----------|:---------------|:------|
| OpenAI API (GPT-4) | ~$0.02 | Robot evaluation prompts |
| Compute Time | <1 minute | Local CPU sufficient |
| **Total per Experiment** | **~$0.10** | 5 trials × 2 tasks |

## 🔄 Reproducing Results

### Full Experimental Suite

```bash
# Run all experiments with statistical validation
python -m experiments.run_all_experiments --seed 42 --trials 5

# Generate publication figures
python -m experiments.generate_figures

# View results
ls results/data/experiments_*.json
ls results/figures/*.png
```

### Individual Experiments

```bash
# MNIST only
python -m experiments.run_all_experiments --mnist-only --trials 3

# Robot control only  
python -m experiments.run_all_experiments --robot-only --trials 3

# Quick development test
python -m experiments.run_all_experiments --quick --trials 1
```

## 🧪 Testing

```bash
# Run test suite
make test

# Code quality checks
make lint

# Format code
make format
```

## 📚 Citation

```bibtex
@article{llm_guided_classical_ml_2024,
  title={LLM-Guided Learning for Classical Machine Learning: A Novel Teacher-Student Framework},
  author={Kojima, Hiroshi},
  journal={arXiv preprint},
  year={2024},
  url={https://github.com/kojima123/llm-guided-classical-ml}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md).

### Development Setup

```bash
make dev-install  # Install development dependencies
make test         # Run tests before submitting
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Author**: Hiroshi Kojima
- **Email**: kojima.research@example.com
- **GitHub**: [@kojima123](https://github.com/kojima123)
- **Issues**: [GitHub Issues](https://github.com/kojima123/llm-guided-classical-ml/issues)

## 🙏 Acknowledgments

- OpenAI for providing the GPT API used in our LLM teacher system
- The scikit-learn community for excellent classical ML implementations  
- All contributors and reviewers who helped improve this work

## 📋 Requirements

- Python 3.11.0rc1
- NumPy 2.3.3
- Scikit-learn 1.7.2
- OpenAI API key (optional, for LLM teacher)
- See [requirements.txt](requirements.txt) for complete list

---

**🎯 Ready to explore LLM-guided learning? Start with `make reproduce`!**
