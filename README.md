# LLM-Guided Learning for Classical Machine Learning

This repository contains the implementation and experimental results for our paper "LLM-Guided Learning for Classical Machine Learning: A Novel Teacher-Student Framework".

## Abstract

We propose a novel framework that leverages Large Language Models (LLMs) as adaptive teachers to guide the learning process of classical machine learning algorithms. Our experiments demonstrate up to 16.7% performance improvement in robot control tasks while revealing important task-dependent characteristics of LLM-guided learning.

## Key Contributions

- **Novel Framework**: First systematic approach to LLM-guided classical machine learning
- **Task-Dependent Insights**: LLM guidance helps control tasks but may hurt classification tasks
- **Practical Implementation**: Complete, reproducible experimental framework
- **Performance Improvements**: Significant improvements in robot control scenarios

## Repository Structure

```
├── paper.md                    # Main paper
├── README.md                   # This file
├── src/                        # Source code
│   ├── llm_elm_hybrid.py      # Main implementation
│   ├── mnist_experiments.py   # MNIST classification experiments
│   └── robot_experiments.py   # Robot control experiments
├── results/                    # Experimental results
│   ├── figures/               # Generated plots and figures
│   └── data/                  # Raw experimental data
├── requirements.txt           # Python dependencies
└── experiments/               # Experiment scripts
    ├── run_mnist.py
    └── run_robot.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-guided-classical-ml.git
cd llm-guided-classical-ml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up OpenAI API key (optional, for LLM teacher):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### MNIST Classification Experiment

```python
from src.mnist_experiments import run_mnist_comparison

# Run comparison between guided and unguided learning
results = run_mnist_comparison(
    n_samples=1000,
    test_samples=200,
    use_llm_teacher=True
)
```

### Robot Control Experiment

```python
from src.robot_experiments import run_robot_comparison

# Run robot navigation experiment
results = run_robot_comparison(
    n_episodes=10,
    steps_per_episode=100,
    use_llm_teacher=True
)
```

## Key Results

### Robot Control (Where LLM Guidance Helps)

| Method | Final Performance | Improvement | Collisions |
|:-------|:------------------|:------------|:-----------|
| Traditional ELM | 0.540 | +0.090 | 8 |
| **LLM-Guided ELM** | **0.630** | **+0.210** | **5** |

- **16.7% performance improvement**
- **2.3x better learning improvement**
- **37.5% reduction in collisions**

### MNIST Classification (Where LLM Guidance Hurts)

| Method | Test Accuracy | Training Time |
|:-------|:--------------|:--------------|
| **Traditional ELM** | **81.00%** | **0.107s** |
| LLM-Guided ELM | 72.25% | 1.038s |

- LLM guidance interferes with already-optimal analytical solution
- Demonstrates task-dependent nature of guidance effectiveness

## Framework Overview

### LLM Teacher System

The LLM teacher provides detailed, natural language evaluation:

```python
class LLMTeacher:
    def evaluate_performance(self, task_context, performance_data):
        # Constructs detailed evaluation prompt
        # Returns score (0.0-1.0) and natural language feedback
        # Enables adaptive learning rate adjustment
```

### Student Algorithms

We implement two classical ML variants:

1. **Traditional ELM**: Standard extreme learning machine
2. **Activation Reversed ELM**: Novel variant with activation function position reversal

### Task Environments

1. **MNIST Classification**: Standard digit recognition
2. **Robot Navigation**: 2D navigation with obstacle avoidance

## When to Use LLM Guidance

Based on our findings, LLM guidance is most effective when:

✅ **Use LLM Guidance For:**
- Control and navigation tasks
- Multi-faceted evaluation criteria
- Base algorithms with room for improvement
- Tasks requiring contextual understanding

❌ **Avoid LLM Guidance For:**
- Simple classification tasks
- Already-optimal algorithms
- Tasks with clear, objective metrics
- Resource-constrained scenarios

## Reproducing Results

### Full Experimental Suite

```bash
# Run all experiments
python experiments/run_all_experiments.py

# Generate paper figures
python experiments/generate_figures.py
```

### Individual Experiments

```bash
# MNIST comparison
python experiments/run_mnist.py --samples 1000 --llm-teacher

# Robot control comparison  
python experiments/run_robot.py --episodes 10 --llm-teacher

# Ablation studies
python experiments/run_ablations.py
```

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- Scikit-learn
- OpenAI API (optional, for LLM teacher)

See `requirements.txt` for complete list.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{llm_guided_classical_ml_2024,
  title={LLM-Guided Learning for Classical Machine Learning: A Novel Teacher-Student Framework},
  author={[Your Name]},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions or collaboration opportunities, please contact [your-email@example.com].

## Acknowledgments

- OpenAI for providing the GPT API used in our LLM teacher system
- The scikit-learn community for excellent classical ML implementations
- All contributors and reviewers who helped improve this work
