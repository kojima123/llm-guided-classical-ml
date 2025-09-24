"""
Minecraft Day1 Survival Experiment
Tests LLM-guided learning in Minecraft environment
"""

import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from dataclasses import asdict
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.loop.controller import MinecraftController, EpisodeConfig, EpisodeStats

def run_minecraft_experiment(n_episodes: int = 5,
                           use_llm_teacher: bool = True,
                           save_results: bool = True,
                           seed: int = 42) -> Dict[str, Any]:
    """Run Minecraft Day1 survival experiment"""
    
    print(f"🎮 Starting Minecraft Day1 Experiment")
    print(f"Episodes: {n_episodes}, LLM Teacher: {use_llm_teacher}, Seed: {seed}")
    
    # Set random seed
    np.random.seed(seed)
    
    # Configure episode
    config = EpisodeConfig(
        max_episode_s=180.0,  # 3 minutes per episode
        tick_ms=200,  # Slower for better observation
        macro_budget_s=5.0,
        llm_interval_s=30.0 if use_llm_teacher else 999999,  # Disable LLM if not using
        goal="Day1 Survival: Collect 10 wood, build shelter",
        save_logs=True,
        log_dir=f"results/minecraft_logs/{'llm' if use_llm_teacher else 'baseline'}"
    )
    
    # Run episodes
    controller = MinecraftController(config)
    episode_stats: List[EpisodeStats] = []
    
    for i in range(n_episodes):
        episode_id = f"{'llm' if use_llm_teacher else 'baseline'}_ep{i:02d}_{int(time.time())}"
        episode_seed = seed + i
        
        print(f"\n📊 Episode {i+1}/{n_episodes}: {episode_id}")
        
        try:
            stats = controller.run_episode(episode_id, episode_seed)
            episode_stats.append(stats)
            
            # Print episode summary
            print(f"  ✅ Duration: {stats.duration_s:.1f}s")
            print(f"  🪵 Wood: {stats.wood_collected}/10")
            print(f"  🏠 Shelter: {'✅' if stats.shelter_built else '❌'}")
            print(f"  🎯 Success: {'✅' if stats.success else '❌'}")
            print(f"  🤖 Macros: {stats.macros_executed}")
            print(f"  🧠 LLM Evals: {stats.llm_evaluations}")
            print(f"  💰 Reward: {stats.total_reward:.2f}")
            
        except Exception as e:
            print(f"  ❌ Episode failed: {e}")
            continue
    
    if not episode_stats:
        print("❌ No episodes completed successfully")
        return {}
    
    # Calculate aggregate statistics
    results = calculate_aggregate_stats(episode_stats, use_llm_teacher)
    
    # Print summary
    print_experiment_summary(results)
    
    # Save results
    if save_results:
        save_experiment_results(results, episode_stats, use_llm_teacher)
    
    return results

def calculate_aggregate_stats(episode_stats: List[EpisodeStats], 
                            use_llm_teacher: bool) -> Dict[str, Any]:
    """Calculate aggregate statistics from episode results"""
    
    if not episode_stats:
        return {}
    
    # Extract metrics
    durations = [s.duration_s for s in episode_stats]
    wood_collected = [s.wood_collected for s in episode_stats]
    success_rate = sum(s.success for s in episode_stats) / len(episode_stats)
    shelter_rate = sum(s.shelter_built for s in episode_stats) / len(episode_stats)
    total_rewards = [s.total_reward for s in episode_stats]
    macro_counts = [s.macros_executed for s in episode_stats]
    llm_eval_counts = [s.llm_evaluations for s in episode_stats]
    
    # Calculate statistics
    results = {
        'experiment_type': 'llm_guided' if use_llm_teacher else 'baseline',
        'n_episodes': len(episode_stats),
        'success_rate': success_rate,
        'shelter_rate': shelter_rate,
        
        # Duration stats
        'avg_duration': np.mean(durations),
        'std_duration': np.std(durations),
        'min_duration': np.min(durations),
        'max_duration': np.max(durations),
        
        # Wood collection stats
        'avg_wood': np.mean(wood_collected),
        'std_wood': np.std(wood_collected),
        'min_wood': np.min(wood_collected),
        'max_wood': np.max(wood_collected),
        
        # Reward stats
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'min_reward': np.min(total_rewards),
        'max_reward': np.max(total_rewards),
        
        # Action stats
        'avg_macros': np.mean(macro_counts),
        'std_macros': np.std(macro_counts),
        'avg_llm_evals': np.mean(llm_eval_counts),
        'std_llm_evals': np.std(llm_eval_counts),
        
        # Cost estimates
        'total_api_calls': sum(llm_eval_counts),
        'estimated_cost': sum(s.api_cost_estimate for s in episode_stats),
        
        # Raw data
        'episode_data': [asdict(s) for s in episode_stats]
    }
    
    return results

def print_experiment_summary(results: Dict[str, Any]):
    """Print experiment summary"""
    if not results:
        return
    
    print(f"\n🎯 Experiment Summary ({results['experiment_type']})")
    print("=" * 50)
    print(f"Episodes: {results['n_episodes']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Shelter Rate: {results['shelter_rate']:.1%}")
    print(f"Avg Duration: {results['avg_duration']:.1f}±{results['std_duration']:.1f}s")
    print(f"Avg Wood: {results['avg_wood']:.1f}±{results['std_wood']:.1f}")
    print(f"Avg Reward: {results['avg_reward']:.2f}±{results['std_reward']:.2f}")
    print(f"Avg Macros: {results['avg_macros']:.1f}±{results['std_macros']:.1f}")
    print(f"Avg LLM Evals: {results['avg_llm_evals']:.1f}±{results['std_llm_evals']:.1f}")
    print(f"API Calls: {results['total_api_calls']}")
    print(f"Est. Cost: ${results['estimated_cost']:.3f}")

def save_experiment_results(results: Dict[str, Any], 
                          episode_stats: List[EpisodeStats],
                          use_llm_teacher: bool):
    """Save experiment results to files"""
    
    # Create results directory
    results_dir = "results/minecraft_experiments"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save aggregate results
    exp_type = 'llm_guided' if use_llm_teacher else 'baseline'
    timestamp = int(time.time())
    
    results_file = os.path.join(results_dir, f"{exp_type}_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 Results saved to: {results_file}")
    
    # Create visualization
    create_experiment_plots(results, episode_stats, results_dir, exp_type, timestamp)

def create_experiment_plots(results: Dict[str, Any],
                          episode_stats: List[EpisodeStats],
                          results_dir: str,
                          exp_type: str,
                          timestamp: int):
    """Create visualization plots"""
    
    if not episode_stats:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Minecraft Day1 Experiment - {exp_type.replace("_", " ").title()}', fontsize=16)
    
    # Episode success over time
    episodes = list(range(1, len(episode_stats) + 1))
    successes = [1 if s.success else 0 for s in episode_stats]
    
    axes[0, 0].plot(episodes, successes, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_title('Success Rate Over Episodes')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Success (1=Yes, 0=No)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.1, 1.1)
    
    # Wood collection
    wood_counts = [s.wood_collected for s in episode_stats]
    axes[0, 1].bar(episodes, wood_counts, alpha=0.7, color='brown')
    axes[0, 1].axhline(y=10, color='red', linestyle='--', label='Goal (10 wood)')
    axes[0, 1].set_title('Wood Collection per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Wood Collected')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reward progression
    rewards = [s.total_reward for s in episode_stats]
    axes[1, 0].plot(episodes, rewards, 'o-', color='green', linewidth=2, markersize=8)
    axes[1, 0].set_title('Total Reward per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Total Reward')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Macro efficiency (reward per macro)
    efficiency = [s.total_reward / max(s.macros_executed, 1) for s in episode_stats]
    axes[1, 1].plot(episodes, efficiency, 'o-', color='purple', linewidth=2, markersize=8)
    axes[1, 1].set_title('Efficiency (Reward per Macro)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Reward / Macro')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(results_dir, f"{exp_type}_plots_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plots saved to: {plot_file}")

def compare_experiments(baseline_results: Dict[str, Any],
                       llm_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare baseline vs LLM-guided experiments"""
    
    if not baseline_results or not llm_results:
        print("❌ Cannot compare: missing results")
        return {}
    
    comparison = {
        'baseline': baseline_results,
        'llm_guided': llm_results,
        'improvements': {
            'success_rate': llm_results['success_rate'] - baseline_results['success_rate'],
            'avg_wood': llm_results['avg_wood'] - baseline_results['avg_wood'],
            'avg_reward': llm_results['avg_reward'] - baseline_results['avg_reward'],
            'efficiency': (llm_results['avg_reward'] / max(llm_results['avg_macros'], 1)) - 
                         (baseline_results['avg_reward'] / max(baseline_results['avg_macros'], 1))
        }
    }
    
    print(f"\n🔄 Comparison Results")
    print("=" * 30)
    print(f"Success Rate: {comparison['improvements']['success_rate']:+.1%}")
    print(f"Wood Collection: {comparison['improvements']['avg_wood']:+.1f}")
    print(f"Total Reward: {comparison['improvements']['avg_reward']:+.2f}")
    print(f"Efficiency: {comparison['improvements']['efficiency']:+.3f}")
    
    return comparison

def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description='Run Minecraft Day1 Survival Experiment')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--no-llm', action='store_true', help='Run baseline without LLM teacher')
    parser.add_argument('--compare', action='store_true', help='Run both baseline and LLM experiments')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if args.compare:
        print("🔄 Running comparison experiment...")
        
        # Run baseline
        print("\n1️⃣ Running baseline experiment...")
        baseline_results = run_minecraft_experiment(
            n_episodes=args.episodes,
            use_llm_teacher=False,
            seed=args.seed
        )
        
        # Run LLM-guided
        print("\n2️⃣ Running LLM-guided experiment...")
        llm_results = run_minecraft_experiment(
            n_episodes=args.episodes,
            use_llm_teacher=True,
            seed=args.seed + 1000  # Different seed for fair comparison
        )
        
        # Compare results
        comparison = compare_experiments(baseline_results, llm_results)
        
        # Save comparison
        if comparison:
            results_dir = "results/minecraft_experiments"
            os.makedirs(results_dir, exist_ok=True)
            comparison_file = os.path.join(results_dir, f"comparison_{int(time.time())}.json")
            with open(comparison_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"📁 Comparison saved to: {comparison_file}")
    
    else:
        # Run single experiment
        use_llm = not args.no_llm
        run_minecraft_experiment(
            n_episodes=args.episodes,
            use_llm_teacher=use_llm,
            seed=args.seed
        )

if __name__ == "__main__":
    main()
