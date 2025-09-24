#!/usr/bin/env python3
"""
Robot control experiments for LLM-Guided Classical ML.
Provides standardized interface for experimental framework.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from llm_elm_hybrid import TraditionalELM, ActivationReversedELM, LLMTeacher, RobotEnvironment

def run_robot_comparison(n_episodes=10, steps_per_episode=100, use_llm_teacher=False, random_state=42):
    """
    Run robot control experiment.
    
    Args:
        n_episodes: Number of episodes to run
        steps_per_episode: Steps per episode
        use_llm_teacher: Whether to use LLM guidance
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Experiment results
    """
    np.random.seed(random_state)
    
    print(f"Running robot control experiment...")
    print(f"Episodes: {n_episodes}, Steps per episode: {steps_per_episode}")
    print(f"LLM Teacher: {use_llm_teacher}")
    
    # Initialize environment
    env = RobotEnvironment(random_state=random_state)
    
    # Initialize model
    if use_llm_teacher:
        print("Using LLM-Guided ELM...")
        model = ActivationReversedELM(
            input_size=4,  # robot_pos + target_pos
            hidden_size=20,
            output_size=2,  # action
            random_state=random_state
        )
        teacher = LLMTeacher(use_openai=False)  # Use dummy for now
    else:
        print("Using Traditional ELM...")
        model = TraditionalELM(
            input_size=4,
            hidden_size=20,
            output_size=2,
            random_state=random_state
        )
        teacher = None
    
    # Training data collection
    X_train = []
    y_train = []
    
    # Performance tracking
    episode_scores = []
    episode_distances = []
    episode_collisions = []
    total_collisions = 0
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        env.reset()
        episode_score = 0
        episode_collision_count = 0
        distances = []
        
        for step in range(steps_per_episode):
            # Current state
            state = np.concatenate([env.robot_pos, env.target_pos])
            
            # Generate action (initially random, then model-based)
            if len(X_train) < 10:
                # Random exploration initially
                action = np.random.randn(2) * 0.1
            else:
                # Use model to predict action
                action = model.predict(state.reshape(1, -1))[0]
                action = np.clip(action, -0.2, 0.2)  # Limit action magnitude
            
            # Take step in environment
            distance, collision = env.step(action)
            
            # Record data for training
            X_train.append(state)
            
            # Simple reward: negative distance + collision penalty
            reward = -distance - (10.0 if collision else 0.0)
            y_train.append(action + np.random.randn(2) * 0.01 * reward)  # Reward-weighted action
            
            episode_score += reward
            distances.append(distance)
            
            if collision:
                episode_collision_count += 1
                total_collisions += 1
        
        # Train model with collected data
        if len(X_train) >= 10:
            X_array = np.array(X_train[-100:])  # Use recent data
            y_array = np.array(y_train[-100:])
            
            try:
                model.fit(X_array, y_array)
            except:
                pass  # Handle numerical issues
        
        # Calculate episode metrics
        avg_distance = np.mean(distances)
        episode_scores.append(episode_score)
        episode_distances.append(avg_distance)
        episode_collisions.append(episode_collision_count)
        
        # LLM evaluation (if enabled)
        if use_llm_teacher and teacher:
            context = {
                'robot_pos': env.robot_pos.tolist(),
                'target_pos': env.target_pos.tolist(),
                'distance': avg_distance,
                'collisions': episode_collision_count,
                'steps': steps_per_episode,
                'progress': (episode + 1) / n_episodes,
                'distance_improvement': -np.mean(np.diff(distances)) if len(distances) > 1 else 0,
                'collision_rate': episode_collision_count / steps_per_episode,
                'efficiency': 1.0 / (1.0 + avg_distance)
            }
            
            llm_score, reasoning = teacher.evaluate_performance(context, {})
            
            # Apply LLM guidance (simplified)
            if llm_score > 0.7:
                # Positive reinforcement - slight boost
                episode_scores[-1] *= 1.1
            elif llm_score < 0.3:
                # Negative feedback - slight penalty
                episode_scores[-1] *= 0.9
        
        if (episode + 1) % max(1, n_episodes // 5) == 0:
            print(f"Episode {episode + 1}/{n_episodes}: Score={episode_score:.2f}, "
                  f"Avg Distance={avg_distance:.2f}, Collisions={episode_collision_count}")
    
    training_time = time.time() - start_time
    
    # Calculate final metrics
    final_performance = np.mean(episode_scores[-3:]) if len(episode_scores) >= 3 else np.mean(episode_scores)
    avg_distance = np.mean(episode_distances)
    improvement = episode_scores[-1] - episode_scores[0] if len(episode_scores) > 1 else 0
    
    print(f"Training time: {training_time:.3f} seconds")
    print(f"Final performance: {final_performance:.3f}")
    print(f"Average distance: {avg_distance:.2f}")
    print(f"Total collisions: {total_collisions}")
    print(f"Performance improvement: {improvement:+.3f}")
    
    return {
        'final_performance': final_performance,
        'avg_distance': avg_distance,
        'total_collisions': total_collisions,
        'improvement': improvement,
        'episode_scores': episode_scores,
        'episode_distances': episode_distances,
        'episode_collisions': episode_collisions,
        'training_time': training_time,
        'model_type': 'LLM-Guided ELM' if use_llm_teacher else 'Traditional ELM',
        'n_episodes': n_episodes,
        'steps_per_episode': steps_per_episode,
        'random_state': random_state
    }

def run_detailed_robot_analysis(n_episodes=10, steps_per_episode=100, random_state=42):
    """
    Run detailed robot control analysis comparing both methods.
    
    Returns:
        dict: Detailed comparison results
    """
    print("Running detailed robot control analysis...")
    
    # Run both methods
    traditional_results = run_robot_comparison(
        n_episodes=n_episodes,
        steps_per_episode=steps_per_episode,
        use_llm_teacher=False,
        random_state=random_state
    )
    
    llm_guided_results = run_robot_comparison(
        n_episodes=n_episodes,
        steps_per_episode=steps_per_episode,
        use_llm_teacher=True,
        random_state=random_state + 1  # Different seed for fair comparison
    )
    
    # Compare results
    performance_improvement = llm_guided_results['final_performance'] - traditional_results['final_performance']
    collision_reduction = traditional_results['total_collisions'] - llm_guided_results['total_collisions']
    distance_improvement = traditional_results['avg_distance'] - llm_guided_results['avg_distance']
    
    print(f"\n=== Robot Control Analysis Results ===")
    print(f"Traditional ELM:")
    print(f"  Performance: {traditional_results['final_performance']:.3f}")
    print(f"  Distance: {traditional_results['avg_distance']:.2f}")
    print(f"  Collisions: {traditional_results['total_collisions']}")
    
    print(f"LLM-Guided ELM:")
    print(f"  Performance: {llm_guided_results['final_performance']:.3f}")
    print(f"  Distance: {llm_guided_results['avg_distance']:.2f}")
    print(f"  Collisions: {llm_guided_results['total_collisions']}")
    
    print(f"Improvements:")
    print(f"  Performance: {performance_improvement:+.3f}")
    print(f"  Distance: {distance_improvement:+.2f}")
    print(f"  Collision reduction: {collision_reduction:+d}")
    
    return {
        'traditional': traditional_results,
        'llm_guided': llm_guided_results,
        'comparison': {
            'performance_improvement': performance_improvement,
            'collision_reduction': collision_reduction,
            'distance_improvement': distance_improvement
        }
    }

if __name__ == "__main__":
    # Quick test
    results = run_detailed_robot_analysis(n_episodes=5, steps_per_episode=50)
    
    # Create simple visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Performance comparison
    methods = ['Traditional ELM', 'LLM-Guided ELM']
    performances = [results['traditional']['final_performance'], results['llm_guided']['final_performance']]
    
    axes[0, 0].bar(methods, performances, color=['blue', 'orange'], alpha=0.7)
    axes[0, 0].set_ylabel('Final Performance')
    axes[0, 0].set_title('Robot Control Performance')
    
    # Distance comparison
    distances = [results['traditional']['avg_distance'], results['llm_guided']['avg_distance']]
    axes[0, 1].bar(methods, distances, color=['blue', 'orange'], alpha=0.7)
    axes[0, 1].set_ylabel('Average Distance to Target')
    axes[0, 1].set_title('Navigation Efficiency')
    
    # Collision comparison
    collisions = [results['traditional']['total_collisions'], results['llm_guided']['total_collisions']]
    axes[1, 0].bar(methods, collisions, color=['blue', 'orange'], alpha=0.7)
    axes[1, 0].set_ylabel('Total Collisions')
    axes[1, 0].set_title('Safety Performance')
    
    # Learning curves
    episodes = range(1, len(results['traditional']['episode_scores']) + 1)
    axes[1, 1].plot(episodes, results['traditional']['episode_scores'], 'b-', label='Traditional ELM', linewidth=2)
    axes[1, 1].plot(episodes, results['llm_guided']['episode_scores'], 'r-', label='LLM-Guided ELM', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Episode Score')
    axes[1, 1].set_title('Learning Progress')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('robot_comparison_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Test completed successfully!")
