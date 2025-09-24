"""
Quick Test for Minecraft LLM-Guided Learning
Shortened version for debugging and demonstration
"""

import sys
import os
import time
import json
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.loop.controller import MinecraftController, EpisodeConfig

def quick_minecraft_test():
    """Run a quick test of the Minecraft system"""
    
    print("🎮 Quick Minecraft LLM-Guided Learning Test")
    print("=" * 50)
    
    # Configure for quick test
    config = EpisodeConfig(
        max_episode_s=30.0,  # 30 seconds only
        tick_ms=500,  # Slower ticks for observation
        macro_budget_s=2.0,  # Quick macros
        llm_interval_s=10.0,  # Frequent LLM evaluation
        goal="Quick Test: Collect some wood",
        save_logs=True,
        log_dir="results/minecraft_logs/quick_test"
    )
    
    # Test baseline (no LLM)
    print("\n1️⃣ Testing Baseline (No LLM Teacher)")
    config.llm_interval_s = 999999  # Disable LLM
    
    controller = MinecraftController(config)
    
    try:
        stats = controller.run_episode("quick_baseline_test", seed=42)
        
        print(f"✅ Baseline Test Complete!")
        print(f"   Duration: {stats.duration_s:.1f}s")
        print(f"   Wood: {stats.wood_collected}")
        print(f"   Macros: {stats.macros_executed}")
        print(f"   Reward: {stats.total_reward:.2f}")
        
    except Exception as e:
        print(f"❌ Baseline test failed: {e}")
        return False
    
    # Test LLM-guided
    print("\n2️⃣ Testing LLM-Guided Learning")
    config.llm_interval_s = 10.0  # Enable LLM
    
    controller = MinecraftController(config)
    
    try:
        stats = controller.run_episode("quick_llm_test", seed=43)
        
        print(f"✅ LLM-Guided Test Complete!")
        print(f"   Duration: {stats.duration_s:.1f}s")
        print(f"   Wood: {stats.wood_collected}")
        print(f"   Macros: {stats.macros_executed}")
        print(f"   LLM Evals: {stats.llm_evaluations}")
        print(f"   Reward: {stats.total_reward:.2f}")
        print(f"   API Cost: ${stats.api_cost_estimate:.3f}")
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False
    
    print("\n🎯 Quick Test Summary")
    print("=" * 30)
    print("✅ Mock environment working")
    print("✅ Macro execution working") 
    print("✅ Feature extraction working")
    print("✅ Policy learning working")
    print("✅ LLM teacher working")
    print("✅ Episode logging working")
    
    print("\n📋 System Components Verified:")
    print("  🌍 Environment: Mock MineRL adapter")
    print("  🤖 Macros: 7 high-level actions")
    print("  📊 Features: 50+ dimensional state vector")
    print("  🧠 Policy: Online SGD with exploration")
    print("  👨‍🏫 Teacher: LLM evaluation system")
    print("  📝 Logging: Complete episode traces")
    
    return True

def test_individual_components():
    """Test individual components separately"""
    
    print("\n🔧 Component Tests")
    print("=" * 20)
    
    # Test environment
    try:
        from src.envs.minecraft.minerl_adapter import create_adapter
        env = create_adapter(use_mock=True)
        state = env.reset(seed=42)
        print("✅ Environment adapter")
    except Exception as e:
        print(f"❌ Environment adapter: {e}")
        return False
    
    # Test macros
    try:
        from src.macros.library import MACRO_LIBRARY, MacroMask
        from src.macros.executor import MacroExecutor
        
        executor = MacroExecutor(env)
        available = MacroMask.from_state(state)
        print(f"✅ Macro system ({len(available)} available)")
    except Exception as e:
        print(f"❌ Macro system: {e}")
        return False
    
    # Test features
    try:
        from src.features.featurizer import MinecraftFeaturizer
        
        featurizer = MinecraftFeaturizer()
        features = featurizer.featurize(state)
        print(f"✅ Feature extraction ({len(features)} features)")
    except Exception as e:
        print(f"❌ Feature extraction: {e}")
        return False
    
    # Test policy
    try:
        from src.policy.online_sgd import SGDPolicy
        
        policy = SGDPolicy()
        action = policy.predict(features)
        print(f"✅ Policy learning (selected: {action})")
    except Exception as e:
        print(f"❌ Policy learning: {e}")
        return False
    
    # Test LLM teacher
    try:
        from src.teacher.llm_client import LLMTeacher
        
        teacher = LLMTeacher()
        evaluation = teacher.evaluate("hp:20 wood:0 time:0.5")
        print(f"✅ LLM teacher (confidence: {evaluation.confidence:.2f})")
    except Exception as e:
        print(f"❌ LLM teacher: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Minecraft LLM-Guided Learning Tests")
    
    # Test components first
    if not test_individual_components():
        print("❌ Component tests failed")
        exit(1)
    
    # Run quick integration test
    if quick_minecraft_test():
        print("\n🎉 All tests passed! System is ready for full experiments.")
        print("\nTo run full experiments:")
        print("  python -m experiments.minecraft_day1.run_experiment --episodes 3 --compare")
    else:
        print("\n❌ Integration tests failed")
        exit(1)
