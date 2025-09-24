"""
Main Controller for Minecraft LLM-Guided Learning
Orchestrates the interaction between environment, policy, and teacher
"""

import time
import json
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import os

from ..envs.minecraft.minerl_adapter import create_adapter, MinecraftState
from ..macros.library import MacroMask, get_macro_by_name
from ..macros.executor import MacroExecutor
from ..features.featurizer import MinecraftFeaturizer, HistoryBuffer, FeatureConfig
from ..policy.online_sgd import SGDPolicy
from ..teacher.llm_client import LLMTeacher, RubricWeights

@dataclass
class EpisodeConfig:
    """Configuration for episode execution"""
    max_episode_s: float = 300.0  # 5 minutes max
    tick_ms: int = 100  # 100ms per tick
    macro_budget_s: float = 3.0  # Max time per macro
    llm_interval_s: float = 45.0  # LLM evaluation interval
    goal: str = "Day1 Survival: Collect 10 wood, build 2x2 shelter"
    save_logs: bool = True
    log_dir: str = "results/minecraft_logs"

@dataclass
class EpisodeStats:
    """Statistics from episode execution"""
    episode_id: str
    duration_s: float
    total_steps: int
    macros_executed: int
    llm_evaluations: int
    final_inventory: Dict[str, int]
    final_hp: float
    final_hunger: float
    success: bool
    wood_collected: int
    shelter_built: bool
    survival_time: float
    total_reward: float
    api_calls: int
    api_cost_estimate: float

class MinecraftController:
    """Main controller for Minecraft LLM-guided learning"""
    
    def __init__(self, config: EpisodeConfig = None):
        self.config = config or EpisodeConfig()
        
        # Initialize components
        self.env = create_adapter("MineRLTreechop-v0", use_mock=True)
        self.executor = MacroExecutor(self.env)
        self.featurizer = MinecraftFeaturizer(FeatureConfig())
        self.policy = SGDPolicy()
        self.teacher = LLMTeacher()
        self.rubric = RubricWeights()
        self.history = HistoryBuffer()
        
        # Episode state
        self.current_state: Optional[MinecraftState] = None
        self.episode_start_time: float = 0
        self.last_llm_eval_time: float = 0
        self.step_count: int = 0
        self.macro_count: int = 0
        self.llm_eval_count: int = 0
        
        # Logging
        self.episode_log: List[Dict[str, Any]] = []
        
    def run_episode(self, episode_id: str = None, seed: int = None) -> EpisodeStats:
        """Run a complete episode"""
        if episode_id is None:
            episode_id = f"episode_{int(time.time())}"
        
        print(f"Starting episode: {episode_id}")
        
        # Reset environment and state
        self.current_state = self.env.reset(seed=seed)
        self.episode_start_time = time.time()
        self.last_llm_eval_time = self.episode_start_time
        self.step_count = 0
        self.macro_count = 0
        self.llm_eval_count = 0
        self.episode_log = []
        self.history = HistoryBuffer()
        
        # Initial state logging
        self._log_event("episode_start", {
            "episode_id": episode_id,
            "goal": self.config.goal,
            "initial_state": self._state_to_dict(self.current_state)
        })
        
        # Main episode loop
        done = False
        while not done and self._get_episode_time() < self.config.max_episode_s:
            
            # Get current features
            features = self.featurizer.featurize(
                self.current_state, 
                self.history.get_recent_actions(5)
            )
            
            # Determine available macros
            available_macros = MacroMask.from_state(self.current_state)
            available_macro_names = [macro.value for macro in available_macros]
            
            # Select macro using policy
            selected_macro_name = self.policy.predict(
                features, 
                available_macros=available_macro_names
            )
            
            # Get macro specification
            macro_spec = get_macro_by_name(selected_macro_name)
            if macro_spec is None:
                print(f"Warning: Unknown macro {selected_macro_name}, using idle_scan")
                macro_spec = get_macro_by_name("idle_scan")
            
            # Execute macro
            macro_report = self.executor.execute(macro_spec, self.config.macro_budget_s)
            self.macro_count += 1
            
            # Update featurizer action history
            self.featurizer.update_action_history(selected_macro_name)
            
            # Calculate reward (simple heuristic for now)
            reward = self._calculate_reward(macro_report)
            
            # Add to history
            self.history.push(
                self.current_state,
                selected_macro_name,
                reward,
                time.time()
            )
            
            # Log macro execution
            self._log_event("macro_execution", {
                "macro": selected_macro_name,
                "success": macro_report.success,
                "duration": macro_report.duration_s,
                "events": macro_report.events,
                "delta_inv": macro_report.delta_inv,
                "reward": reward
            })
            
            # Check if LLM evaluation is needed
            if self._should_evaluate_with_llm():
                self._perform_llm_evaluation(features, selected_macro_name, reward)
            
            # Update current state (simplified - in real implementation would get from env)
            self._update_state_from_report(macro_report)
            
            # Check termination conditions
            done = self._check_termination()
            self.step_count += 1
            
            # Small delay to simulate real-time
            time.sleep(self.config.tick_ms / 1000.0)
        
        # Generate episode statistics
        stats = self._generate_episode_stats(episode_id)
        
        # Save logs if requested
        if self.config.save_logs:
            self._save_episode_log(episode_id)
        
        print(f"Episode {episode_id} completed: {stats.duration_s:.1f}s, "
              f"{stats.macros_executed} macros, success={stats.success}")
        
        return stats
    
    def _should_evaluate_with_llm(self) -> bool:
        """Determine if LLM evaluation should be triggered"""
        current_time = time.time()
        time_since_last = current_time - self.last_llm_eval_time
        
        # Time-based trigger
        if time_since_last >= self.config.llm_interval_s:
            return True
        
        # Event-based triggers
        recent_events = []
        for record in self.episode_log[-5:]:  # Last 5 events
            if record["type"] == "macro_execution":
                recent_events.extend(record["data"]["events"])
        
        # Trigger on important events
        important_events = ["took_damage", "crafted_table", "shelter_complete", "night_approaching"]
        if any(event in recent_events for event in important_events):
            return True
        
        return False
    
    def _perform_llm_evaluation(self, features: np.ndarray, last_action: str, reward: float):
        """Perform LLM evaluation and update policy"""
        
        # Create state summary for LLM
        summary = self._create_state_summary()
        
        # Get LLM evaluation
        evaluation = self.teacher.evaluate(summary, self.config.goal)
        self.llm_eval_count += 1
        self.last_llm_eval_time = time.time()
        
        # Log LLM evaluation
        self._log_event("llm_evaluation", {
            "summary": summary,
            "evaluation": {
                "progress": evaluation.progress,
                "safety": evaluation.safety,
                "efficiency": evaluation.efficiency,
                "suggested_policy": evaluation.suggested_policy,
                "confidence": evaluation.confidence,
                "comment": evaluation.comment
            }
        })
        
        # Update policy based on LLM suggestion
        if evaluation.suggested_policy != last_action:
            # Calculate sample weight
            sample_weight = self.rubric.compute_sample_weight(evaluation)
            
            # Update policy with LLM suggestion
            self.policy.partial_fit(
                features,
                evaluation.suggested_policy,
                sample_weight=sample_weight
            )
            
            self._log_event("policy_update", {
                "suggested_action": evaluation.suggested_policy,
                "actual_action": last_action,
                "sample_weight": sample_weight,
                "confidence": evaluation.confidence
            })
    
    def _create_state_summary(self) -> str:
        """Create natural language summary of current state"""
        if not self.current_state:
            return "No state available"
        
        state = self.current_state
        recent_actions = self.history.get_recent_actions(5)
        
        # Build summary
        summary_parts = []
        
        # Basic stats
        summary_parts.append(f"hp:{state.hp:.0f}/20")
        summary_parts.append(f"hunger:{state.hunger:.0f}/20")
        summary_parts.append(f"time_of_day:{state.time_of_day:.2f}")
        
        # Inventory
        inv_parts = []
        for item, count in state.inventory.items():
            if count > 0:
                inv_parts.append(f"{item}:{count}")
        if inv_parts:
            summary_parts.append("inventory:" + ",".join(inv_parts))
        else:
            summary_parts.append("inventory:empty")
        
        # Recent actions
        if recent_actions:
            summary_parts.append(f"recent_actions:{','.join(recent_actions[-3:])}")
        
        # Environmental info
        if state.hostile_count > 0:
            summary_parts.append(f"hostiles:{state.hostile_count}")
        
        summary_parts.append(f"light:{state.light_level:.0f}")
        
        return " ".join(summary_parts)
    
    def _calculate_reward(self, macro_report) -> float:
        """Calculate reward for macro execution"""
        reward = 0.0
        
        # Success bonus
        if macro_report.success:
            reward += 1.0
        else:
            reward -= 0.5
        
        # Inventory improvements
        for item, delta in macro_report.delta_inv.items():
            if item == "log" and delta > 0:
                reward += delta * 0.5  # Wood is valuable
            elif item == "planks" and delta > 0:
                reward += delta * 0.3
            elif item == "crafting_table" and delta > 0:
                reward += 2.0  # Crafting table is important
        
        # Event bonuses
        for event in macro_report.events:
            if "crafted" in event:
                reward += 1.0
            elif "built" in event:
                reward += 2.0
            elif "damage" in event:
                reward -= 1.0
        
        return reward
    
    def _update_state_from_report(self, macro_report):
        """Update current state based on macro execution report"""
        if macro_report.final_state:
            self.current_state = macro_report.final_state
        else:
            # Simple state update based on inventory changes
            for item, delta in macro_report.delta_inv.items():
                if item in self.current_state.inventory:
                    self.current_state.inventory[item] += delta
                else:
                    self.current_state.inventory[item] = delta
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Time limit
        if self._get_episode_time() >= self.config.max_episode_s:
            return True
        
        # Success condition: 10 wood + shelter
        wood_count = self.current_state.inventory.get("log", 0)
        has_shelter = self.current_state.inventory.get("crafting_table", 0) > 0
        
        if wood_count >= 10 and has_shelter:
            return True
        
        # Failure condition: death
        if self.current_state.hp <= 0:
            return True
        
        return False
    
    def _get_episode_time(self) -> float:
        """Get elapsed episode time"""
        return time.time() - self.episode_start_time
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event"""
        self.episode_log.append({
            "timestamp": time.time(),
            "episode_time": self._get_episode_time(),
            "type": event_type,
            "data": data
        })
    
    def _state_to_dict(self, state: MinecraftState) -> Dict[str, Any]:
        """Convert state to dictionary for logging"""
        return {
            "hp": state.hp,
            "hunger": state.hunger,
            "time_of_day": state.time_of_day,
            "position": state.position.tolist(),
            "inventory": state.inventory,
            "hostile_count": state.hostile_count,
            "light_level": state.light_level
        }
    
    def _generate_episode_stats(self, episode_id: str) -> EpisodeStats:
        """Generate episode statistics"""
        duration = self._get_episode_time()
        
        # Calculate success metrics
        wood_collected = self.current_state.inventory.get("log", 0)
        shelter_built = self.current_state.inventory.get("crafting_table", 0) > 0
        success = wood_collected >= 10 and shelter_built
        
        # Calculate total reward
        total_reward = sum(
            record["data"]["reward"] 
            for record in self.episode_log 
            if record["type"] == "macro_execution" and "reward" in record["data"]
        )
        
        return EpisodeStats(
            episode_id=episode_id,
            duration_s=duration,
            total_steps=self.step_count,
            macros_executed=self.macro_count,
            llm_evaluations=self.llm_eval_count,
            final_inventory=self.current_state.inventory.copy(),
            final_hp=self.current_state.hp,
            final_hunger=self.current_state.hunger,
            success=success,
            wood_collected=wood_collected,
            shelter_built=shelter_built,
            survival_time=duration,
            total_reward=total_reward,
            api_calls=self.llm_eval_count,
            api_cost_estimate=self.llm_eval_count * 0.002  # Rough estimate
        )
    
    def _save_episode_log(self, episode_id: str):
        """Save episode log to file"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        log_file = os.path.join(self.config.log_dir, f"{episode_id}.jsonl")
        with open(log_file, 'w') as f:
            for record in self.episode_log:
                f.write(json.dumps(record) + '\n')
        
        print(f"Episode log saved to: {log_file}")

def run_episode(env_name: str = "MineRLTreechop-v0",
                config: EpisodeConfig = None,
                episode_id: str = None,
                seed: int = None) -> EpisodeStats:
    """Convenience function to run a single episode"""
    controller = MinecraftController(config)
    return controller.run_episode(episode_id, seed)
