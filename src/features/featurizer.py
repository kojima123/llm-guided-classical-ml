"""
Feature Extractor for Minecraft State
Converts game state into numerical features for ML models
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

from ..envs.minecraft.minerl_adapter import MinecraftState
from ..macros.library import MacroType

@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    normalize_position: bool = True
    include_inventory: bool = True
    include_time: bool = True
    include_health: bool = True
    include_recent_actions: bool = True
    recent_action_window: int = 5
    position_scale: float = 100.0  # Scale for position normalization

class MinecraftFeaturizer:
    """Extracts numerical features from Minecraft state"""
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.action_history: List[str] = []
        self.feature_names: List[str] = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Build list of feature names for debugging"""
        self.feature_names = []
        
        if self.config.include_health:
            self.feature_names.extend(["hp_normalized", "hunger_normalized"])
        
        if self.config.include_time:
            self.feature_names.extend(["time_of_day", "is_day", "is_night"])
        
        if self.config.normalize_position:
            self.feature_names.extend(["pos_x_norm", "pos_y_norm", "pos_z_norm"])
        
        self.feature_names.extend(["yaw_sin", "yaw_cos", "pitch_norm"])
        
        if self.config.include_inventory:
            items = ["log", "planks", "stick", "crafting_table", "wooden_pickaxe"]
            for item in items:
                self.feature_names.extend([f"has_{item}", f"count_{item}_norm"])
        
        self.feature_names.extend(["hostile_count_norm", "light_level_norm"])
        
        if self.config.include_recent_actions:
            for i in range(self.config.recent_action_window):
                for action_type in MacroType:
                    self.feature_names.append(f"recent_{action_type.value}_{i}")
    
    def featurize(self, state: MinecraftState, recent_actions: List[str] = None) -> np.ndarray:
        """Convert state to feature vector"""
        features = []
        
        # Health and hunger
        if self.config.include_health:
            features.extend([
                state.hp / 20.0,  # Normalize to 0-1
                state.hunger / 20.0
            ])
        
        # Time features
        if self.config.include_time:
            features.append(state.time_of_day)
            features.append(1.0 if 0.2 < state.time_of_day < 0.8 else 0.0)  # is_day
            features.append(1.0 if state.time_of_day > 0.8 or state.time_of_day < 0.2 else 0.0)  # is_night
        
        # Position features
        if self.config.normalize_position:
            features.extend([
                state.position[0] / self.config.position_scale,
                state.position[1] / self.config.position_scale,
                state.position[2] / self.config.position_scale
            ])
        
        # Orientation features (using sin/cos for cyclical nature)
        features.extend([
            np.sin(np.radians(state.yaw)),
            np.cos(np.radians(state.yaw)),
            state.pitch / 90.0  # Normalize pitch to -1 to 1
        ])
        
        # Inventory features
        if self.config.include_inventory:
            important_items = ["log", "planks", "stick", "crafting_table", "wooden_pickaxe"]
            for item in important_items:
                count = state.inventory.get(item, 0)
                features.extend([
                    1.0 if count > 0 else 0.0,  # has_item
                    min(count / 64.0, 1.0)  # normalized count (max stack = 64)
                ])
        
        # Environmental features
        features.extend([
            min(state.hostile_count / 5.0, 1.0),  # Normalize hostile count
            state.light_level / 15.0  # Normalize light level
        ])
        
        # Recent actions features
        if self.config.include_recent_actions:
            if recent_actions is None:
                recent_actions = self.action_history[-self.config.recent_action_window:]
            
            # Pad with empty actions if needed
            while len(recent_actions) < self.config.recent_action_window:
                recent_actions = [""] + recent_actions
            
            # One-hot encode recent actions
            for i in range(self.config.recent_action_window):
                action = recent_actions[i] if i < len(recent_actions) else ""
                for action_type in MacroType:
                    features.append(1.0 if action == action_type.value else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def update_action_history(self, action: str):
        """Update the action history"""
        self.action_history.append(action)
        if len(self.action_history) > self.config.recent_action_window * 2:
            self.action_history = self.action_history[-self.config.recent_action_window:]
    
    def get_feature_dimension(self) -> int:
        """Get the dimension of the feature vector"""
        return len(self.feature_names)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
        return self.feature_names.copy()
    
    def describe_features(self, features: np.ndarray) -> Dict[str, float]:
        """Create a human-readable description of features"""
        if len(features) != len(self.feature_names):
            raise ValueError(f"Feature vector length {len(features)} doesn't match expected {len(self.feature_names)}")
        
        return dict(zip(self.feature_names, features))

class HistoryBuffer:
    """Buffer for maintaining state and action history"""
    
    def __init__(self, max_length: int = 300):
        self.max_length = max_length
        self.states: List[MinecraftState] = []
        self.actions: List[str] = []
        self.rewards: List[float] = []
        self.timestamps: List[float] = []
    
    def push(self, state: MinecraftState, action: str, reward: float, timestamp: float):
        """Add new entry to history"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.timestamps.append(timestamp)
        
        # Trim if too long
        if len(self.states) > self.max_length:
            self.states = self.states[-self.max_length:]
            self.actions = self.actions[-self.max_length:]
            self.rewards = self.rewards[-self.max_length:]
            self.timestamps = self.timestamps[-self.max_length:]
    
    def get_recent_actions(self, window: int = 5) -> List[str]:
        """Get recent actions"""
        return self.actions[-window:] if len(self.actions) >= window else self.actions
    
    def get_recent_states(self, window: int = 5) -> List[MinecraftState]:
        """Get recent states"""
        return self.states[-window:] if len(self.states) >= window else self.states
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of recent history"""
        if not self.states:
            return {}
        
        recent_states = self.get_recent_states(10)
        recent_actions = self.get_recent_actions(10)
        recent_rewards = self.rewards[-10:] if len(self.rewards) >= 10 else self.rewards
        
        return {
            "avg_hp": np.mean([s.hp for s in recent_states]),
            "avg_hunger": np.mean([s.hunger for s in recent_states]),
            "total_reward": sum(recent_rewards),
            "action_counts": {action: recent_actions.count(action) for action in set(recent_actions)},
            "current_inventory": recent_states[-1].inventory if recent_states else {},
            "time_of_day": recent_states[-1].time_of_day if recent_states else 0.5
        }
