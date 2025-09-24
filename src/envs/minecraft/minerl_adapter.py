"""
MineRL Environment Adapter for LLM-Guided Learning
Provides a standardized interface for Minecraft environments
"""

import numpy as np
import gym
from typing import Dict, Any, Tuple, Optional, Protocol
from dataclasses import dataclass
import time

# Try to import MineRL, fallback to mock if not available
try:
    import minerl
    MINERL_AVAILABLE = True
except ImportError:
    MINERL_AVAILABLE = False
    print("MineRL not available, using mock environment")

@dataclass
class MinecraftState:
    """Parsed Minecraft state"""
    hp: float  # 0-20
    hunger: float  # 0-20
    time_of_day: float  # 0-1 (0=dawn, 0.5=noon, 1=midnight)
    position: np.ndarray  # [x, y, z]
    yaw: float  # -180 to 180
    pitch: float  # -90 to 90
    inventory: Dict[str, int]  # item counts
    hostile_count: int  # nearby hostile mobs
    light_level: float  # 0-15

class EnvAdapter(Protocol):
    """Protocol for environment adapters"""
    def reset(self, seed: Optional[int] = None) -> MinecraftState: ...
    def step(self, action: Dict[str, Any]) -> Tuple[MinecraftState, float, bool, Dict]: ...
    def render(self) -> Optional[np.ndarray]: ...

class MockMineRLAdapter:
    """Mock MineRL environment for testing"""
    
    def __init__(self, env_name: str = "MineRLTreechop-v0"):
        self.env_name = env_name
        self.time_step = 0
        self.max_steps = 1000
        self.wood_collected = 0
        self.position = np.array([0.0, 64.0, 0.0])
        
    def reset(self, seed: Optional[int] = None) -> MinecraftState:
        if seed is not None:
            np.random.seed(seed)
        
        self.time_step = 0
        self.wood_collected = 0
        self.position = np.array([0.0, 64.0, 0.0])
        
        return self._get_state()
    
    def step(self, action: Dict[str, Any]) -> Tuple[MinecraftState, float, bool, Dict]:
        self.time_step += 1
        
        # Simulate wood collection
        if action.get("attack", 0) > 0 and np.random.random() < 0.1:
            self.wood_collected += 1
        
        # Simulate movement
        if "camera" in action:
            camera = action["camera"]
            if len(camera) >= 2:
                self.position[0] += camera[0] * 0.1
                self.position[2] += camera[1] * 0.1
        
        state = self._get_state()
        reward = self.wood_collected * 0.1
        done = self.time_step >= self.max_steps or self.wood_collected >= 10
        info = {"wood_collected": self.wood_collected}
        
        return state, reward, done, info
    
    def render(self) -> Optional[np.ndarray]:
        # Return a simple mock image
        return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    def _get_state(self) -> MinecraftState:
        return MinecraftState(
            hp=20.0,
            hunger=20.0,
            time_of_day=min(1.0, self.time_step / 1000.0),
            position=self.position.copy(),
            yaw=0.0,
            pitch=0.0,
            inventory={"log": self.wood_collected, "planks": 0, "stick": 0},
            hostile_count=0,
            light_level=15.0
        )

class MineRLAdapter:
    """Real MineRL environment adapter"""
    
    def __init__(self, env_name: str = "MineRLTreechop-v0"):
        if not MINERL_AVAILABLE:
            raise ImportError("MineRL is not available. Install with: pip install minerl")
        
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.last_obs = None
        
    def reset(self, seed: Optional[int] = None) -> MinecraftState:
        if seed is not None:
            self.env.seed(seed)
        
        obs = self.env.reset()
        self.last_obs = obs
        return self._parse_observation(obs)
    
    def step(self, action: Dict[str, Any]) -> Tuple[MinecraftState, float, bool, Dict]:
        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs
        state = self._parse_observation(obs)
        return state, reward, done, info
    
    def render(self) -> Optional[np.ndarray]:
        return self.env.render()
    
    def _parse_observation(self, obs) -> MinecraftState:
        """Parse MineRL observation into standardized state"""
        # Extract basic stats
        hp = float(obs.get("life_stats", {}).get("life", 20.0))
        hunger = float(obs.get("life_stats", {}).get("food", 20.0))
        
        # Extract position and orientation
        pos = obs.get("location_stats", {})
        position = np.array([
            pos.get("xpos", 0.0),
            pos.get("ypos", 64.0),
            pos.get("zpos", 0.0)
        ])
        yaw = float(pos.get("yaw", 0.0))
        pitch = float(pos.get("pitch", 0.0))
        
        # Extract inventory
        inventory = {}
        if "inventory" in obs:
            inv = obs["inventory"]
            for item_name in ["log", "planks", "stick", "crafting_table", "wooden_pickaxe"]:
                if item_name in inv:
                    inventory[item_name] = int(inv[item_name])
                else:
                    inventory[item_name] = 0
        
        # Estimate time of day (simplified)
        time_of_day = 0.5  # Default to noon
        
        # Estimate hostile count (simplified)
        hostile_count = 0
        
        # Estimate light level (simplified)
        light_level = 15.0
        
        return MinecraftState(
            hp=hp,
            hunger=hunger,
            time_of_day=time_of_day,
            position=position,
            yaw=yaw,
            pitch=pitch,
            inventory=inventory,
            hostile_count=hostile_count,
            light_level=light_level
        )

def create_adapter(env_name: str = "MineRLTreechop-v0", use_mock: bool = None) -> EnvAdapter:
    """Factory function to create appropriate adapter"""
    if use_mock is None:
        use_mock = not MINERL_AVAILABLE
    
    if use_mock:
        print(f"Using mock adapter for {env_name}")
        return MockMineRLAdapter(env_name)
    else:
        print(f"Using real MineRL adapter for {env_name}")
        return MineRLAdapter(env_name)
