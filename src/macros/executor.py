"""
Macro Executor for Minecraft Actions
Converts high-level macro commands into low-level MineRL actions
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .library import MacroSpec, MacroReport, MacroType
from ..envs.minecraft.minerl_adapter import EnvAdapter, MinecraftState

class MacroExecutor:
    """Executes macro actions in the Minecraft environment"""
    
    def __init__(self, env: EnvAdapter):
        self.env = env
        self.base_location: Optional[np.ndarray] = None
        self.execution_history: List[MacroReport] = []
    
    def execute(self, macro: MacroSpec, budget_s: float) -> MacroReport:
        """Execute a macro action with time budget"""
        start_time = time.time()
        initial_state = self._get_current_state()
        initial_inventory = initial_state.inventory.copy()
        
        events = []
        success = False
        error_message = None
        
        try:
            if macro.macro_type == MacroType.COLLECT_WOOD:
                success, events = self._execute_collect_wood(budget_s)
            elif macro.macro_type == MacroType.CRAFT_PLANKS:
                success, events = self._execute_craft_planks()
            elif macro.macro_type == MacroType.CRAFT_TABLE:
                success, events = self._execute_craft_table()
            elif macro.macro_type == MacroType.BUILD_SHELTER:
                success, events = self._execute_build_shelter(budget_s)
            elif macro.macro_type == MacroType.MINE_STONE:
                success, events = self._execute_mine_stone(budget_s)
            elif macro.macro_type == MacroType.RETREAT_TO_BASE:
                success, events = self._execute_retreat_to_base(budget_s)
            elif macro.macro_type == MacroType.IDLE_SCAN:
                success, events = self._execute_idle_scan()
            else:
                error_message = f"Unknown macro type: {macro.macro_type}"
                
        except Exception as e:
            error_message = str(e)
            events.append(f"error: {error_message}")
        
        duration = time.time() - start_time
        final_state = self._get_current_state()
        
        # Calculate inventory changes
        delta_inv = {}
        for item in set(initial_inventory.keys()) | set(final_state.inventory.keys()):
            initial_count = initial_inventory.get(item, 0)
            final_count = final_state.inventory.get(item, 0)
            if initial_count != final_count:
                delta_inv[item] = final_count - initial_count
        
        report = MacroReport(
            success=success,
            duration_s=duration,
            events=events,
            delta_inv=delta_inv,
            final_state=final_state,
            error_message=error_message
        )
        
        self.execution_history.append(report)
        return report
    
    def _get_current_state(self) -> MinecraftState:
        """Get current state from environment"""
        # This is a simplified approach - in real implementation,
        # we'd need to track state properly
        return MinecraftState(
            hp=20.0, hunger=20.0, time_of_day=0.5,
            position=np.array([0.0, 64.0, 0.0]),
            yaw=0.0, pitch=0.0,
            inventory={"log": 0, "planks": 0, "crafting_table": 0},
            hostile_count=0, light_level=15.0
        )
    
    def _execute_collect_wood(self, budget_s: float) -> tuple[bool, List[str]]:
        """Execute wood collection macro"""
        events = []
        start_time = time.time()
        wood_collected = 0
        
        # Look for trees and punch them
        for step in range(int(budget_s * 10)):  # 10 steps per second
            if time.time() - start_time > budget_s:
                break
            
            # Simulate looking around for trees
            action = {
                "camera": [np.random.uniform(-5, 5), np.random.uniform(-5, 5)],
                "attack": 1,
                "forward": 1 if step % 20 < 10 else 0
            }
            
            state, reward, done, info = self.env.step(action)
            
            # Check if wood was collected
            if "wood_collected" in info and info["wood_collected"] > wood_collected:
                wood_collected = info["wood_collected"]
                events.append(f"collected_wood: {wood_collected}")
            
            if done:
                break
        
        success = wood_collected > 0
        if success:
            events.append(f"wood_collection_complete: {wood_collected} logs")
        else:
            events.append("wood_collection_failed: no logs found")
        
        return success, events
    
    def _execute_craft_planks(self) -> tuple[bool, List[str]]:
        """Execute plank crafting macro"""
        events = []
        
        # Simulate crafting planks from logs
        # In real MineRL, this would involve inventory manipulation
        action = {"craft": "planks"}
        state, reward, done, info = self.env.step(action)
        
        events.append("crafted_planks")
        return True, events
    
    def _execute_craft_table(self) -> tuple[bool, List[str]]:
        """Execute crafting table creation macro"""
        events = []
        
        # Simulate crafting table creation
        action = {"craft": "crafting_table"}
        state, reward, done, info = self.env.step(action)
        
        events.append("crafted_table")
        return True, events
    
    def _execute_build_shelter(self, budget_s: float) -> tuple[bool, List[str]]:
        """Execute shelter building macro"""
        events = []
        start_time = time.time()
        
        # Simulate building a 2x2 shelter
        for step in range(int(budget_s * 2)):  # 2 steps per second for building
            if time.time() - start_time > budget_s:
                break
            
            # Place blocks to form shelter
            action = {
                "place": "planks",
                "camera": [0, 0],
                "jump": 0
            }
            
            state, reward, done, info = self.env.step(action)
            events.append(f"placed_block: step_{step}")
            
            if done:
                break
        
        events.append("shelter_construction_complete")
        
        # Set base location
        current_state = self._get_current_state()
        self.base_location = current_state.position.copy()
        
        return True, events
    
    def _execute_mine_stone(self, budget_s: float) -> tuple[bool, List[str]]:
        """Execute stone mining macro"""
        events = []
        start_time = time.time()
        stone_mined = 0
        
        # Simulate mining stone
        for step in range(int(budget_s * 5)):  # 5 steps per second
            if time.time() - start_time > budget_s:
                break
            
            action = {
                "attack": 1,
                "camera": [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
            }
            
            state, reward, done, info = self.env.step(action)
            
            # Simulate stone collection
            if step % 10 == 0:  # Collect stone every 10 steps
                stone_mined += 1
                events.append(f"mined_stone: {stone_mined}")
            
            if done:
                break
        
        success = stone_mined > 0
        events.append(f"mining_complete: {stone_mined} stone")
        
        return success, events
    
    def _execute_retreat_to_base(self, budget_s: float) -> tuple[bool, List[str]]:
        """Execute retreat to base macro"""
        events = []
        
        if self.base_location is None:
            events.append("no_base_location")
            return False, events
        
        start_time = time.time()
        current_state = self._get_current_state()
        
        # Calculate direction to base
        direction = self.base_location - current_state.position
        distance = np.linalg.norm(direction)
        
        if distance < 2.0:
            events.append("already_at_base")
            return True, events
        
        # Move towards base
        for step in range(int(budget_s * 10)):
            if time.time() - start_time > budget_s:
                break
            
            # Simplified movement towards base
            action = {
                "forward": 1,
                "camera": [direction[0] * 0.1, direction[2] * 0.1]
            }
            
            state, reward, done, info = self.env.step(action)
            
            if done:
                break
        
        events.append("retreat_complete")
        return True, events
    
    def _execute_idle_scan(self) -> tuple[bool, List[str]]:
        """Execute idle scanning macro"""
        events = []
        
        # Look around to scan environment
        for i in range(4):  # Look in 4 directions
            action = {
                "camera": [90, 0],  # Turn 90 degrees
                "jump": 0
            }
            state, reward, done, info = self.env.step(action)
        
        events.append("environment_scan_complete")
        return True, events
