"""
Macro Library for Minecraft Actions
Defines high-level actions that can be executed by the agent
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import numpy as np

class MacroType(Enum):
    COLLECT_WOOD = "collect_wood"
    CRAFT_PLANKS = "craft_planks"
    CRAFT_TABLE = "craft_table"
    BUILD_SHELTER = "build_shelter_2x2"
    MINE_STONE = "mine_stone"
    RETREAT_TO_BASE = "retreat_to_base"
    IDLE_SCAN = "idle_scan"

@dataclass
class MacroSpec:
    """Specification for a macro action"""
    name: str
    macro_type: MacroType
    preconditions: Dict[str, Any]  # Required conditions to execute
    timeout_s: float  # Maximum execution time
    params: Optional[Dict[str, Any]] = None  # Additional parameters
    description: str = ""

@dataclass
class MacroReport:
    """Report of macro execution"""
    success: bool
    duration_s: float
    events: List[str]  # Events that occurred during execution
    delta_inv: Dict[str, int]  # Changes in inventory
    final_state: Optional[Any] = None
    error_message: Optional[str] = None

# Define all available macros
MACRO_LIBRARY = {
    MacroType.COLLECT_WOOD: MacroSpec(
        name="collect_wood",
        macro_type=MacroType.COLLECT_WOOD,
        preconditions={"near_tree": True, "has_tool": False},  # No tool needed for punching
        timeout_s=10.0,
        description="Collect wood by punching trees"
    ),
    
    MacroType.CRAFT_PLANKS: MacroSpec(
        name="craft_planks",
        macro_type=MacroType.CRAFT_PLANKS,
        preconditions={"has_log": True},
        timeout_s=5.0,
        description="Craft planks from logs"
    ),
    
    MacroType.CRAFT_TABLE: MacroSpec(
        name="craft_table",
        macro_type=MacroType.CRAFT_TABLE,
        preconditions={"has_planks": True, "plank_count": 4},
        timeout_s=5.0,
        description="Craft a crafting table"
    ),
    
    MacroType.BUILD_SHELTER: MacroSpec(
        name="build_shelter_2x2",
        macro_type=MacroType.BUILD_SHELTER,
        preconditions={"has_planks": True, "plank_count": 16, "safe_location": True},
        timeout_s=30.0,
        description="Build a simple 2x2 shelter"
    ),
    
    MacroType.MINE_STONE: MacroSpec(
        name="mine_stone",
        macro_type=MacroType.MINE_STONE,
        preconditions={"has_pickaxe": True, "near_stone": True},
        timeout_s=15.0,
        description="Mine stone blocks"
    ),
    
    MacroType.RETREAT_TO_BASE: MacroSpec(
        name="retreat_to_base",
        macro_type=MacroType.RETREAT_TO_BASE,
        preconditions={"has_base": True},
        timeout_s=20.0,
        description="Return to base/shelter"
    ),
    
    MacroType.IDLE_SCAN: MacroSpec(
        name="idle_scan",
        macro_type=MacroType.IDLE_SCAN,
        preconditions={},  # No preconditions
        timeout_s=2.0,
        description="Look around and scan environment"
    )
}

class MacroMask:
    """Utility class to determine which macros are available given current state"""
    
    @staticmethod
    def from_state(state) -> List[MacroType]:
        """Return list of available macros based on current state"""
        available = []
        
        # Always available
        available.append(MacroType.IDLE_SCAN)
        
        # Check wood collection
        if MacroMask._near_tree(state):
            available.append(MacroType.COLLECT_WOOD)
        
        # Check crafting
        if state.inventory.get("log", 0) > 0:
            available.append(MacroType.CRAFT_PLANKS)
        
        if state.inventory.get("planks", 0) >= 4:
            available.append(MacroType.CRAFT_TABLE)
        
        # Check building
        if state.inventory.get("planks", 0) >= 16:
            available.append(MacroType.BUILD_SHELTER)
        
        # Check mining
        if (state.inventory.get("wooden_pickaxe", 0) > 0 and 
            MacroMask._near_stone(state)):
            available.append(MacroType.MINE_STONE)
        
        # Check retreat (if base exists)
        if MacroMask._has_base(state):
            available.append(MacroType.RETREAT_TO_BASE)
        
        return available
    
    @staticmethod
    def _near_tree(state) -> bool:
        """Check if near a tree (simplified)"""
        # In real implementation, would check for nearby log blocks
        return True  # Assume always near trees for now
    
    @staticmethod
    def _near_stone(state) -> bool:
        """Check if near stone blocks"""
        # In real implementation, would check for nearby stone blocks
        return state.position[1] < 60  # Assume underground = near stone
    
    @staticmethod
    def _has_base(state) -> bool:
        """Check if player has built a base"""
        # In real implementation, would track base location
        return state.inventory.get("crafting_table", 0) > 0

def get_macro_by_name(name: str) -> Optional[MacroSpec]:
    """Get macro specification by name"""
    for macro_type, spec in MACRO_LIBRARY.items():
        if spec.name == name:
            return spec
    return None

def get_all_macro_names() -> List[str]:
    """Get list of all macro names"""
    return [spec.name for spec in MACRO_LIBRARY.values()]
