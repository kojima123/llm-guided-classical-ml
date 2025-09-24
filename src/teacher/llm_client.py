"""
LLM Teacher Client for Minecraft
Provides natural language evaluation and guidance for macro selection
"""

import json
import hashlib
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import os

# Try to import OpenAI, fallback to mock if not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

@dataclass
class LLMEvaluation:
    """LLM evaluation result"""
    progress: float  # 0.0-1.0
    safety: float    # 0.0-1.0
    efficiency: float # 0.0-1.0
    suggested_policy: str
    confidence: float # 0.0-1.0
    comment: str
    raw_response: str = ""

class LLMTeacher:
    """LLM-based teacher for Minecraft macro learning"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 temperature: float = 0.0,
                 max_tokens: int = 200,
                 cache_ttl_s: int = 600):
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_ttl_s = cache_ttl_s
        
        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE and self.api_key:
            openai.api_key = self.api_key
            self.use_real_llm = True
        else:
            self.use_real_llm = False
            print("Using mock LLM teacher (OpenAI not available or no API key)")
        
        # Simple in-memory cache
        self.cache: Dict[str, tuple[LLMEvaluation, float]] = {}
        
        # Available macros for validation
        self.valid_macros = [
            "collect_wood", "craft_planks", "craft_table", 
            "build_shelter_2x2", "mine_stone", "retreat_to_base", "idle_scan"
        ]
    
    def evaluate(self, state_summary: str, goal: str = "Day1 Survival") -> LLMEvaluation:
        """Evaluate current situation and provide guidance"""
        
        # Create cache key
        cache_key = self._create_cache_key(state_summary, goal)
        
        # Check cache
        if cache_key in self.cache:
            cached_eval, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl_s:
                return cached_eval
        
        # Generate evaluation
        if self.use_real_llm:
            evaluation = self._evaluate_with_openai(state_summary, goal)
        else:
            evaluation = self._evaluate_with_mock(state_summary, goal)
        
        # Cache result
        self.cache[cache_key] = (evaluation, time.time())
        
        return evaluation
    
    def _create_cache_key(self, state_summary: str, goal: str) -> str:
        """Create cache key from inputs"""
        content = f"{goal}|{state_summary}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _evaluate_with_openai(self, state_summary: str, goal: str) -> LLMEvaluation:
        """Evaluate using real OpenAI API"""
        
        prompt = self._create_prompt(state_summary, goal)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            response_text = response.choices[0].message.content.strip()
            return self._parse_response(response_text)
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._evaluate_with_mock(state_summary, goal)
    
    def _evaluate_with_mock(self, state_summary: str, goal: str) -> LLMEvaluation:
        """Mock evaluation for testing"""
        
        # Simple heuristic-based evaluation
        progress = 0.5
        safety = 0.8
        efficiency = 0.6
        suggested_policy = "collect_wood"
        confidence = 0.7
        comment = "Continue gathering resources"
        
        # Parse state summary for better heuristics
        if "wood:" in state_summary:
            wood_count = self._extract_number(state_summary, "wood:")
            if wood_count >= 10:
                progress = 0.8
                suggested_policy = "craft_planks"
                comment = "Good wood collection, craft planks"
        
        if "hp:" in state_summary:
            hp = self._extract_number(state_summary, "hp:")
            if hp < 10:
                safety = 0.3
                suggested_policy = "retreat_to_base"
                comment = "Low health, seek safety"
        
        if "time_of_day:" in state_summary:
            time_val = self._extract_number(state_summary, "time_of_day:")
            if time_val > 0.8:  # Night time
                safety = 0.4
                if "crafting_table:0" in state_summary:
                    suggested_policy = "build_shelter_2x2"
                    comment = "Night approaching, build shelter"
                else:
                    suggested_policy = "retreat_to_base"
                    comment = "Night time, stay safe"
        
        return LLMEvaluation(
            progress=progress,
            safety=safety,
            efficiency=efficiency,
            suggested_policy=suggested_policy,
            confidence=confidence,
            comment=comment,
            raw_response=f"Mock evaluation for: {state_summary[:50]}..."
        )
    
    def _extract_number(self, text: str, prefix: str) -> float:
        """Extract number following a prefix in text"""
        try:
            start = text.find(prefix) + len(prefix)
            end = start
            while end < len(text) and (text[end].isdigit() or text[end] == '.'):
                end += 1
            return float(text[start:end])
        except:
            return 0.0
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """あなたはMinecraftの戦術コーチです。数値ルーブリックで厳密に評価し、必ずJSONのみを出力してください。

評価基準:
- progress: 目標達成の進捗度 (0.0-1.0)
- safety: プレイヤーの安全性 (0.0-1.0) 
- efficiency: 行動の効率性 (0.0-1.0)
- suggested_policy: 推奨する次の行動
- confidence: 評価の信頼度 (0.0-1.0)
- comment: 20文字以内の簡潔な助言

利用可能な行動: collect_wood, craft_planks, craft_table, build_shelter_2x2, mine_stone, retreat_to_base, idle_scan"""
    
    def _create_prompt(self, state_summary: str, goal: str) -> str:
        """Create prompt for LLM evaluation"""
        return f"""現在の目的: {goal}

状態要約: {state_summary}

上記の状況を評価し、以下のJSON形式で回答してください:
{{"progress": 0.0-1.0, "safety": 0.0-1.0, "efficiency": 0.0-1.0, "suggested_policy": "行動名", "confidence": 0.0-1.0, "comment": "助言"}}"""
    
    def _parse_response(self, response_text: str) -> LLMEvaluation:
        """Parse LLM response into evaluation object"""
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                data = json.loads(json_text)
                
                # Validate and clamp values
                progress = max(0.0, min(1.0, float(data.get('progress', 0.5))))
                safety = max(0.0, min(1.0, float(data.get('safety', 0.5))))
                efficiency = max(0.0, min(1.0, float(data.get('efficiency', 0.5))))
                confidence = max(0.0, min(1.0, float(data.get('confidence', 0.5))))
                
                suggested_policy = data.get('suggested_policy', 'idle_scan')
                if suggested_policy not in self.valid_macros:
                    suggested_policy = 'idle_scan'
                
                comment = str(data.get('comment', 'No comment'))[:50]  # Limit length
                
                return LLMEvaluation(
                    progress=progress,
                    safety=safety,
                    efficiency=efficiency,
                    suggested_policy=suggested_policy,
                    confidence=confidence,
                    comment=comment,
                    raw_response=response_text
                )
            
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
        
        # Fallback to mock evaluation
        return self._evaluate_with_mock("", "")
    
    def clear_cache(self):
        """Clear evaluation cache"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        valid_entries = sum(1 for _, (_, timestamp) in self.cache.items() 
                          if current_time - timestamp < self.cache_ttl_s)
        
        return {
            'total_entries': len(self.cache),
            'valid_entries': valid_entries,
            'cache_ttl_s': self.cache_ttl_s,
            'use_real_llm': self.use_real_llm
        }

class RubricWeights:
    """Weights for combining evaluation scores"""
    
    def __init__(self, progress: float = 0.5, safety: float = 0.3, efficiency: float = 0.2):
        total = progress + safety + efficiency
        self.progress = progress / total
        self.safety = safety / total
        self.efficiency = efficiency / total
    
    def compute_reward(self, evaluation: LLMEvaluation) -> float:
        """Compute weighted reward from evaluation"""
        return (self.progress * evaluation.progress + 
                self.safety * evaluation.safety + 
                self.efficiency * evaluation.efficiency)
    
    def compute_sample_weight(self, evaluation: LLMEvaluation, 
                            base_weight: float = 1.0) -> float:
        """Compute sample weight for training"""
        reward = self.compute_reward(evaluation)
        confidence_factor = evaluation.confidence
        
        # Weight is higher for better performance and higher confidence
        return base_weight * confidence_factor * (0.5 + 0.5 * reward)
