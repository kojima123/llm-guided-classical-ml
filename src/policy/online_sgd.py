"""
Online SGD Policy for Minecraft Macro Selection
Uses SGDClassifier for real-time learning of macro selection
"""

import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

from ..macros.library import MacroType, get_all_macro_names

class SGDPolicy:
    """Online SGD policy for macro selection"""
    
    def __init__(self, 
                 macros: List[str] = None,
                 learning_rate: float = 1e-3,
                 epsilon_start: float = 0.3,
                 epsilon_final: float = 0.05,
                 epsilon_decay_steps: int = 500,
                 random_state: int = 42):
        
        self.macros = macros or get_all_macro_names()
        self.n_classes = len(self.macros)
        self.macro_to_idx = {macro: i for i, macro in enumerate(self.macros)}
        self.idx_to_macro = {i: macro for i, macro in enumerate(self.macros)}
        
        # Exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay_steps = epsilon_decay_steps
        self.step_count = 0
        
        # Initialize classifier
        self.classifier = SGDClassifier(
            loss='log_loss',  # For probability estimates
            learning_rate='constant',
            eta0=learning_rate,
            random_state=random_state,
            warm_start=True
        )
        
        # Feature scaling
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Initialize with dummy data to set up classes
        self._initialize_classifier()
        
        # Training history
        self.training_history = []
        
    def _initialize_classifier(self):
        """Initialize classifier with dummy data to establish classes"""
        n_features = 50  # Placeholder, will be updated with real data
        dummy_X = np.random.randn(self.n_classes, n_features)
        dummy_y = list(range(self.n_classes))
        
        self.classifier.fit(dummy_X, dummy_y)
        self.scaler.fit(dummy_X)
        self.scaler_fitted = True
    
    def predict(self, features: np.ndarray, 
                available_macros: List[str] = None,
                mask: List[MacroType] = None) -> str:
        """Predict next macro action"""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Update scaler if needed
        if not self.scaler_fitted or features.shape[1] != len(self.scaler.mean_):
            self.scaler = StandardScaler()
            self.scaler.fit(features)
            self.scaler_fitted = True
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get current epsilon for exploration
        epsilon = self._get_current_epsilon()
        
        # Determine available actions
        if available_macros is not None:
            available_indices = [self.macro_to_idx[macro] for macro in available_macros 
                               if macro in self.macro_to_idx]
        elif mask is not None:
            available_indices = [self.macro_to_idx[macro_type.value] for macro_type in mask 
                               if macro_type.value in self.macro_to_idx]
        else:
            available_indices = list(range(self.n_classes))
        
        if not available_indices:
            # Fallback to idle_scan if no actions available
            return "idle_scan"
        
        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            # Random exploration
            chosen_idx = np.random.choice(available_indices)
        else:
            # Greedy exploitation
            try:
                probabilities = self.classifier.predict_proba(features_scaled)[0]
                # Mask unavailable actions
                masked_probs = np.full(self.n_classes, -np.inf)
                masked_probs[available_indices] = probabilities[available_indices]
                chosen_idx = np.argmax(masked_probs)
            except:
                # Fallback to random if prediction fails
                chosen_idx = np.random.choice(available_indices)
        
        self.step_count += 1
        return self.idx_to_macro[chosen_idx]
    
    def partial_fit(self, features: np.ndarray, action: str, 
                   sample_weight: float = 1.0):
        """Update policy with new experience"""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        if action not in self.macro_to_idx:
            print(f"Warning: Unknown action {action}, skipping update")
            return
        
        # Update scaler if needed
        if not self.scaler_fitted or features.shape[1] != len(self.scaler.mean_):
            # Partial fit for scaler
            self.scaler.partial_fit(features)
        else:
            # Update existing scaler
            self.scaler.partial_fit(features)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Convert action to label
        label = self.macro_to_idx[action]
        
        # Update classifier
        try:
            self.classifier.partial_fit(features_scaled, [label], 
                                      sample_weight=[sample_weight])
            
            # Record training history
            self.training_history.append({
                'step': self.step_count,
                'action': action,
                'sample_weight': sample_weight,
                'epsilon': self._get_current_epsilon()
            })
            
        except Exception as e:
            print(f"Warning: Failed to update classifier: {e}")
    
    def _get_current_epsilon(self) -> float:
        """Get current epsilon value for exploration"""
        if self.step_count >= self.epsilon_decay_steps:
            return self.epsilon_final
        
        # Linear decay
        decay_ratio = self.step_count / self.epsilon_decay_steps
        return self.epsilon_start + (self.epsilon_final - self.epsilon_start) * decay_ratio
    
    def get_action_probabilities(self, features: np.ndarray) -> Dict[str, float]:
        """Get probability distribution over actions"""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        if not self.scaler_fitted:
            return {macro: 1.0/len(self.macros) for macro in self.macros}
        
        try:
            features_scaled = self.scaler.transform(features)
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            return {self.idx_to_macro[i]: prob for i, prob in enumerate(probabilities)}
        except:
            # Fallback to uniform distribution
            return {macro: 1.0/len(self.macros) for macro in self.macros}
    
    def save(self, filepath: str):
        """Save policy to file"""
        policy_data = {
            'macros': self.macros,
            'macro_to_idx': self.macro_to_idx,
            'idx_to_macro': self.idx_to_macro,
            'classifier': self.classifier,
            'scaler': self.scaler,
            'scaler_fitted': self.scaler_fitted,
            'step_count': self.step_count,
            'training_history': self.training_history,
            'epsilon_start': self.epsilon_start,
            'epsilon_final': self.epsilon_final,
            'epsilon_decay_steps': self.epsilon_decay_steps
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(policy_data, f)
    
    def load(self, filepath: str):
        """Load policy from file"""
        with open(filepath, 'rb') as f:
            policy_data = pickle.load(f)
        
        self.macros = policy_data['macros']
        self.macro_to_idx = policy_data['macro_to_idx']
        self.idx_to_macro = policy_data['idx_to_macro']
        self.classifier = policy_data['classifier']
        self.scaler = policy_data['scaler']
        self.scaler_fitted = policy_data['scaler_fitted']
        self.step_count = policy_data['step_count']
        self.training_history = policy_data['training_history']
        self.epsilon_start = policy_data['epsilon_start']
        self.epsilon_final = policy_data['epsilon_final']
        self.epsilon_decay_steps = policy_data['epsilon_decay_steps']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get policy statistics"""
        return {
            'step_count': self.step_count,
            'current_epsilon': self._get_current_epsilon(),
            'n_classes': self.n_classes,
            'scaler_fitted': self.scaler_fitted,
            'training_samples': len(self.training_history),
            'action_distribution': self._get_action_distribution()
        }
    
    def _get_action_distribution(self) -> Dict[str, int]:
        """Get distribution of actions in training history"""
        action_counts = {}
        for record in self.training_history:
            action = record['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        return action_counts
