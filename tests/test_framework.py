#!/usr/bin/env python3
"""
Test suite for LLM-Guided Classical ML framework.
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_elm_hybrid import LLMTeacher, TraditionalELM, ActivationReversedELM, RobotEnvironment

class TestLLMTeacher:
    """Test LLM Teacher functionality."""
    
    def test_dummy_evaluator(self):
        """Test dummy evaluator returns valid scores."""
        teacher = LLMTeacher(use_openai=False)
        
        context = {
            'robot_pos': [0.5, 0.5],
            'target_pos': [1.0, 1.0],
            'distance': 0.7,
            'collisions': 2,
            'steps': 50
        }
        
        score, reasoning = teacher.evaluate_performance(context, {})
        
        assert 0.0 <= score <= 1.0
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
    
    @patch('openai.OpenAI')
    def test_openai_evaluator_mock(self, mock_openai):
        """Test OpenAI evaluator with mocked API."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "SCORE: 0.75\nREASONING: Good performance"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        teacher = LLMTeacher(use_openai=True)
        
        context = {
            'robot_pos': [0.5, 0.5],
            'target_pos': [1.0, 1.0],
            'distance': 0.7,
            'collisions': 2,
            'steps': 50
        }
        
        score, reasoning = teacher.evaluate_performance(context, {})
        
        assert score == 0.75
        assert "Good performance" in reasoning

class TestELMModels:
    """Test ELM model implementations."""
    
    def test_traditional_elm_basic(self):
        """Test basic Traditional ELM functionality."""
        np.random.seed(42)
        
        # Generate test data
        X = np.random.randn(100, 4)
        y = np.random.randn(100, 2)
        
        model = TraditionalELM(n_hidden=10, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        
        assert predictions.shape == (10, 2)
        assert not np.any(np.isnan(predictions))
    
    def test_activation_reversed_elm_basic(self):
        """Test basic Activation Reversed ELM functionality."""
        np.random.seed(42)
        
        # Generate test data
        X = np.random.randn(100, 4)
        y = np.random.randn(100, 2)
        
        model = ActivationReversedELM(n_hidden=10, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        
        assert predictions.shape == (10, 2)
        assert not np.any(np.isnan(predictions))
    
    def test_elm_reproducibility(self):
        """Test that ELM models are reproducible with same seed."""
        X = np.random.randn(50, 4)
        y = np.random.randn(50, 2)
        
        # Train two models with same seed
        model1 = TraditionalELM(n_hidden=10, random_state=42)
        model2 = TraditionalELM(n_hidden=10, random_state=42)
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        pred1 = model1.predict(X[:5])
        pred2 = model2.predict(X[:5])
        
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)

class TestRobotEnvironment:
    """Test robot simulation environment."""
    
    def test_robot_environment_initialization(self):
        """Test robot environment initializes correctly."""
        env = RobotEnvironment(random_state=42)
        
        assert hasattr(env, 'robot_pos')
        assert hasattr(env, 'target_pos')
        assert hasattr(env, 'obstacles')
        assert len(env.robot_pos) == 2
        assert len(env.target_pos) == 2
    
    def test_robot_step_function(self):
        """Test robot step function."""
        env = RobotEnvironment(random_state=42)
        initial_pos = env.robot_pos.copy()
        
        # Take a step
        action = np.array([0.1, 0.1])
        distance, collision = env.step(action)
        
        assert isinstance(distance, float)
        assert isinstance(collision, bool)
        assert not np.array_equal(env.robot_pos, initial_pos)
    
    def test_robot_reset_function(self):
        """Test robot reset function."""
        env = RobotEnvironment(random_state=42)
        
        # Take some steps
        for _ in range(5):
            action = np.random.randn(2) * 0.1
            env.step(action)
        
        # Reset environment
        env.reset()
        
        # Should be back to initial state
        assert hasattr(env, 'robot_pos')
        assert hasattr(env, 'target_pos')

class TestReproducibility:
    """Test reproducibility of experiments."""
    
    def test_numpy_seed_consistency(self):
        """Test that numpy random seed produces consistent results."""
        np.random.seed(42)
        random1 = np.random.randn(10)
        
        np.random.seed(42)
        random2 = np.random.randn(10)
        
        np.testing.assert_array_equal(random1, random2)
    
    def test_experiment_reproducibility(self):
        """Test that experiments are reproducible."""
        # This would test the full experiment pipeline
        # For now, just test that the components are deterministic
        
        np.random.seed(42)
        env1 = RobotEnvironment(random_state=42)
        model1 = TraditionalELM(n_hidden=10, random_state=42)
        
        np.random.seed(42)
        env2 = RobotEnvironment(random_state=42)
        model2 = TraditionalELM(n_hidden=10, random_state=42)
        
        # Environments should be identical
        np.testing.assert_array_equal(env1.robot_pos, env2.robot_pos)
        np.testing.assert_array_equal(env1.target_pos, env2.target_pos)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
