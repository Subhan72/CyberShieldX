"""
Shared pytest fixtures and mocks for unit tests.

This module provides reusable test fixtures and mock objects for testing the
toxic content classification system. Fixtures include sample data (texts,
conversations) and mock objects for models, tokenizers, and LLM components.

All fixtures are designed to be isolated and reusable across different test modules.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, Mock, patch
from typing import Dict, List, Optional


# Sample test data fixtures
@pytest.fixture
def sample_text():
    """
    Sample single text message for testing.
    
    Returns a toxic message that should be classified as "Insult" by the model.
    """
    return "You're such an idiot!"


@pytest.fixture
def sample_friendly_text():
    """Sample friendly text message."""
    return "Hey bro, that was hilarious! 😂"


@pytest.fixture
def sample_conversation():
    """Sample conversation thread."""
    return [
        {'user': 'Alice', 'message': 'Hey, how are you?'},
        {'user': 'Bob', 'message': 'I\'m good, thanks! How about you?'},
        {'user': 'Alice', 'message': 'Great! Just working on some stuff.'}
    ]


@pytest.fixture
def sample_banter_conversation():
    """Sample banter conversation."""
    return [
        {'user': 'Alice', 'message': 'You\'re such a nerd! 😂'},
        {'user': 'Bob', 'message': 'Haha, you\'re one to talk! 😄'},
        {'user': 'Alice', 'message': 'LOL, true! We\'re both nerds! 🤣'}
    ]


@pytest.fixture
def sample_toxic_conversation():
    """Sample toxic conversation."""
    return [
        {'user': 'Alice', 'message': 'You\'re such an idiot!'},
        {'user': 'Bob', 'message': 'Why are you always like this?'},
        {'user': 'Alice', 'message': 'Shut up, you worthless piece of trash!'}
    ]


@pytest.fixture
def sample_context():
    """Sample context dictionary."""
    return {
        'has_conversation': True,
        'num_participants': 2,
        'num_messages': 3,
        'participants': ['Alice', 'Bob'],
        'message_sequence': [
            {'user': 'Alice', 'message': 'Test message 1'},
            {'user': 'Bob', 'message': 'Test message 2'},
            {'user': 'Alice', 'message': 'Test message 3'}
        ],
        'reciprocity_score': 0.8,
        'mutual_engagement': True,
        'friendly_indicators': 5,
        'aggressive_indicators': 1,
        'response_patterns': ['playful', 'playful'],
        'tone_consistency': 0.7,
        'relationship_markers': ['friendly_term_bro']
    }


@pytest.fixture
def mock_tokenizer():
    """
    Mock tokenizer for XLM-RoBERTa model.
    
    Returns a MagicMock object that simulates the HuggingFace tokenizer,
    returning tokenized input tensors in the expected format.
    """
    tokenizer = MagicMock()
    tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
    }
    return tokenizer


@pytest.fixture
def mock_model():
    """
    Mock XLM-RoBERTa model that returns "Normal" classification.
    
    This fixture creates a mock model that simulates XLM-RoBERTa inference,
    returning logits that favor the "Normal" label. Used for testing scenarios
    where non-toxic content is expected.
    """
    model = MagicMock()
    
    # Mock model outputs
    # Simulate logits for 5 labels: [Normal, Insult, Hate Speech, Flaming, Sexual Harassment]
    mock_logits = torch.tensor([[2.0, 0.5, 0.3, 0.2, 0.1]])  # Normal has highest logit
    
    # Create mock outputs object
    mock_outputs = MagicMock()
    mock_outputs.logits = mock_logits
    
    # Mock model forward pass
    model.return_value = mock_outputs
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)
    
    return model


@pytest.fixture
def mock_model_toxic():
    """
    Mock XLM-RoBERTa model that returns "Insult" classification.
    
    This fixture creates a mock model that simulates XLM-RoBERTa inference,
    returning logits that favor the "Insult" label. Used for testing scenarios
    where toxic content is expected.
    """
    model = MagicMock()
    
    # Mock model outputs - Insult has highest logit
    mock_logits = torch.tensor([[0.5, 2.0, 0.3, 0.2, 0.1]])
    
    mock_outputs = MagicMock()
    mock_outputs.logits = mock_logits
    
    model.return_value = mock_outputs
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)
    
    return model


@pytest.fixture
def mock_llm_model():
    """Mock LLM (Llama) model."""
    llm = MagicMock()
    
    # Mock LLM response
    mock_response = {
        'choices': [{
            'text': '{"label": "Normal", "is_banter": true, "reasoning": "Friendly banter", "confidence": 0.85}'
        }]
    }
    
    llm.return_value = mock_response
    return llm


@pytest.fixture
def mock_llm_response_agrees():
    """Mock LLM response that agrees with model."""
    return {
        'choices': [{
            'text': '{"label": "Insult", "is_banter": false, "reasoning": "Real insult", "confidence": 0.90}'
        }]
    }


@pytest.fixture
def mock_llm_response_disagrees():
    """Mock LLM response that disagrees with model."""
    return {
        'choices': [{
            'text': '{"label": "Normal", "is_banter": true, "reasoning": "This is banter", "confidence": 0.75}'
        }]
    }


@pytest.fixture
def mock_classification_result():
    """Mock model classification result."""
    return {
        'label': 'Insult',
        'confidence': 0.85,
        'probabilities': {
            'Normal': 0.1,
            'Insult': 0.85,
            'Hate Speech': 0.03,
            'Flaming': 0.01,
            'Sexual Harassment': 0.01
        }
    }


@pytest.fixture
def mock_classification_result_normal():
    """Mock model classification result for Normal."""
    return {
        'label': 'Normal',
        'confidence': 0.90,
        'probabilities': {
            'Normal': 0.90,
            'Insult': 0.05,
            'Hate Speech': 0.02,
            'Flaming': 0.02,
            'Sexual Harassment': 0.01
        }
    }


@pytest.fixture
def mock_llm_result():
    """Mock LLM verification result."""
    return {
        'enabled': True,
        'agrees': True,
        'llm_label': 'Insult',
        'llm_reasoning': 'This is clearly an insult',
        'confidence': 0.88
    }


@pytest.fixture
def mock_llm_result_disagrees():
    """Mock LLM verification result that disagrees."""
    return {
        'enabled': True,
        'agrees': False,
        'llm_label': 'Normal',
        'llm_reasoning': 'This is friendly banter',
        'confidence': 0.75
    }


@pytest.fixture
def mock_conversation_analyzer(mock_classification_result):
    """Mock ConversationAnalyzer instance."""
    analyzer = MagicMock()
    
    def analyze_side_effect(text=None, conversation=None, llm_result=None):
        # Determine if it's a conversation or single text
        is_conversation = conversation is not None and len(conversation) > 0
        num_participants = len(set([msg.get('user', 'unknown') for msg in conversation])) if is_conversation else 1
        num_messages = len(conversation) if is_conversation else 1
        
        result = {
            'classification': {
                'label': mock_classification_result['label'],
                'confidence': mock_classification_result['confidence'],
                'probabilities': mock_classification_result['probabilities']
            },
            'conversational_analysis': {
                'is_banter': False,
                'reasoning': 'Not banter',
                'context_used': is_conversation
            },
            'context': {
                'num_participants': num_participants,
                'num_messages': num_messages,
                'reciprocity_score': 0.8 if is_conversation else 0.0,
                'mutual_engagement': is_conversation,
                'friendly_indicators': 2 if is_conversation else 0,
                'aggressive_indicators': 1
            },
            'final_label': mock_classification_result['label'],
            'final_confidence': mock_classification_result['confidence'],
            'conflict_resolution': {
                'conflict_detected': False,
                'resolution_method': 'model_only',
                'reasoning': 'Using model result'
            }
        }
        
        # Add LLM verification if provided
        if llm_result:
            result['llm_verification'] = llm_result
        
        return result
    
    analyzer.analyze.side_effect = analyze_side_effect
    
    # Mock batch_analyze
    def batch_analyze_side_effect(texts):
        return [analyze_side_effect(text=text) for text in texts]
    
    analyzer.batch_analyze.side_effect = batch_analyze_side_effect
    
    return analyzer


@pytest.fixture
def mock_llm_verifier(mock_llm_result):
    """Mock LLMVerifier instance."""
    verifier = MagicMock()
    verifier.enabled = True
    verifier.verify.return_value = mock_llm_result
    return verifier


@pytest.fixture
def mock_llm_verifier_disabled():
    """Mock LLMVerifier instance (disabled)."""
    verifier = MagicMock()
    verifier.enabled = False
    verifier.verify.return_value = {
        'enabled': False,
        'agrees': None,
        'llm_label': None,
        'llm_reasoning': 'LLM verification not available',
        'confidence': 0.0
    }
    return verifier

