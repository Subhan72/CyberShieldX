"""
Unit tests for LLMVerifier module with mocked LLM.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, Mock
from llm_verification.llm_verifier import LLMVerifier


class TestLLMVerifier:
    """Test suite for LLMVerifier with mocked LLM."""
    
    @patch('llm_verification.llm_verifier.LLAMA_CPP_AVAILABLE', True)
    @patch('llm_verification.llm_verifier.Llama')
    @patch('os.path.exists')
    def test_init_with_model(self, mock_exists, mock_llama):
        """Test LLMVerifier initialization with model path."""
        mock_exists.return_value = True
        mock_llama_instance = MagicMock()
        mock_llama.return_value = mock_llama_instance
        
        verifier = LLMVerifier(model_path='./test_model.gguf')
        
        assert verifier.model_path == './test_model.gguf'
        assert verifier.enabled is True
        assert verifier.model == mock_llama_instance
        mock_llama.assert_called_once()
    
    @patch('llm_verification.llm_verifier.LLAMA_CPP_AVAILABLE', True)
    @patch('llm_verification.llm_verifier.Llama')
    @patch('os.path.exists')
    def test_init_without_model(self, mock_exists, mock_llama):
        """Test LLMVerifier initialization without model path."""
        mock_exists.return_value = False
        
        verifier = LLMVerifier(model_path=None)
        
        assert verifier.enabled is False
        assert verifier.model is None
        mock_llama.assert_not_called()
    
    @patch('llm_verification.llm_verifier.LLAMA_CPP_AVAILABLE', False)
    def test_init_llama_not_available(self):
        """Test LLMVerifier initialization when llama_cpp not available."""
        verifier = LLMVerifier(model_path='./test_model.gguf')
        
        assert verifier.enabled is False
        assert verifier.model is None
    
    @patch('llm_verification.llm_verifier.LLAMA_CPP_AVAILABLE', True)
    @patch('llm_verification.llm_verifier.Llama')
    @patch('os.path.exists')
    def test_init_model_load_error(self, mock_exists, mock_llama):
        """Test LLMVerifier initialization with model load error."""
        mock_exists.return_value = True
        mock_llama.side_effect = Exception("Model load failed")
        
        verifier = LLMVerifier(model_path='./test_model.gguf')
        
        assert verifier.enabled is False
        assert verifier.model is None
    
    @patch('llm_verification.llm_verifier.LLAMA_CPP_AVAILABLE', True)
    @patch('llm_verification.llm_verifier.Llama')
    @patch('os.path.exists')
    def test_verify_enabled_json_response(self, mock_exists, mock_llama):
        """Test verify method - enabled with JSON response."""
        mock_exists.return_value = True
        mock_llama_instance = MagicMock()
        mock_llama_instance.return_value = {
            'choices': [{
                'text': '{"label": "Normal", "is_banter": true, "reasoning": "Friendly banter", "confidence": 0.85}'
            }]
        }
        mock_llama.return_value = mock_llama_instance
        
        verifier = LLMVerifier(model_path='./test_model.gguf')
        result = verifier.verify(
            text="You're such a nerd! 😂",
            model_label="Insult",
            model_confidence=0.8
        )
        
        assert result['enabled'] is True
        assert result['llm_label'] == 'Normal'
        assert result['agrees'] is not None
        assert 'llm_reasoning' in result
        assert result['confidence'] > 0
    
    @patch('llm_verification.llm_verifier.LLAMA_CPP_AVAILABLE', True)
    @patch('llm_verification.llm_verifier.Llama')
    @patch('os.path.exists')
    def test_verify_enabled_text_response(self, mock_exists, mock_llama):
        """Test verify method - enabled with text response (fallback parsing)."""
        mock_exists.return_value = True
        mock_llama_instance = MagicMock()
        mock_llama_instance.return_value = {
            'choices': [{
                'text': 'This is Normal text. It seems like friendly banter between friends.'
            }]
        }
        mock_llama.return_value = mock_llama_instance
        
        verifier = LLMVerifier(model_path='./test_model.gguf')
        result = verifier.verify(
            text="You're such a nerd! 😂",
            model_label="Insult",
            model_confidence=0.8
        )
        
        assert result['enabled'] is True
        assert result['llm_label'] in ['Normal', 'Unknown']
        assert 'llm_reasoning' in result
    
    @patch('llm_verification.llm_verifier.LLAMA_CPP_AVAILABLE', True)
    @patch('llm_verification.llm_verifier.Llama')
    @patch('os.path.exists')
    def test_verify_disabled(self, mock_exists, mock_llama):
        """Test verify method - disabled."""
        mock_exists.return_value = False
        
        verifier = LLMVerifier(model_path=None)
        result = verifier.verify(
            text="Test",
            model_label="Insult",
            model_confidence=0.8
        )
        
        assert result['enabled'] is False
        assert result['agrees'] is None
        assert result['llm_label'] is None
    
    @patch('llm_verification.llm_verifier.LLAMA_CPP_AVAILABLE', True)
    @patch('llm_verification.llm_verifier.Llama')
    @patch('os.path.exists')
    def test_verify_error_handling(self, mock_exists, mock_llama):
        """Test verify method - error handling."""
        mock_exists.return_value = True
        mock_llama_instance = MagicMock()
        mock_llama_instance.side_effect = Exception("Generation failed")
        mock_llama.return_value = mock_llama_instance
        
        verifier = LLMVerifier(model_path='./test_model.gguf')
        result = verifier.verify(
            text="Test",
            model_label="Insult",
            model_confidence=0.8
        )
        
        assert result['enabled'] is True
        assert result['agrees'] is None
        assert 'error' in result['llm_reasoning'].lower()
    
    @patch('llm_verification.llm_verifier.LLAMA_CPP_AVAILABLE', True)
    @patch('llm_verification.llm_verifier.Llama')
    @patch('os.path.exists')
    def test_build_verification_prompt_single_text(self, mock_exists, mock_llama):
        """Test _build_verification_prompt with single text."""
        mock_exists.return_value = True
        mock_llama_instance = MagicMock()
        mock_llama.return_value = mock_llama_instance
        
        verifier = LLMVerifier(model_path='./test_model.gguf')
        prompt = verifier._build_verification_prompt(
            text="You're an idiot!",
            conversation=None,
            model_label="Insult"
        )
        
        assert "You're an idiot!" in prompt
        assert "Insult" in prompt
        assert "JSON" in prompt
        assert "label" in prompt
    
    @patch('llm_verification.llm_verifier.LLAMA_CPP_AVAILABLE', True)
    @patch('llm_verification.llm_verifier.Llama')
    @patch('os.path.exists')
    def test_build_verification_prompt_conversation(self, mock_exists, mock_llama, sample_conversation):
        """Test _build_verification_prompt with conversation."""
        mock_exists.return_value = True
        mock_llama_instance = MagicMock()
        mock_llama.return_value = mock_llama_instance
        
        verifier = LLMVerifier(model_path='./test_model.gguf')
        prompt = verifier._build_verification_prompt(
            text="",
            conversation=sample_conversation,
            model_label="Insult"
        )
        
        assert "Conversation thread" in prompt
        assert "Alice" in prompt or "Bob" in prompt
        assert "Insult" in prompt
    
    @patch('llm_verification.llm_verifier.LLAMA_CPP_AVAILABLE', True)
    @patch('llm_verification.llm_verifier.Llama')
    @patch('os.path.exists')
    def test_parse_llm_response_json_valid(self, mock_exists, mock_llama):
        """Test _parse_llm_response with valid JSON."""
        mock_exists.return_value = True
        mock_llama_instance = MagicMock()
        mock_llama.return_value = mock_llama_instance
        
        verifier = LLMVerifier(model_path='./test_model.gguf')
        response = '{"label": "Normal", "is_banter": true, "reasoning": "Friendly", "confidence": 0.9}'
        result = verifier._parse_llm_response(response, "Insult")
        
        assert result['llm_label'] == 'Normal'
        assert result['is_banter'] is True
        assert result['reasoning'] == 'Friendly'
        assert result['confidence'] == 0.9
        assert 'agrees' in result
    
    @patch('llm_verification.llm_verifier.LLAMA_CPP_AVAILABLE', True)
    @patch('llm_verification.llm_verifier.Llama')
    @patch('os.path.exists')
    def test_parse_llm_response_json_invalid(self, mock_exists, mock_llama):
        """Test _parse_llm_response with invalid JSON (fallback to text)."""
        mock_exists.return_value = True
        mock_llama_instance = MagicMock()
        mock_llama.return_value = mock_llama_instance
        
        verifier = LLMVerifier(model_path='./test_model.gguf')
        response = 'This is Normal text with banter indicators.'
        result = verifier._parse_llm_response(response, "Insult")
        
        assert result['llm_label'] in ['Normal', 'Unknown']
        assert 'reasoning' in result
        assert 'agrees' in result
    
    @patch('llm_verification.llm_verifier.LLAMA_CPP_AVAILABLE', True)
    @patch('llm_verification.llm_verifier.Llama')
    @patch('os.path.exists')
    def test_determine_agreement_model_normal_llm_normal(self, mock_exists, mock_llama):
        """Test agreement determination - both Normal."""
        mock_exists.return_value = True
        mock_llama_instance = MagicMock()
        mock_llama.return_value = mock_llama_instance
        
        verifier = LLMVerifier(model_path='./test_model.gguf')
        # This is tested through _parse_llm_response
        response = '{"label": "Normal", "is_banter": false, "reasoning": "Test", "confidence": 0.8}'
        result = verifier._parse_llm_response(response, "Normal")
        
        assert result['agrees'] is True
    
    @patch('llm_verification.llm_verifier.LLAMA_CPP_AVAILABLE', True)
    @patch('llm_verification.llm_verifier.Llama')
    @patch('os.path.exists')
    def test_determine_agreement_model_toxic_llm_banter(self, mock_exists, mock_llama):
        """Test agreement determination - model toxic, LLM thinks banter."""
        mock_exists.return_value = True
        mock_llama_instance = MagicMock()
        mock_llama.return_value = mock_llama_instance
        
        verifier = LLMVerifier(model_path='./test_model.gguf')
        response = '{"label": "Normal", "is_banter": true, "reasoning": "Banter", "confidence": 0.8}'
        result = verifier._parse_llm_response(response, "Insult")
        
        # If LLM thinks it's banter (Normal), it agrees with banter detection
        assert result['agrees'] is True  # LLM says Normal (banter), which is correct
    
    @patch('llm_verification.llm_verifier.LLAMA_CPP_AVAILABLE', True)
    @patch('llm_verification.llm_verifier.Llama')
    @patch('os.path.exists')
    def test_determine_agreement_model_toxic_llm_toxic_same(self, mock_exists, mock_llama):
        """Test agreement determination - both toxic, same label."""
        mock_exists.return_value = True
        mock_llama_instance = MagicMock()
        mock_llama.return_value = mock_llama_instance
        
        verifier = LLMVerifier(model_path='./test_model.gguf')
        response = '{"label": "Insult", "is_banter": false, "reasoning": "Real insult", "confidence": 0.9}'
        result = verifier._parse_llm_response(response, "Insult")
        
        assert result['agrees'] is True
    
    @patch('llm_verification.llm_verifier.LLAMA_CPP_AVAILABLE', True)
    @patch('llm_verification.llm_verifier.Llama')
    @patch('os.path.exists')
    def test_verify_with_conversation(self, mock_exists, mock_llama, sample_conversation):
        """Test verify method with conversation context."""
        mock_exists.return_value = True
        mock_llama_instance = MagicMock()
        mock_llama_instance.return_value = {
            'choices': [{
                'text': '{"label": "Normal", "is_banter": true, "reasoning": "Friendly", "confidence": 0.85}'
            }]
        }
        mock_llama.return_value = mock_llama_instance
        
        verifier = LLMVerifier(model_path='./test_model.gguf')
        result = verifier.verify(
            text="",
            conversation=sample_conversation,
            model_label="Insult",
            model_confidence=0.8
        )
        
        assert result['enabled'] is True
        assert result['llm_label'] == 'Normal'
        # Verify that conversation was included in prompt
        mock_llama_instance.assert_called_once()
        call_args = mock_llama_instance.call_args[0][0]
        assert "Alice" in call_args or "Bob" in call_args

