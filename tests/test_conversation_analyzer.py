"""
Unit tests for ConversationAnalyzer module with mocked models.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from conversational_analysis.conversation_analyzer import ConversationAnalyzer


class TestConversationAnalyzer:
    """Test suite for ConversationAnalyzer with mocked models."""
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_init(self, mock_tokenizer, mock_model):
        """Test ConversationAnalyzer initialization."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        
        assert analyzer is not None
        assert analyzer.model_path == './test_model'
        assert analyzer.context_extractor is not None
        assert analyzer.banter_detector is not None
        mock_model_instance.eval.assert_called_once()
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_classify_with_model_normal(self, mock_tokenizer, mock_model):
        """Test model classification - Normal label."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        # Normal has highest logit
        mock_logits = torch.tensor([[2.0, 0.5, 0.3, 0.2, 0.1]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model_instance.return_value = mock_outputs
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        result = analyzer._classify_with_model("Hello there")
        
        assert result['label'] == 'Normal'
        assert result['confidence'] > 0.5
        assert 'probabilities' in result
        assert 'Normal' in result['probabilities']
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_classify_with_model_toxic(self, mock_tokenizer, mock_model):
        """Test model classification - Toxic label."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        # Insult has highest logit
        mock_logits = torch.tensor([[0.5, 2.0, 0.3, 0.2, 0.1]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model_instance.return_value = mock_outputs
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        result = analyzer._classify_with_model("You're an idiot!")
        
        assert result['label'] == 'Insult'
        assert result['confidence'] > 0.5
        assert 'probabilities' in result
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_analyze_single_text(self, mock_tokenizer, mock_model):
        """Test analyze method with single text."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_logits = torch.tensor([[2.0, 0.5, 0.3, 0.2, 0.1]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model_instance.return_value = mock_outputs
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        result = analyzer.analyze(text="Hello there")
        
        assert 'classification' in result
        assert 'conversational_analysis' in result
        assert 'context' in result
        assert 'final_label' in result
        assert 'final_confidence' in result
        assert 'conflict_resolution' in result
        assert result['final_label'] is not None
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_analyze_conversation(self, mock_tokenizer, mock_model, sample_conversation):
        """Test analyze method with conversation."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_logits = torch.tensor([[2.0, 0.5, 0.3, 0.2, 0.1]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model_instance.return_value = mock_outputs
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        result = analyzer.analyze(conversation=sample_conversation)
        
        assert 'classification' in result
        assert 'conversational_analysis' in result
        assert 'context' in result
        assert result['context']['num_participants'] == 2
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_analyze_input_validation_no_input(self, mock_tokenizer, mock_model):
        """Test analyze method input validation - no input."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        
        with pytest.raises(ValueError, match="Either 'text' or 'conversation' must be provided"):
            analyzer.analyze()
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_analyze_input_validation_empty_conversation(self, mock_tokenizer, mock_model):
        """Test analyze method input validation - empty conversation."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        
        with pytest.raises(ValueError, match="Conversation list cannot be empty"):
            analyzer.analyze(conversation=[])
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_resolve_conflict_priority1_banter(self, mock_tokenizer, mock_model):
        """Test conflict resolution - Priority 1: Banter detection."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        
        final_label, final_confidence, conflict_info = analyzer._resolve_conflict(
            model_label='Insult',
            model_confidence=0.85,
            is_banter=True,
            llm_result=None
        )
        
        assert final_label == 'Normal'
        assert conflict_info['resolution_method'] == 'banter_detection'
        assert 'banter' in conflict_info['reasoning'].lower()
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_resolve_conflict_priority2_agreement(self, mock_tokenizer, mock_model, mock_llm_result):
        """Test conflict resolution - Priority 2: Agreement."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        
        final_label, final_confidence, conflict_info = analyzer._resolve_conflict(
            model_label='Insult',
            model_confidence=0.85,
            is_banter=False,
            llm_result=mock_llm_result
        )
        
        assert final_label == 'Insult'
        assert conflict_info['resolution_method'] == 'agreement'
        assert 'agree' in conflict_info['reasoning'].lower()
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_resolve_conflict_priority3_model_high_confidence(self, mock_tokenizer, mock_model, mock_llm_result_disagrees):
        """Test conflict resolution - Priority 3: Model high, LLM low."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        
        # Model high confidence, LLM low confidence
        llm_result = {
            'enabled': True,
            'agrees': False,
            'llm_label': 'Normal',
            'llm_confidence': 0.5,  # Low
            'llm_reasoning': 'Test'
        }
        
        final_label, final_confidence, conflict_info = analyzer._resolve_conflict(
            model_label='Insult',
            model_confidence=0.85,  # High
            is_banter=False,
            llm_result=llm_result
        )
        
        assert final_label == 'Insult'
        assert conflict_info['resolution_method'] == 'model_high_confidence'
        assert conflict_info['conflict_detected'] is True
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_resolve_conflict_priority3_llm_high_confidence(self, mock_tokenizer, mock_model):
        """Test conflict resolution - Priority 3: Model low, LLM high."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        
        # Model low confidence, LLM high confidence
        llm_result = {
            'enabled': True,
            'agrees': False,
            'llm_label': 'Normal',
            'llm_confidence': 0.9,  # High
            'llm_reasoning': 'Test'
        }
        
        final_label, final_confidence, conflict_info = analyzer._resolve_conflict(
            model_label='Insult',
            model_confidence=0.5,  # Low
            is_banter=False,
            llm_result=llm_result
        )
        
        assert final_label == 'Normal'
        assert conflict_info['resolution_method'] == 'llm_high_confidence'
        assert conflict_info['conflict_detected'] is True
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_resolve_conflict_priority3_both_medium(self, mock_tokenizer, mock_model):
        """Test conflict resolution - Priority 3: Both medium confidence."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        
        llm_result = {
            'enabled': True,
            'agrees': False,
            'llm_label': 'Normal',
            'llm_confidence': 0.7,  # Medium
            'llm_reasoning': 'Test'
        }
        
        final_label, final_confidence, conflict_info = analyzer._resolve_conflict(
            model_label='Insult',
            model_confidence=0.7,  # Medium
            is_banter=False,
            llm_result=llm_result
        )
        
        assert final_label == 'Insult'
        assert conflict_info['resolution_method'] == 'model_fallback_medium'
        assert conflict_info['conflict_detected'] is True
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_resolve_conflict_priority3_both_high_model_higher(self, mock_tokenizer, mock_model):
        """Test conflict resolution - Priority 3: Both high, model higher."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        
        llm_result = {
            'enabled': True,
            'agrees': False,
            'llm_label': 'Normal',
            'llm_confidence': 0.85,  # High
            'llm_reasoning': 'Test'
        }
        
        final_label, final_confidence, conflict_info = analyzer._resolve_conflict(
            model_label='Insult',
            model_confidence=0.9,  # Higher
            is_banter=False,
            llm_result=llm_result
        )
        
        assert final_label == 'Insult'
        assert conflict_info['resolution_method'] == 'higher_confidence_model'
        assert conflict_info['conflict_detected'] is True
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_resolve_conflict_priority3_both_high_llm_higher(self, mock_tokenizer, mock_model):
        """Test conflict resolution - Priority 3: Both high, LLM higher."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        
        llm_result = {
            'enabled': True,
            'agrees': False,
            'llm_label': 'Normal',
            'llm_confidence': 0.95,  # Higher
            'llm_reasoning': 'Test'
        }
        
        final_label, final_confidence, conflict_info = analyzer._resolve_conflict(
            model_label='Insult',
            model_confidence=0.85,  # High but lower
            is_banter=False,
            llm_result=llm_result
        )
        
        assert final_label == 'Normal'
        assert conflict_info['resolution_method'] == 'higher_confidence_llm'
        assert conflict_info['conflict_detected'] is True
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_resolve_conflict_priority4_fallback(self, mock_tokenizer, mock_model):
        """Test conflict resolution - Priority 4: Final fallback."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        
        llm_result = {
            'enabled': True,
            'agrees': False,
            'llm_label': 'Normal',
            'llm_confidence': 0.55,  # Edge case
            'llm_reasoning': 'Test'
        }
        
        final_label, final_confidence, conflict_info = analyzer._resolve_conflict(
            model_label='Insult',
            model_confidence=0.55,  # Edge case
            is_banter=False,
            llm_result=llm_result
        )
        
        assert final_label == 'Insult'
        assert conflict_info['resolution_method'] == 'model_fallback'
        assert conflict_info['conflict_detected'] is True
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_resolve_conflict_no_llm(self, mock_tokenizer, mock_model):
        """Test conflict resolution - No LLM verification."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        
        final_label, final_confidence, conflict_info = analyzer._resolve_conflict(
            model_label='Insult',
            model_confidence=0.85,
            is_banter=False,
            llm_result=None
        )
        
        assert final_label == 'Insult'
        assert conflict_info['resolution_method'] == 'model_only'
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_analyze_with_llm_result(self, mock_tokenizer, mock_model, mock_llm_result):
        """Test analyze method with LLM result."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_logits = torch.tensor([[2.0, 0.5, 0.3, 0.2, 0.1]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model_instance.return_value = mock_outputs
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        result = analyzer.analyze(text="Hello", llm_result=mock_llm_result)
        
        assert 'llm_verification' in result
        assert result['llm_verification'] == mock_llm_result
    
    @patch('conversational_analysis.conversation_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('conversational_analysis.conversation_analyzer.AutoTokenizer.from_pretrained')
    def test_batch_analyze(self, mock_tokenizer, mock_model):
        """Test batch_analyze method."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_logits = torch.tensor([[2.0, 0.5, 0.3, 0.2, 0.1]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model_instance.return_value = mock_outputs
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        analyzer = ConversationAnalyzer(model_path='./test_model')
        texts = ["Hello", "How are you?", "Goodbye"]
        results = analyzer.batch_analyze(texts)
        
        assert len(results) == 3
        assert all('final_label' in r for r in results)

