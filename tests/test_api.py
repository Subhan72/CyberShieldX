"""
Unit tests for FastAPI endpoints with mocked analyzers.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api.app import app, conversation_analyzer, llm_verifier


class TestAPI:
    """Test suite for FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_analyzer(self, mock_classification_result):
        """Mock conversation analyzer."""
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
    def mock_llm_verifier_enabled(self, mock_llm_result):
        """Mock LLM verifier (enabled)."""
        verifier = MagicMock()
        verifier.enabled = True
        verifier.verify.return_value = mock_llm_result
        return verifier
    
    @pytest.fixture
    def mock_llm_verifier_disabled(self):
        """Mock LLM verifier (disabled)."""
        verifier = MagicMock()
        verifier.enabled = False
        return verifier
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_check_analyzer_initialized(self, client, mock_analyzer):
        """Test health check - analyzer initialized."""
        import api.app as app_module
        original_analyzer = app_module.conversation_analyzer
        original_verifier = app_module.llm_verifier
        
        try:
            app_module.conversation_analyzer = mock_analyzer
            app_module.llm_verifier = None
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["conversation_analyzer"] is True
            assert data["llm_verifier"] is False
        finally:
            app_module.conversation_analyzer = original_analyzer
            app_module.llm_verifier = original_verifier
    
    def test_health_check_analyzer_not_initialized(self, client):
        """Test health check - analyzer not initialized."""
        import api.app as app_module
        original_analyzer = app_module.conversation_analyzer
        original_verifier = app_module.llm_verifier
        
        try:
            app_module.conversation_analyzer = None
            app_module.llm_verifier = None
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["conversation_analyzer"] is False
        finally:
            app_module.conversation_analyzer = original_analyzer
            app_module.llm_verifier = original_verifier
    
    def test_health_check_llm_enabled(self, client, mock_analyzer, mock_llm_verifier_enabled):
        """Test health check - LLM enabled."""
        import api.app as app_module
        original_analyzer = app_module.conversation_analyzer
        original_verifier = app_module.llm_verifier
        
        try:
            app_module.conversation_analyzer = mock_analyzer
            app_module.llm_verifier = mock_llm_verifier_enabled
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["llm_verifier"] is True
        finally:
            app_module.conversation_analyzer = original_analyzer
            app_module.llm_verifier = original_verifier
    
    def test_analyze_single_text(self, client, mock_analyzer):
        """Test /analyze endpoint with single text."""
        import api.app as app_module
        original_analyzer = app_module.conversation_analyzer
        original_verifier = app_module.llm_verifier
        
        try:
            app_module.conversation_analyzer = mock_analyzer
            app_module.llm_verifier = None
            response = client.post(
                "/analyze",
                json={"text": "You're an idiot!"}
            )
            assert response.status_code == 200
            data = response.json()
            assert 'classification' in data
            assert 'final_label' in data
            assert 'conflict_resolution' in data
            mock_analyzer.analyze.assert_called()
        finally:
            app_module.conversation_analyzer = original_analyzer
            app_module.llm_verifier = original_verifier
    
    def test_analyze_conversation(self, client, mock_analyzer, sample_conversation):
        """Test /analyze endpoint with conversation."""
        import api.app as app_module
        original_analyzer = app_module.conversation_analyzer
        original_verifier = app_module.llm_verifier
        
        try:
            app_module.conversation_analyzer = mock_analyzer
            app_module.llm_verifier = None
            conversation_data = [
                {"user": msg['user'], "message": msg['message']}
                for msg in sample_conversation
            ]
            response = client.post(
                "/analyze",
                json={"conversation": conversation_data}
            )
            assert response.status_code == 200
            data = response.json()
            assert 'classification' in data
            assert 'context' in data
            assert data['context']['num_participants'] == 2
        finally:
            app_module.conversation_analyzer = original_analyzer
            app_module.llm_verifier = original_verifier
    
    def test_analyze_with_llm_verification(self, client, mock_analyzer, mock_llm_verifier_enabled):
        """Test /analyze endpoint with LLM verification."""
        import api.app as app_module
        original_analyzer = app_module.conversation_analyzer
        original_verifier = app_module.llm_verifier
        
        try:
            app_module.conversation_analyzer = mock_analyzer
            app_module.llm_verifier = mock_llm_verifier_enabled
            response = client.post(
                "/analyze",
                json={"text": "You're an idiot!"}
            )
            assert response.status_code == 200
            data = response.json()
            assert 'llm_verification' in data
            assert mock_llm_verifier_enabled.verify.called
        finally:
            app_module.conversation_analyzer = original_analyzer
            app_module.llm_verifier = original_verifier
    
    def test_analyze_input_validation_no_input(self, client, mock_analyzer):
        """Test /analyze endpoint - input validation (no input)."""
        import api.app as app_module
        original_analyzer = app_module.conversation_analyzer
        original_verifier = app_module.llm_verifier
        
        try:
            app_module.conversation_analyzer = mock_analyzer
            app_module.llm_verifier = None
            response = client.post(
                "/analyze",
                json={}
            )
            assert response.status_code == 400
            assert "Either 'text' or 'conversation' must be provided" in response.json()["detail"]
        finally:
            app_module.conversation_analyzer = original_analyzer
            app_module.llm_verifier = original_verifier
    
    def test_analyze_input_validation_empty_conversation(self, client, mock_analyzer):
        """Test /analyze endpoint - input validation (empty conversation)."""
        import api.app as app_module
        original_analyzer = app_module.conversation_analyzer
        original_verifier = app_module.llm_verifier
        
        try:
            app_module.conversation_analyzer = mock_analyzer
            app_module.llm_verifier = None
            response = client.post(
                "/analyze",
                json={"conversation": []}
            )
            assert response.status_code == 400
            assert "Conversation list cannot be empty" in response.json()["detail"]
        finally:
            app_module.conversation_analyzer = original_analyzer
            app_module.llm_verifier = original_verifier
    
    def test_analyze_analyzer_not_initialized(self, client):
        """Test /analyze endpoint - analyzer not initialized."""
        import api.app as app_module
        original_analyzer = app_module.conversation_analyzer
        original_verifier = app_module.llm_verifier
        
        try:
            app_module.conversation_analyzer = None
            app_module.llm_verifier = None
            response = client.post(
                "/analyze",
                json={"text": "Test"}
            )
            assert response.status_code == 503
            assert "not initialized" in response.json()["detail"]
        finally:
            app_module.conversation_analyzer = original_analyzer
            app_module.llm_verifier = original_verifier
    
    def test_analyze_error_handling(self, client, mock_analyzer):
        """Test /analyze endpoint - error handling."""
        import api.app as app_module
        original_analyzer = app_module.conversation_analyzer
        original_verifier = app_module.llm_verifier
        
        mock_analyzer.analyze.side_effect = Exception("Analysis failed")
        
        try:
            app_module.conversation_analyzer = mock_analyzer
            app_module.llm_verifier = None
            response = client.post(
                "/analyze",
                json={"text": "Test"}
            )
            assert response.status_code == 500
            assert "Analysis failed" in response.json()["detail"]
        finally:
            app_module.conversation_analyzer = original_analyzer
            app_module.llm_verifier = original_verifier
    
    def test_analyze_final_label_always_set(self, client, mock_analyzer):
        """Test /analyze endpoint - final_label always set."""
        import api.app as app_module
        original_analyzer = app_module.conversation_analyzer
        original_verifier = app_module.llm_verifier
        
        # Mock analyzer that doesn't return final_label
        mock_analyzer.analyze.return_value = {
            'classification': {'label': 'Insult', 'confidence': 0.8}
        }
        
        try:
            app_module.conversation_analyzer = mock_analyzer
            app_module.llm_verifier = None
            response = client.post(
                "/analyze",
                json={"text": "Test"}
            )
            assert response.status_code == 200
            data = response.json()
            assert 'final_label' in data
            assert data['final_label'] is not None
        finally:
            app_module.conversation_analyzer = original_analyzer
            app_module.llm_verifier = original_verifier
    
    def test_batch_analyze(self, client, mock_analyzer):
        """Test /batch_analyze endpoint."""
        import api.app as app_module
        original_analyzer = app_module.conversation_analyzer
        original_verifier = app_module.llm_verifier
        
        try:
            app_module.conversation_analyzer = mock_analyzer
            app_module.llm_verifier = None
            response = client.post(
                "/batch_analyze",
                json={"texts": ["Text 1", "Text 2", "Text 3"]}
            )
            assert response.status_code == 200
            data = response.json()
            assert 'results' in data
            assert 'count' in data
            assert data['count'] == 3
            assert len(data['results']) == 3
        finally:
            app_module.conversation_analyzer = original_analyzer
            app_module.llm_verifier = original_verifier
    
    def test_batch_analyze_empty_list(self, client, mock_analyzer):
        """Test /batch_analyze endpoint - empty list."""
        import api.app as app_module
        original_analyzer = app_module.conversation_analyzer
        original_verifier = app_module.llm_verifier
        
        try:
            app_module.conversation_analyzer = mock_analyzer
            app_module.llm_verifier = None
            response = client.post(
                "/batch_analyze",
                json={"texts": []}
            )
            # Should still work, just return empty results
            assert response.status_code == 200
            data = response.json()
            assert data['count'] == 0
        finally:
            app_module.conversation_analyzer = original_analyzer
            app_module.llm_verifier = original_verifier
    
    def test_batch_analyze_analyzer_not_initialized(self, client):
        """Test /batch_analyze endpoint - analyzer not initialized."""
        import api.app as app_module
        original_analyzer = app_module.conversation_analyzer
        original_verifier = app_module.llm_verifier
        
        try:
            app_module.conversation_analyzer = None
            app_module.llm_verifier = None
            response = client.post(
                "/batch_analyze",
                json={"texts": ["Test"]}
            )
            assert response.status_code == 503
            assert "not initialized" in response.json()["detail"]
        finally:
            app_module.conversation_analyzer = original_analyzer
            app_module.llm_verifier = original_verifier
    
    def test_batch_analyze_error_handling(self, client, mock_analyzer):
        """Test /batch_analyze endpoint - error handling."""
        import api.app as app_module
        original_analyzer = app_module.conversation_analyzer
        original_verifier = app_module.llm_verifier
        
        mock_analyzer.batch_analyze.side_effect = Exception("Batch analysis failed")
        
        try:
            app_module.conversation_analyzer = mock_analyzer
            app_module.llm_verifier = None
            response = client.post(
                "/batch_analyze",
                json={"texts": ["Test"]}
            )
            assert response.status_code == 500
            assert "Batch analysis failed" in response.json()["detail"]
        finally:
            app_module.conversation_analyzer = original_analyzer
            app_module.llm_verifier = original_verifier
    
    def test_batch_analyze_partial_failures(self, client, mock_analyzer):
        """Test /batch_analyze endpoint - partial failures handled."""
        import api.app as app_module
        original_analyzer = app_module.conversation_analyzer
        original_verifier = app_module.llm_verifier
        
        # Mock batch_analyze to return some errors
        def batch_analyze_side_effect(texts):
            results = []
            for i, text in enumerate(texts):
                if i == 1:  # Second text fails
                    results.append({'error': 'Analysis failed', 'text': text})
                else:
                    results.append({
                        'classification': {'label': 'Normal', 'confidence': 0.9},
                        'final_label': 'Normal'
                    })
            return results
        
        mock_analyzer.batch_analyze.side_effect = batch_analyze_side_effect
        
        try:
            app_module.conversation_analyzer = mock_analyzer
            app_module.llm_verifier = None
            response = client.post(
                "/batch_analyze",
                json={"texts": ["Text 1", "Text 2", "Text 3"]}
            )
            assert response.status_code == 200
            data = response.json()
            assert data['count'] == 3
            assert 'error' in data['results'][1]
            assert 'final_label' in data['results'][0]
        finally:
            app_module.conversation_analyzer = original_analyzer
            app_module.llm_verifier = original_verifier

