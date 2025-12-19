"""
Unit tests for BanterDetector module.
"""

import pytest
from conversational_analysis.banter_detector import BanterDetector


class TestBanterDetector:
    """Test suite for BanterDetector."""
    
    def test_init(self):
        """Test BanterDetector initialization."""
        detector = BanterDetector()
        assert detector.reciprocity_threshold == 0.3
        assert detector.friendly_ratio_threshold == 0.4
        assert detector.mutual_engagement_threshold == 2
        assert detector.tone_consistency_threshold == 0.6
        assert len(detector.severe_indicators) > 0
        assert len(detector.one_sided_patterns) > 0
    
    def test_detect_banter_normal_label(self):
        """Test banter detection with Normal label (early exit)."""
        detector = BanterDetector()
        context = {'has_conversation': True}
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Normal', 0.9
        )
        assert is_banter is False
        assert 'Normal' in reasoning
        assert confidence == 0.9
    
    def test_detect_banter_single_message(self):
        """Test banter detection with single message."""
        detector = BanterDetector()
        context = {
            'has_conversation': False,
            'friendly_indicators': 5,
            'aggressive_indicators': 0,
            'text': 'Hey bro! That was hilarious! 😂'
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert is_banter is True
        assert 'friendly indicators' in reasoning.lower()
    
    def test_detect_banter_single_message_insufficient(self):
        """Test banter detection with single message - insufficient indicators."""
        detector = BanterDetector()
        context = {
            'has_conversation': False,
            'friendly_indicators': 1,
            'aggressive_indicators': 0,
            'text': 'Hello'
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert is_banter is False
        assert 'insufficient' in reasoning.lower()
    
    def test_detect_banter_severe_indicators_override(self):
        """Test Rule 1: Severe indicators override banter detection."""
        detector = BanterDetector()
        context = {
            'has_conversation': True,
            'message_sequence': [
                {'user': 'Alice', 'message': 'You should kill yourself'}
            ]
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert is_banter is False
        assert 'severe' in reasoning.lower()
    
    def test_detect_banter_rule2_reciprocity_high(self):
        """Test Rule 2: High reciprocity indicates banter."""
        detector = BanterDetector()
        context = {
            'has_conversation': True,
            'reciprocity_score': 0.9,
            'mutual_engagement': True,
            'num_participants': 2,
            'num_messages': 4,
            'friendly_indicators': 10,
            'aggressive_indicators': 2,
            'response_patterns': ['playful', 'playful'],
            'tone_consistency': 0.8,
            'relationship_markers': ['friendly_term_bro'],
            'message_sequence': [
                {'user': 'Alice', 'message': 'Test'},
                {'user': 'Bob', 'message': 'Test'}
            ],
            'participants': ['Alice', 'Bob']
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert 'reciprocity' in reasoning.lower()
        # Should be banter due to high scores
        assert is_banter is True
    
    def test_detect_banter_rule2_reciprocity_low(self):
        """Test Rule 2: Low reciprocity indicates not banter."""
        detector = BanterDetector()
        context = {
            'has_conversation': True,
            'reciprocity_score': 0.1,  # Very low
            'mutual_engagement': False,
            'num_participants': 2,
            'num_messages': 4,
            'friendly_indicators': 1,
            'aggressive_indicators': 5,
            'response_patterns': [],
            'tone_consistency': 0.2,
            'relationship_markers': [],
            'message_sequence': [
                {'user': 'Alice', 'message': 'Test'},
                {'user': 'Bob', 'message': 'Test'}
            ],
            'participants': ['Alice', 'Bob']
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert 'low reciprocity' in reasoning.lower()
        assert is_banter is False
    
    def test_detect_banter_rule3_mutual_engagement(self):
        """Test Rule 3: Mutual engagement check."""
        detector = BanterDetector()
        context = {
            'has_conversation': True,
            'reciprocity_score': 0.8,
            'mutual_engagement': True,
            'num_participants': 2,
            'num_messages': 4,
            'friendly_indicators': 8,
            'aggressive_indicators': 1,
            'response_patterns': ['playful'],
            'tone_consistency': 0.7,
            'relationship_markers': [],
            'message_sequence': [
                {'user': 'Alice', 'message': 'Test'},
                {'user': 'Bob', 'message': 'Test'}
            ],
            'participants': ['Alice', 'Bob']
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert 'mutual engagement' in reasoning.lower()
    
    def test_detect_banter_rule4_friendly_ratio_high(self):
        """Test Rule 4: High friendly ratio indicates banter."""
        detector = BanterDetector()
        context = {
            'has_conversation': True,
            'reciprocity_score': 0.8,
            'mutual_engagement': True,
            'num_participants': 2,
            'num_messages': 4,
            'friendly_indicators': 10,
            'aggressive_indicators': 2,  # Ratio = 10/12 = 0.83 > 0.4
            'response_patterns': ['playful'],
            'tone_consistency': 0.7,
            'relationship_markers': [],
            'message_sequence': [
                {'user': 'Alice', 'message': 'Test'},
                {'user': 'Bob', 'message': 'Test'}
            ],
            'participants': ['Alice', 'Bob']
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert 'friendly' in reasoning.lower()
    
    def test_detect_banter_rule4_friendly_ratio_low(self):
        """Test Rule 4: Low friendly ratio indicates not banter."""
        detector = BanterDetector()
        context = {
            'has_conversation': True,
            'reciprocity_score': 0.8,
            'mutual_engagement': True,
            'num_participants': 2,
            'num_messages': 4,
            'friendly_indicators': 2,
            'aggressive_indicators': 8,  # Ratio = 2/10 = 0.2 < 0.4
            'response_patterns': [],
            'tone_consistency': 0.7,
            'relationship_markers': [],
            'message_sequence': [
                {'user': 'Alice', 'message': 'Test'},
                {'user': 'Bob', 'message': 'Test'}
            ],
            'participants': ['Alice', 'Bob']
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert 'low friendly' in reasoning.lower()
    
    def test_detect_banter_rule5_playful_patterns(self):
        """Test Rule 5: Playful response patterns indicate banter."""
        detector = BanterDetector()
        context = {
            'has_conversation': True,
            'reciprocity_score': 0.8,
            'mutual_engagement': True,
            'num_participants': 2,
            'num_messages': 4,
            'friendly_indicators': 8,
            'aggressive_indicators': 1,
            'response_patterns': ['playful', 'playful', 'playful'],
            'tone_consistency': 0.7,
            'relationship_markers': [],
            'message_sequence': [
                {'user': 'Alice', 'message': 'Test'},
                {'user': 'Bob', 'message': 'Test'}
            ],
            'participants': ['Alice', 'Bob']
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert 'playful' in reasoning.lower()
    
    def test_detect_banter_rule5_defensive_patterns(self):
        """Test Rule 5: Defensive response patterns indicate not banter."""
        detector = BanterDetector()
        context = {
            'has_conversation': True,
            'reciprocity_score': 0.8,
            'mutual_engagement': True,
            'num_participants': 2,
            'num_messages': 4,
            'friendly_indicators': 2,
            'aggressive_indicators': 5,
            'response_patterns': ['defensive', 'defensive'],
            'tone_consistency': 0.3,
            'relationship_markers': [],
            'message_sequence': [
                {'user': 'Alice', 'message': 'Test'},
                {'user': 'Bob', 'message': 'Test'}
            ],
            'participants': ['Alice', 'Bob']
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert 'defensive' in reasoning.lower()
    
    def test_detect_banter_rule5_escalating_patterns(self):
        """Test Rule 5: Escalating response patterns indicate not banter."""
        detector = BanterDetector()
        context = {
            'has_conversation': True,
            'reciprocity_score': 0.8,
            'mutual_engagement': True,
            'num_participants': 2,
            'num_messages': 4,
            'friendly_indicators': 1,
            'aggressive_indicators': 8,
            'response_patterns': ['escalating', 'escalating'],
            'tone_consistency': 0.2,
            'relationship_markers': [],
            'message_sequence': [
                {'user': 'Alice', 'message': 'Test'},
                {'user': 'Bob', 'message': 'Test'}
            ],
            'participants': ['Alice', 'Bob']
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert 'escalating' in reasoning.lower() or 'escalation' in reasoning.lower()
        assert is_banter is False
    
    def test_detect_banter_rule6_tone_consistency_high(self):
        """Test Rule 6: High tone consistency indicates banter."""
        detector = BanterDetector()
        context = {
            'has_conversation': True,
            'reciprocity_score': 0.8,
            'mutual_engagement': True,
            'num_participants': 2,
            'num_messages': 4,
            'friendly_indicators': 8,
            'aggressive_indicators': 1,
            'response_patterns': ['playful'],
            'tone_consistency': 0.8,  # High consistency
            'relationship_markers': [],
            'message_sequence': [
                {'user': 'Alice', 'message': 'Test'},
                {'user': 'Bob', 'message': 'Test'}
            ],
            'participants': ['Alice', 'Bob']
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert 'consistent tone' in reasoning.lower()
    
    def test_detect_banter_rule6_tone_consistency_low(self):
        """Test Rule 6: Low tone consistency indicates not banter."""
        detector = BanterDetector()
        context = {
            'has_conversation': True,
            'reciprocity_score': 0.8,
            'mutual_engagement': True,
            'num_participants': 2,
            'num_messages': 4,
            'friendly_indicators': 2,
            'aggressive_indicators': 5,
            'response_patterns': [],
            'tone_consistency': 0.3,  # Low consistency
            'relationship_markers': [],
            'message_sequence': [
                {'user': 'Alice', 'message': 'Test'},
                {'user': 'Bob', 'message': 'Test'}
            ],
            'participants': ['Alice', 'Bob']
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert 'inconsistent' in reasoning.lower()
    
    def test_detect_banter_rule7_one_sided_aggression(self):
        """Test Rule 7: One-sided aggression indicates not banter."""
        detector = BanterDetector()
        context = {
            'has_conversation': True,
            'reciprocity_score': 0.8,
            'mutual_engagement': True,
            'num_participants': 2,
            'num_messages': 4,
            'friendly_indicators': 1,
            'aggressive_indicators': 5,
            'response_patterns': [],
            'tone_consistency': 0.7,
            'relationship_markers': [],
            'message_sequence': [
                {'user': 'Alice', 'message': 'You always do this wrong'},
                {'user': 'Alice', 'message': 'You never listen'},
                {'user': 'Alice', 'message': 'Why are you always like this'},
                {'user': 'Bob', 'message': 'I try'}
            ],
            'participants': ['Alice', 'Bob']
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert 'one-sided' in reasoning.lower()
        assert is_banter is False
    
    def test_detect_banter_rule7_no_one_sided_aggression(self):
        """Test Rule 7: No one-sided aggression indicates potential banter."""
        detector = BanterDetector()
        context = {
            'has_conversation': True,
            'reciprocity_score': 0.8,
            'mutual_engagement': True,
            'num_participants': 2,
            'num_messages': 4,
            'friendly_indicators': 8,
            'aggressive_indicators': 1,
            'response_patterns': ['playful'],
            'tone_consistency': 0.7,
            'relationship_markers': [],
            'message_sequence': [
                {'user': 'Alice', 'message': 'Test'},
                {'user': 'Bob', 'message': 'Test'}
            ],
            'participants': ['Alice', 'Bob']
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert 'no one-sided' in reasoning.lower()
    
    def test_detect_banter_rule8_relationship_markers(self):
        """Test Rule 8: Friendly relationship markers indicate banter."""
        detector = BanterDetector()
        context = {
            'has_conversation': True,
            'reciprocity_score': 0.8,
            'mutual_engagement': True,
            'num_participants': 2,
            'num_messages': 4,
            'friendly_indicators': 8,
            'aggressive_indicators': 1,
            'response_patterns': ['playful'],
            'tone_consistency': 0.7,
            'relationship_markers': ['friendly_term_bro', 'friendly_term_buddy'],
            'message_sequence': [
                {'user': 'Alice', 'message': 'Test'},
                {'user': 'Bob', 'message': 'Test'}
            ],
            'participants': ['Alice', 'Bob']
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert 'relationship markers' in reasoning.lower() or 'friendly' in reasoning.lower()
    
    def test_detect_banter_banter_score_threshold(self):
        """Test banter score threshold (0.6) for decision."""
        detector = BanterDetector()
        # High banter score context
        context = {
            'has_conversation': True,
            'reciprocity_score': 0.9,
            'mutual_engagement': True,
            'num_participants': 2,
            'num_messages': 4,
            'friendly_indicators': 15,
            'aggressive_indicators': 1,
            'response_patterns': ['playful', 'playful', 'playful'],
            'tone_consistency': 0.9,
            'relationship_markers': ['friendly_term_bro'],
            'message_sequence': [
                {'user': 'Alice', 'message': 'Test'},
                {'user': 'Bob', 'message': 'Test'}
            ],
            'participants': ['Alice', 'Bob']
        }
        is_banter, reasoning, confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        assert is_banter is True
        assert '0.6' in reasoning or 'threshold' in reasoning.lower()
    
    def test_detect_banter_confidence_adjustment(self):
        """Test confidence adjustment when banter is detected."""
        detector = BanterDetector()
        context = {
            'has_conversation': True,
            'reciprocity_score': 0.9,
            'mutual_engagement': True,
            'num_participants': 2,
            'num_messages': 4,
            'friendly_indicators': 15,
            'aggressive_indicators': 1,
            'response_patterns': ['playful', 'playful'],
            'tone_consistency': 0.9,
            'relationship_markers': ['friendly_term_bro'],
            'message_sequence': [
                {'user': 'Alice', 'message': 'Test'},
                {'user': 'Bob', 'message': 'Test'}
            ],
            'participants': ['Alice', 'Bob']
        }
        is_banter, reasoning, adjusted_confidence = detector.detect_banter(
            context, 'Insult', 0.8
        )
        if is_banter:
            assert adjusted_confidence > 0.8  # Confidence should increase
    
    def test_check_severe_indicators_suicide(self):
        """Test severe indicators check - suicide."""
        detector = BanterDetector()
        context = {
            'message_sequence': [
                {'user': 'Alice', 'message': 'You should kill yourself'}
            ]
        }
        result = detector._check_severe_indicators(context)
        assert result is True
    
    def test_check_severe_indicators_rape(self):
        """Test severe indicators check - sexual assault."""
        detector = BanterDetector()
        context = {
            'message_sequence': [
                {'user': 'Alice', 'message': 'That was sexual assault'}
            ]
        }
        result = detector._check_severe_indicators(context)
        assert result is True
    
    def test_check_severe_indicators_threat(self):
        """Test severe indicators check - threat."""
        detector = BanterDetector()
        context = {
            'message_sequence': [
                {'user': 'Alice', 'message': 'I will harm you'}
            ]
        }
        result = detector._check_severe_indicators(context)
        assert result is True
    
    def test_check_severe_indicators_single_message(self):
        """Test severe indicators check - single message."""
        detector = BanterDetector()
        context = {
            'text': 'You should kill yourself'
        }
        result = detector._check_severe_indicators(context)
        assert result is True
    
    def test_check_severe_indicators_no_severe(self):
        """Test severe indicators check - no severe indicators."""
        detector = BanterDetector()
        context = {
            'message_sequence': [
                {'user': 'Alice', 'message': 'Hello there'}
            ]
        }
        result = detector._check_severe_indicators(context)
        assert result is False
    
    def test_check_one_sided_aggression_detected(self):
        """Test one-sided aggression detection."""
        detector = BanterDetector()
        context = {
            'participants': ['Alice', 'Bob'],
            'message_sequence': [
                {'user': 'Alice', 'message': 'You always do this wrong'},
                {'user': 'Alice', 'message': 'You never listen'},
                {'user': 'Alice', 'message': 'Why are you always like this'},
                {'user': 'Bob', 'message': 'I try'}
            ]
        }
        result = detector._check_one_sided_aggression(context)
        assert result is True
    
    def test_check_one_sided_aggression_not_detected(self):
        """Test one-sided aggression - not detected (balanced)."""
        detector = BanterDetector()
        context = {
            'participants': ['Alice', 'Bob'],
            'message_sequence': [
                {'user': 'Alice', 'message': 'Hello'},
                {'user': 'Bob', 'message': 'Hi'}
            ]
        }
        result = detector._check_one_sided_aggression(context)
        assert result is False
    
    def test_check_one_sided_aggression_insufficient_participants(self):
        """Test one-sided aggression - insufficient participants."""
        detector = BanterDetector()
        context = {
            'participants': ['Alice'],
            'message_sequence': [
                {'user': 'Alice', 'message': 'Test'}
            ]
        }
        result = detector._check_one_sided_aggression(context)
        assert result is False

