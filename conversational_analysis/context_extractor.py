"""
Context Extractor Module

This module extracts contextual information from conversations to support banter detection
and toxic content classification. It analyzes conversation structure, participant engagement,
language patterns, and relationship indicators.

The extractor processes both single messages and conversation threads, identifying:
- Participant information and engagement patterns
- Reciprocity scores (balanced participation)
- Friendly vs aggressive language indicators
- Response patterns (playful, defensive, escalating)
- Tone consistency across messages
- Relationship markers (friendly terms, mentions)
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
import emoji


class ContextExtractor:
    """
    Extracts contextual information from conversations for analysis.
    
    This class analyzes conversation structure and language patterns to provide
    contextual features that help distinguish friendly banter from real cyberbullying.
    It processes both single messages and conversation threads, extracting features
    such as participant engagement, language indicators, and response patterns.
    
    Attributes:
        friendly_emojis: List of emoji characters that indicate friendly/positive sentiment
        casual_phrases: List of casual phrases commonly used in friendly banter
        positive_words: List of positive words that suggest friendly interaction
    """
    
    def __init__(self):
        # Friendly language indicators used for banter detection
        # These patterns help identify friendly interactions vs aggressive ones
        self.friendly_emojis = ['😊', '😄', '😂', '😆', '😜', '😎', '🤣', '😉', '😋', '😍', 
                                '❤️', '💕', '👍', '👌', '✌️', '🤝', '🙌', '🎉', '🔥', '💯']
        self.casual_phrases = ['lol', 'haha', 'lmao', 'rofl', 'hehe', 'hahaha', 'lolol', 
                               'bro', 'dude', 'man', 'buddy', 'pal', 'mate', 'friend']
        self.positive_words = ['love', 'great', 'awesome', 'amazing', 'cool', 'nice', 
                               'good', 'best', 'fun', 'happy', 'glad', 'thanks', 'thank you']
        
    def extract_context(self, text: str, conversation: Optional[List[Dict]] = None) -> Dict:
        """
        Extract context from text or conversation thread.
        
        This is the main entry point for context extraction. It routes to appropriate
        extraction methods based on whether a conversation thread is provided or just
        a single message.
        
        Args:
            text: Single message text (used for single message analysis)
            conversation: Optional list of conversation messages, each containing:
                         - 'user': User identifier (string)
                         - 'message': Message text (string)
                         - 'timestamp': Optional timestamp (string)
        
        Returns:
            Dictionary containing extracted context features:
            - 'has_conversation': Boolean indicating if conversation thread was provided
            - 'num_participants': Number of unique participants
            - 'num_messages': Total number of messages
            - 'participants': List of participant identifiers
            - 'message_sequence': List of message dictionaries
            - 'reciprocity_score': Score indicating balanced participation (0.0-1.0)
            - 'mutual_engagement': Boolean indicating mutual engagement
            - 'friendly_indicators': Count of friendly language indicators
            - 'aggressive_indicators': Count of aggressive language indicators
            - 'response_patterns': List of response pattern classifications
            - 'tone_consistency': Score indicating tone consistency (0.0-1.0)
            - 'relationship_markers': List of relationship marker strings
        """
        context = {
            'has_conversation': conversation is not None and len(conversation) > 1,
            'num_participants': 0,
            'num_messages': 1,
            'participants': [],
            'message_sequence': [],
            'reciprocity_score': 0.0,
            'mutual_engagement': False,
            'friendly_indicators': 0,
            'aggressive_indicators': 0,
            'response_patterns': [],
            'tone_consistency': 0.0,
            'relationship_markers': []
        }
        
        if conversation and len(conversation) > 1:
            # Extract from conversation thread
            context.update(self._extract_from_conversation(conversation))
        else:
            # Extract from single message
            context.update(self._extract_from_single_message(text))
            
        return context
    
    def _extract_from_conversation(self, conversation: List[Dict]) -> Dict:
        """
        Extract context from a conversation thread.
        
        This method performs comprehensive analysis of a multi-message conversation,
        extracting features such as participant engagement, language patterns, and
        response dynamics.
        
        Args:
            conversation: List of message dictionaries with 'user' and 'message' keys
        
        Returns:
            Dictionary with extracted conversation context features
        """
        context = {
            'num_messages': len(conversation),
            'participants': list(set([msg.get('user', 'unknown') for msg in conversation])),
            'num_participants': len(set([msg.get('user', 'unknown') for msg in conversation])),
            'message_sequence': conversation
        }
        
        # Analyze message patterns
        user_messages = defaultdict(list)
        for msg in conversation:
            user = msg.get('user', 'unknown')
            message_text = msg.get('message', '')
            user_messages[user].append(message_text)
        
        # Calculate reciprocity score: measures how balanced participation is
        # High reciprocity (close to 1.0) means both users engage similarly (characteristic of banter)
        # Low reciprocity (close to 0.0) means one user dominates (may indicate bullying)
        if len(context['participants']) >= 2:
            message_counts = [len(user_messages[user]) for user in context['participants']]
            if len(message_counts) >= 2:
                # Reciprocity formula: ratio of minimum to maximum message count
                # Example: User A sends 3 messages, User B sends 5 messages
                # Reciprocity = 3/5 = 0.6 (moderate balance)
                min_message_count = min(message_counts)
                max_message_count = max(message_counts)
                if max_message_count > 0:
                    context['reciprocity_score'] = min_message_count / max_message_count
                # Mutual engagement threshold: reciprocity > 0.3 indicates some balance
                context['mutual_engagement'] = context['reciprocity_score'] > 0.3
        
        # Analyze friendly vs aggressive indicators across conversation
        friendly_count = 0
        aggressive_count = 0
        response_patterns = []
        
        for i, msg in enumerate(conversation):
            text = msg.get('message', '').lower()
            
        # Count friendly language indicators
        friendly_count += self._count_friendly_language_indicators(text)
        
        # Count aggressive language indicators
        aggressive_count += self._count_aggressive_language_indicators(text)
            
            # Analyze response patterns
            if i > 0:
                prev_text = conversation[i-1].get('message', '').lower()
                pattern = self._analyze_response_pattern(prev_text, text)
                response_patterns.append(pattern)
        
        context['friendly_indicators'] = friendly_count
        context['aggressive_indicators'] = aggressive_count
        context['response_patterns'] = response_patterns
        
        # Tone consistency: similar tone across messages suggests mutual understanding (banter)
        # Inconsistent tone may indicate conflict or misunderstanding
        if len(conversation) > 2:
            message_tones = [self._calculate_message_sentiment_score(msg.get('message', '')) for msg in conversation]
            if message_tones:
                # Calculate consistency: lower variance = more consistent tone
                # Convert variance to consistency score (1.0 - variance, capped at 1.0)
                tone_variance = self._calculate_variance(message_tones)
                context['tone_consistency'] = 1.0 - min(tone_variance, 1.0)
        
        # Relationship markers
        context['relationship_markers'] = self._extract_relationship_markers(conversation)
        
        return context
    
    def _extract_from_single_message(self, text: str) -> Dict:
        """
        Extract context from a single message (no conversation thread).
        
        For single messages, only basic language indicators can be extracted since
        there's no conversation structure to analyze.
        
        Args:
            text: Single message text string
        
        Returns:
            Dictionary with limited context features (no reciprocity, mutual engagement, etc.)
        """
        text_lower = text.lower()
        
        return {
            'num_messages': 1,
            'participants': ['single_user'],
            'num_participants': 1,
            'text': text,  # Store original text for severe indicator checking
            'friendly_indicators': self._count_friendly_language_indicators(text_lower),
            'aggressive_indicators': self._count_aggressive_language_indicators(text_lower),
            'reciprocity_score': 0.0,  # No reciprocity in single message
            'mutual_engagement': False,
            'tone_consistency': 0.0,
            'message_sequence': []  # Empty for single message
        }
    
    def _count_friendly_language_indicators(self, text: str) -> int:
        """
        Count friendly language indicators in text.
        
        This method identifies positive language patterns that suggest friendly interaction:
        - Emoji characters (😊, 😂, etc.)
        - Casual phrases (lol, haha, bro, etc.)
        - Positive words (love, great, awesome, etc.)
        - Exclamation marks (often indicate excitement, not aggression)
        
        Args:
            text: Text string to analyze (should be lowercase for phrase matching)
        
        Returns:
            Integer count of friendly indicators found in the text
        """
        count = 0
        
        # Emojis
        emoji_count = emoji.emoji_count(text)
        count += emoji_count
        
        # Casual phrases
        for phrase in self.casual_phrases:
            count += text.count(phrase)
        
        # Positive words
        for word in self.positive_words:
            count += text.count(word)
        
        # Exclamation marks (often indicate excitement, not aggression)
        count += text.count('!') * 0.5
        
        return int(count)
    
    def _count_aggressive_language_indicators(self, text: str) -> int:
        """
        Count aggressive language indicators in text.
        
        This method identifies negative language patterns that suggest aggressive interaction:
        - ALL CAPS text (aggressive shouting)
        - Aggressive words/phrases (idiot, shut up, hate you, etc.)
        - Multiple question/exclamation marks (often aggressive)
        
        Note: This method checks for ALL CAPS on the original text (before lowercasing)
        because case-insensitive matching would miss this pattern.
        
        Args:
            text: Text string to analyze (original case preserved for ALL CAPS detection)
        
        Returns:
            Integer count of aggressive indicators found in the text
        """
        count = 0
        
        # Check for ALL CAPS (aggressive shouting) on original text before lowercasing
        # This must be checked separately because [A-Z] with IGNORECASE matches both cases
        all_caps_matches = len(re.findall(r'[A-Z]{3,}', text))
        count += all_caps_matches
        
        # Convert to lowercase for other pattern matching
        text_lower = text.lower()
        
        # Aggressive words/phrases (case-insensitive)
        aggressive_patterns = [
            r'\b(you\s+(are|re)\s+(an?\s+)?(idiot|stupid|dumb|fool|loser|pathetic|worthless))\b',
            r'\b(shut\s+up|fuck\s+off|go\s+to\s+hell|kill\s+yourself|die)\b',
            r'\b(hate|despise|loathe)\s+you\b',
            r'\b(never\s+talk|don\'?t\s+talk|stop\s+talking)\b',
        ]
        
        for pattern in aggressive_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            count += len(matches)
        
        # Multiple question/exclamation marks (often aggressive)
        count += len(re.findall(r'[!?]{2,}', text))
        
        return count
    
    def _analyze_response_pattern(self, prev_text: str, current_text: str) -> str:
        """Analyze the pattern of response (defensive, playful, aggressive, etc.)."""
        prev_lower = prev_text.lower()
        curr_lower = current_text.lower()
        
        # Check if response is defensive
        defensive_phrases = ['why are you', 'what did i do', 'i didn\'t', 'that\'s not true', 
                            'you\'re wrong', 'stop saying']
        if any(phrase in curr_lower for phrase in defensive_phrases):
            return 'defensive'
        
        # Check if response is playful/reciprocal
        playful_indicators = ['lol', 'haha', 'lmao', '😄', '😂', '😆']
        if any(indicator in prev_lower or indicator in curr_lower for indicator in playful_indicators):
            return 'playful'
        
        # Check if response escalates aggression
        prev_aggressive = self._count_aggressive_language_indicators(prev_lower)
        curr_aggressive = self._count_aggressive_language_indicators(curr_lower)
        
        if curr_aggressive > prev_aggressive:
            return 'escalating'
        
        # Check if response de-escalates
        if curr_aggressive < prev_aggressive:
            return 'de-escalating'
        
        # If both have same level (including both 0), return neutral
        return 'neutral'
    
    def _calculate_message_sentiment_score(self, text: str) -> float:
        """
        Calculate message sentiment score based on friendly vs aggressive indicators.
        
        This method computes a sentiment score by comparing the ratio of friendly
        to aggressive language indicators. A score close to 1.0 indicates very
        positive sentiment, while a score close to 0.0 indicates very negative sentiment.
        
        Args:
            text: Message text to analyze
        
        Returns:
            Float sentiment score (0.0 = very negative, 1.0 = very positive, 0.5 = neutral)
        """
        friendly = self._count_friendly_language_indicators(text.lower())
        aggressive = self._count_aggressive_language_indicators(text.lower())
        
        total = friendly + aggressive
        if total == 0:
            return 0.5  # Neutral
        
        return friendly / total
    
    def _calculate_variance(self, values: List[float]) -> float:
        """
        Calculate variance of a list of values.
        
        Variance measures how spread out the values are. Lower variance indicates
        more consistent values (useful for tone consistency analysis).
        
        Args:
            values: List of float values to calculate variance for
        
        Returns:
            Float variance value (0.0 if all values are identical, higher for more spread)
        """
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _extract_relationship_markers(self, conversation: List[Dict]) -> List[str]:
        """
        Extract relationship markers from conversation.
        
        This method identifies indicators of established relationships, such as:
        - Mutual mentions of participant names
        - Friendly relationship terms (friend, buddy, bro, etc.)
        
        These markers suggest that participants know each other, which makes
        banter more likely than real cyberbullying.
        
        Args:
            conversation: List of message dictionaries
        
        Returns:
            List of relationship marker strings (e.g., 'mentions_Alice', 'friendly_term_bro')
        """
        markers = []
        all_text = ' '.join([msg.get('message', '') for msg in conversation]).lower()
        
        # Check for mutual mentions
        participants = list(set([msg.get('user', 'unknown') for msg in conversation]))
        if len(participants) >= 2:
            for user in participants:
                if user != 'unknown' and user.lower() in all_text:
                    markers.append(f'mentions_{user}')
        
        # Check for friendly terms
        friendly_terms = ['friend', 'buddy', 'bro', 'dude', 'mate', 'pal']
        for term in friendly_terms:
            if term in all_text:
                markers.append(f'friendly_term_{term}')
        
        return markers

