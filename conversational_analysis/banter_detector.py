"""
Banter Detector Module

This module implements sophisticated logic for distinguishing friendly banter from
real cyberbullying. The detector analyzes conversation patterns, participant engagement,
language indicators, and response patterns to determine if an interaction is playful
or harmful.

The detection system uses a scoring-based approach where multiple rules contribute
to a banter evidence score. If the score exceeds a threshold (60%), the interaction
is classified as friendly banter and should be labeled as "Normal" rather than toxic.
"""

from typing import Dict, Tuple, Optional, List
import re


class BanterDetector:
    """
    Detects friendly banter vs real cyberbullying using comprehensive logic rules.
    
    This class implements a multi-rule scoring system that evaluates various aspects
    of a conversation to determine if it represents friendly banter or real cyberbullying.
    The detector considers:
    - Reciprocity: Are both parties actively engaging?
    - Mutual engagement: Do participants contribute equally?
    - Language indicators: Ratio of friendly vs aggressive language
    - Response patterns: Playful, defensive, or escalating responses
    - Tone consistency: Similar tone across messages suggests banter
    - One-sided aggression: Unbalanced aggression indicates real bullying
    - Relationship markers: Friendly terms suggest established relationships
    
    Attributes:
        reciprocity_threshold: Minimum reciprocity score (0.0-1.0) to consider banter.
                              Higher values require more balanced participation.
        friendly_ratio_threshold: Minimum ratio of friendly to aggressive indicators
                                 (0.0-1.0) to consider banter. 0.4 means at least
                                 40% of indicators must be friendly.
        mutual_engagement_threshold: Minimum number of messages from each participant
                                     to establish mutual engagement (default: 2).
        tone_consistency_threshold: Minimum tone consistency score (0.0-1.0) across
                                    messages. Higher values indicate more consistent tone.
        severe_indicators: List of regex patterns for severe cyberbullying that
                          override banter detection (e.g., threats, self-harm references).
        one_sided_patterns: List of regex patterns indicating one-sided aggression
                           (e.g., "you always", "why are you always").
    """
    
    def __init__(self):
        # Decision thresholds for banter detection
        # These values were tuned based on analysis of banter vs bullying conversations
        self.reciprocity_threshold = 0.3  # Minimum reciprocity for banter (30% balanced participation)
        self.friendly_ratio_threshold = 0.4  # Minimum friendly/aggressive ratio (40% friendly)
        self.mutual_engagement_threshold = 2  # Minimum messages from each participant
        self.tone_consistency_threshold = 0.6  # Minimum tone consistency (60% consistent)
        
        # Severe cyberbullying indicators that override banter detection
        # These patterns indicate serious harm and cannot be considered banter
        self.severe_indicators = [
            r'\b(kill\s+yourself|kys|die|suicide)\b',  # Self-harm references
            r'\b(rape|sexual\s+assault|molest)\b',  # Sexual violence
            r'\b(threat|threaten|harm|hurt|violence)\b',  # Threats of harm
            r'\b(never\s+talk|don\'?t\s+ever|stay\s+away)\b',  # Exclusion commands
            r'\b(worthless|pathetic|disgusting|hate\s+you)\b',  # Severe dehumanization
        ]
        
        # One-sided aggression patterns indicating unbalanced interaction
        # These suggest one person is attacking while the other is not reciprocating
        self.one_sided_patterns = [
            r'\b(you\s+always|you\s+never|you\'?re\s+always)\b',  # Accusatory generalizations
            r'\b(why\s+do\s+you|why\s+are\s+you\s+always)\b',  # Hostile questioning
            r'\b(stop\s+being|you\s+need\s+to\s+stop)\b',  # Controlling commands
        ]
    
    def detect_banter(self, context: Dict, initial_label: str, initial_confidence: float) -> Tuple[bool, str, float]:
        """
        Detect if the conversation is friendly banter or real cyberbullying.
        
        This is the main entry point for banter detection. It routes to appropriate
        analysis methods based on the input type (single message vs conversation thread).
        
        Args:
            context: Context dictionary from ContextExtractor containing conversation
                    features, participant info, and language indicators.
            initial_label: Initial classification label from the XLM-RoBERTa model.
                          If this is 'Normal', banter detection is skipped.
            initial_confidence: Initial confidence score from the model (0.0 to 1.0).
        
        Returns:
            Tuple containing:
            - is_banter (bool): True if friendly banter is detected, False otherwise
            - reasoning (str): Human-readable explanation of the banter detection decision
            - confidence (float): Adjusted confidence score after banter analysis
        """
        # If model already classified as Normal, no banter override needed
        # (banter detection is only relevant when model predicts toxic content)
        if initial_label == 'Normal':
            return False, "Initial classification is Normal, no banter override needed", initial_confidence
        
        # Route to appropriate analysis method based on context availability
        if not context.get('has_conversation', False):
            # Single message: use conservative approach (requires strong evidence)
            return self._analyze_single_message_for_banter(context, initial_label, initial_confidence)
        
        # Conversation thread: use comprehensive multi-rule analysis
        return self._analyze_conversation_for_banter(context, initial_label, initial_confidence)
    
    def _analyze_conversation_for_banter(self, context: Dict, initial_label: str, confidence: float) -> Tuple[bool, str, float]:
        """
        Analyze conversation thread for banter detection using multi-rule scoring system.
        
        This method implements a comprehensive scoring system that evaluates 8 different
        rules to determine if a conversation represents friendly banter. Each rule
        contributes to a banter evidence score, and if the final score exceeds 60%,
        the conversation is classified as banter.
        
        Args:
            context: Context dictionary with conversation features and indicators
            initial_label: Initial model classification label
            confidence: Initial model confidence score
        
        Returns:
            Tuple of (is_banter, reasoning, adjusted_confidence)
        """
        reasoning_parts = []
        banter_evidence_score = 0.0
        maximum_possible_score = 0.0
        
        # Rule 1: Check for severe cyberbullying indicators (OVERRIDE - always real bullying)
        # This rule has highest priority and overrides all other banter indicators
        severe_detected = self._check_severe_indicators(context)
        if severe_detected:
            return False, "Severe cyberbullying indicators detected - cannot be banter", confidence
        
        # Rule 2: Reciprocity Analysis
        # High reciprocity indicates both parties are actively engaging (characteristic of banter)
        banter_evidence_score, maximum_possible_score, reasoning = self._evaluate_reciprocity_rule(
            context, banter_evidence_score, maximum_possible_score, reasoning_parts
        )
        reasoning_parts = reasoning
        
        # Rule 3: Mutual Engagement
        # Both participants should contribute multiple messages for mutual banter
        banter_evidence_score, maximum_possible_score, reasoning = self._evaluate_mutual_engagement_rule(
            context, banter_evidence_score, maximum_possible_score, reasoning_parts
        )
        reasoning_parts = reasoning
        
        # Rule 4: Friendly vs Aggressive Indicators Ratio
        # Banter should have more friendly language than aggressive language
        banter_evidence_score, maximum_possible_score, reasoning = self._evaluate_friendly_ratio_rule(
            context, banter_evidence_score, maximum_possible_score, reasoning_parts
        )
        reasoning_parts = reasoning
        
        # Rule 5: Response Patterns Analysis
        # Playful responses suggest banter, defensive/escalating responses suggest real conflict
        banter_evidence_score, maximum_possible_score, reasoning = self._evaluate_response_patterns_rule(
            context, banter_evidence_score, maximum_possible_score, reasoning_parts
        )
        reasoning_parts = reasoning
        
        # Rule 6: Tone Consistency
        # Consistent tone across messages suggests mutual understanding (banter)
        banter_evidence_score, maximum_possible_score, reasoning = self._evaluate_tone_consistency_rule(
            context, banter_evidence_score, maximum_possible_score, reasoning_parts
        )
        reasoning_parts = reasoning
        
        # Rule 7: One-sided Aggression Check
        # Unbalanced aggression indicates real bullying, not mutual banter
        banter_evidence_score, maximum_possible_score, reasoning = self._evaluate_one_sided_aggression_rule(
            context, banter_evidence_score, maximum_possible_score, reasoning_parts
        )
        reasoning_parts = reasoning
        
        # Rule 8: Relationship Markers
        # Friendly relationship terms suggest established relationships (more likely banter)
        banter_evidence_score, maximum_possible_score, reasoning = self._evaluate_relationship_markers_rule(
            context, banter_evidence_score, maximum_possible_score, reasoning_parts
        )
        reasoning_parts = reasoning
        
        # Calculate final banter probability from evidence score
        if maximum_possible_score > 0:
            banter_probability = max(0.0, min(1.0, banter_evidence_score / maximum_possible_score))
        else:
            banter_probability = 0.0
        
        # Decision threshold: need at least 60% banter score to classify as banter
        BANTER_THRESHOLD = 0.6
        is_banter = banter_probability >= BANTER_THRESHOLD
        
        # Combine all reasoning parts into final explanation
        reasoning = "; ".join(reasoning_parts)
        reasoning += f" | Banter score: {banter_probability:.2f} (threshold: {BANTER_THRESHOLD})"
        
        # Adjust confidence based on banter detection
        if is_banter:
            # If banter detected, increase confidence in Normal label
            # Formula: increase confidence by 30% of remaining gap to 1.0
            adjusted_confidence = min(1.0, confidence + (1 - confidence) * 0.3)
        else:
            # If not banter, maintain original confidence in toxic label
            adjusted_confidence = confidence
        
        return is_banter, reasoning, adjusted_confidence
    
    def _evaluate_reciprocity_rule(self, context: Dict, banter_score: float, max_score: float, reasoning_parts: List[str]) -> Tuple[float, float, List[str]]:
        """
        Evaluate Rule 2: Reciprocity Analysis.
        
        High reciprocity (balanced participation) indicates mutual engagement characteristic of banter.
        """
        reciprocity = context.get('reciprocity_score', 0.0)
        max_score += 1.0
        if reciprocity >= self.reciprocity_threshold:
            banter_score += 1.0
            reasoning_parts.append(f"High reciprocity ({reciprocity:.2f}) - both parties engaged")
        else:
            reasoning_parts.append(f"Low reciprocity ({reciprocity:.2f}) - one-sided interaction")
        return banter_score, max_score, reasoning_parts
    
    def _evaluate_mutual_engagement_rule(self, context: Dict, banter_score: float, max_score: float, reasoning_parts: List[str]) -> Tuple[float, float, List[str]]:
        """
        Evaluate Rule 3: Mutual Engagement.
        
        Both participants should contribute multiple messages for mutual banter to occur.
        """
        mutual_engagement = context.get('mutual_engagement', False)
        num_participants = context.get('num_participants', 1)
        num_messages = context.get('num_messages', 1)
        max_score += 1.0
        if mutual_engagement and num_participants >= 2 and num_messages >= self.mutual_engagement_threshold:
            banter_score += 1.0
            reasoning_parts.append(f"Mutual engagement: {num_participants} participants, {num_messages} messages")
        else:
            reasoning_parts.append(f"Limited mutual engagement: {num_participants} participants, {num_messages} messages")
        return banter_score, max_score, reasoning_parts
    
    def _evaluate_friendly_ratio_rule(self, context: Dict, banter_score: float, max_score: float, reasoning_parts: List[str]) -> Tuple[float, float, List[str]]:
        """
        Evaluate Rule 4: Friendly vs Aggressive Indicators Ratio.
        
        Banter should have more friendly language indicators than aggressive ones.
        """
        friendly_count = context.get('friendly_indicators', 0)
        aggressive_count = context.get('aggressive_indicators', 0)
        total_indicators = friendly_count + aggressive_count
        max_score += 1.0
        if total_indicators > 0:
            friendly_ratio = friendly_count / total_indicators
            if friendly_ratio >= self.friendly_ratio_threshold:
                banter_score += 1.0
                reasoning_parts.append(f"High friendly indicators ratio ({friendly_ratio:.2f})")
            else:
                reasoning_parts.append(f"Low friendly indicators ratio ({friendly_ratio:.2f})")
        else:
            reasoning_parts.append("No clear friendly/aggressive indicators")
        return banter_score, max_score, reasoning_parts
    
    def _evaluate_response_patterns_rule(self, context: Dict, banter_score: float, max_score: float, reasoning_parts: List[str]) -> Tuple[float, float, List[str]]:
        """
        Evaluate Rule 5: Response Patterns Analysis.
        
        Playful responses suggest banter, while defensive or escalating responses suggest real conflict.
        """
        response_patterns = context.get('response_patterns', [])
        max_score += 1.0
        if response_patterns:
            playful_count = sum(1 for p in response_patterns if p == 'playful')
            defensive_count = sum(1 for p in response_patterns if p == 'defensive')
            escalating_count = sum(1 for p in response_patterns if p == 'escalating')
            
            if playful_count > 0 and escalating_count == 0:
                banter_score += 1.0
                reasoning_parts.append(f"Playful response patterns detected ({playful_count} playful)")
            elif defensive_count > 0:
                banter_score -= 0.5  # Penalize defensive responses (suggests real conflict)
                reasoning_parts.append(f"Defensive responses detected - suggests real conflict")
            elif escalating_count > 0:
                banter_score -= 1.0  # Strong penalty for escalation (definitely not banter)
                reasoning_parts.append(f"Aggressive escalation detected - not banter")
        else:
            reasoning_parts.append("No clear response patterns")
        return banter_score, max_score, reasoning_parts
    
    def _evaluate_tone_consistency_rule(self, context: Dict, banter_score: float, max_score: float, reasoning_parts: List[str]) -> Tuple[float, float, List[str]]:
        """
        Evaluate Rule 6: Tone Consistency.
        
        Consistent tone across messages suggests mutual understanding and banter.
        """
        tone_consistency = context.get('tone_consistency', 0.0)
        max_score += 1.0
        if tone_consistency >= self.tone_consistency_threshold:
            banter_score += 1.0
            reasoning_parts.append(f"Consistent tone across conversation ({tone_consistency:.2f})")
        else:
            reasoning_parts.append(f"Inconsistent tone ({tone_consistency:.2f}) - may indicate conflict")
        return banter_score, max_score, reasoning_parts
    
    def _evaluate_one_sided_aggression_rule(self, context: Dict, banter_score: float, max_score: float, reasoning_parts: List[str]) -> Tuple[float, float, List[str]]:
        """
        Evaluate Rule 7: One-sided Aggression Check.
        
        Unbalanced aggression indicates real bullying, not mutual banter.
        """
        is_one_sided = self._check_one_sided_aggression(context)
        max_score += 1.0
        if is_one_sided:
            banter_score -= 1.0  # Strong penalty for one-sided aggression
            reasoning_parts.append("One-sided aggression pattern detected")
        else:
            banter_score += 0.5  # Reward for balanced interaction
            reasoning_parts.append("No one-sided aggression pattern")
        return banter_score, max_score, reasoning_parts
    
    def _evaluate_relationship_markers_rule(self, context: Dict, banter_score: float, max_score: float, reasoning_parts: List[str]) -> Tuple[float, float, List[str]]:
        """
        Evaluate Rule 8: Relationship Markers.
        
        Friendly relationship terms suggest established relationships (more likely banter).
        """
        relationship_markers = context.get('relationship_markers', [])
        max_score += 0.5
        if relationship_markers:
            friendly_markers = [m for m in relationship_markers if 'friendly' in m]
            if friendly_markers:
                banter_score += 0.5
                reasoning_parts.append(f"Friendly relationship markers found: {len(friendly_markers)}")
        return banter_score, max_score, reasoning_parts
    
    def _analyze_single_message_for_banter(self, context: Dict, initial_label: str, confidence: float) -> Tuple[bool, str, float]:
        """
        Analyze single message for banter detection (no conversation context).
        
        For single messages without conversation context, we use a conservative approach
        because banter is typically identified through interaction patterns. A single
        message requires very strong friendly indicators to be classified as banter.
        
        Args:
            context: Context dictionary (may contain single message text)
            initial_label: Initial model classification label
            confidence: Initial model confidence score
        
        Returns:
            Tuple of (is_banter, reasoning, adjusted_confidence)
        """
        # For single messages, be conservative - default to NOT banter
        # unless there are strong friendly indicators (at least 3 friendly indicators, no aggression)
        
        friendly_count = context.get('friendly_indicators', 0)
        aggressive_count = context.get('aggressive_indicators', 0)
        
        # Check for severe indicators
        severe_detected = self._check_severe_indicators(context)
        if severe_detected:
            return False, "Severe cyberbullying indicators detected in single message", confidence
        
        # If friendly indicators significantly outweigh aggressive, might be banter
        if friendly_count > 0 and aggressive_count == 0 and friendly_count >= 3:
            return True, f"Single message with strong friendly indicators ({friendly_count}) and no aggression", confidence * 0.7
        
        # Default: not enough context to determine banter
        return False, "Single message without conversation context - insufficient information for banter detection", confidence
    
    def _check_severe_indicators(self, context: Dict) -> bool:
        """
        Check for severe cyberbullying indicators that override banter detection.
        
        This method searches for patterns indicating severe harm (threats, self-harm
        references, sexual violence, etc.). If any severe indicator is found, the
        interaction cannot be considered banter regardless of other factors.
        
        Args:
            context: Context dictionary containing message sequence or text
        
        Returns:
            True if severe indicators are detected, False otherwise
        """
        message_sequence = context.get('message_sequence', [])
        
        # Get all text from conversation or single message
        if message_sequence:
            all_text = ' '.join([msg.get('message', '') for msg in message_sequence]).lower()
        else:
            # For single message, check if there's a text field in context
            # This is a fallback - ideally text should be in message_sequence
            all_text = context.get('text', '').lower()
        
        if not all_text:
            return False
        
        for pattern in self.severe_indicators:
            if re.search(pattern, all_text, re.IGNORECASE):
                return True
        
        return False
    
    def _check_one_sided_aggression(self, context: Dict) -> bool:
        """
        Check if aggression is one-sided (not mutual banter).
        
        This method analyzes the distribution of aggressive patterns across participants.
        If one participant shows significantly more aggression than others (ratio < 0.2),
        it indicates one-sided bullying rather than mutual banter.
        
        Args:
            context: Context dictionary containing message sequence and participants
        
        Returns:
            True if one-sided aggression is detected, False otherwise
        """
        message_sequence = context.get('message_sequence', [])
        participants = context.get('participants', [])
        
        if len(participants) < 2 or len(message_sequence) < 2:
            return False
        
        # Count aggressive indicators per participant
        participant_aggression = {}
        for msg in message_sequence:
            user = msg.get('user', 'unknown')
            text = msg.get('message', '').lower()
            
            # Count aggressive patterns
            aggressive_count = 0
            for pattern in self.one_sided_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    aggressive_count += 1
            
            if user not in participant_aggression:
                participant_aggression[user] = 0
            participant_aggression[user] += aggressive_count
        
        # If one participant has significantly more aggression, it's one-sided
        if len(participant_aggression) >= 2:
            aggression_values = list(participant_aggression.values())
            if max(aggression_values) > 0:
                ratio = min(aggression_values) / max(aggression_values) if max(aggression_values) > 0 else 0
                # If ratio is very low (< 0.2), it's one-sided
                if ratio < 0.2:
                    return True
        
        return False

