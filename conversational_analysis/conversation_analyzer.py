"""
Conversation Analyzer Module

This module provides the main orchestrator for toxic content classification that combines:
- XLM-RoBERTa model inference for initial classification
- Context extraction from conversation threads
- Banter detection to distinguish friendly banter from real cyberbullying
- LLM verification (optional) for additional validation
- Conflict resolution to determine the final classification label

The analyzer processes both single messages and conversation threads, providing
comprehensive analysis with detailed reasoning for each classification decision.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .context_extractor import ContextExtractor
from .banter_detector import BanterDetector


class ConversationAnalyzer:
    """
    Main analyzer that combines XLM-RoBERTa model predictions with conversational analysis.
    
    This class orchestrates the complete analysis pipeline:
    1. Initial classification using fine-tuned XLM-RoBERTa model
    2. Context extraction from conversation threads
    3. Banter detection to identify friendly interactions
    4. LLM verification (if enabled) for additional validation
    5. Conflict resolution to determine final label
    
    The conflict resolution system uses a priority-based approach:
    - Highest priority: Banter detection (overrides to "Normal" if detected)
    - Second priority: Agreement between model and LLM
    - Third priority: Confidence-based resolution when models disagree
    - Fallback: Default to model classification
    
    Attributes:
        LABEL_COLUMNS: List of possible classification labels matching training data
        NUM_LABELS: Total number of classification labels
        MAX_LENGTH: Maximum token length for model input (512 tokens)
        model_path: Path to the trained XLM-RoBERTa model
        device: Computing device ('cuda' or 'cpu')
        tokenizer: HuggingFace tokenizer for XLM-RoBERTa
        model: Fine-tuned XLM-RoBERTa classification model
        context_extractor: Instance for extracting conversation context
        banter_detector: Instance for detecting friendly banter
    """
    
    # Classification labels matching the training data format
    LABEL_COLUMNS = ['Normal', 'Insult', 'Hate Speech', 'Flaming', 'Sexual Harassment']
    NUM_LABELS = len(LABEL_COLUMNS)
    MAX_LENGTH = 512  # Maximum sequence length for tokenization
    
    def __init__(self, model_path: str = './models/xlm-roberta-toxic-classifier', device: Optional[str] = None):
        """
        Initialize the conversation analyzer with model and analysis components.
        
        This method initializes all required components:
        - Loads the fine-tuned XLM-RoBERTa model and tokenizer
        - Sets up context extraction for conversation analysis
        - Initializes banter detection capabilities
        - Configures the computing device (GPU if available, else CPU)
        
        Args:
            model_path: Path to the trained XLM-RoBERTa model directory.
                       Should contain model weights, tokenizer config, and vocab files.
            device: Computing device to use ('cuda' for GPU, 'cpu' for CPU, or None for auto-detection).
                   If None, automatically selects CUDA if available, otherwise CPU.
        
        Raises:
            RuntimeError: If model loading fails (e.g., model files not found, incompatible format).
        """
        self.model_path = model_path
        # Auto-detect device if not specified: prefer GPU for faster inference
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize analysis components
        self.context_extractor = ContextExtractor()
        self.banter_detector = BanterDetector()
        
        # Load and initialize the classification model
        self._load_model()
    
    def _load_model(self):
        """
        Load the fine-tuned XLM-RoBERTa model and tokenizer from disk.
        
        This method handles the complete model loading process:
        1. Loads the tokenizer configuration and vocabulary
        2. Loads the model weights (this may take 2-5 minutes on CPU)
        3. Moves the model to the specified device (GPU/CPU)
        4. Sets the model to evaluation mode for inference
        
        The model is loaded in evaluation mode (no gradient computation) for efficient inference.
        
        Raises:
            RuntimeError: If model files are missing, corrupted, or incompatible.
        """
        print(f"Loading model from {self.model_path}...")
        try:
            print("  Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print("  Tokenizer loaded. Loading model weights (this may take 2-5 minutes on CPU)...")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            print("  Model weights loaded. Moving to device...")
            self.model.to(self.device)
            print("  Setting model to evaluation mode...")
            self.model.eval()
            print(f"[OK] Model loaded successfully on {self.device}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
    
    def analyze(self, 
                text: Optional[str] = None, 
                conversation: Optional[List[Dict]] = None,
                llm_result: Optional[Dict] = None) -> Dict:
        """
        Analyze text or conversation for toxic content classification with banter detection and LLM verification.
        
        Args:
            text: Single message text (required if conversation is None)
            conversation: List of conversation messages with 'user' and 'message' keys
            llm_result: Optional LLM verification result from LLMVerifier
            
        Returns:
            Dictionary with classification results, banter detection, LLM verification, and final label
        """
        # Validate input
        if conversation is not None and len(conversation) == 0:
            raise ValueError("Conversation list cannot be empty")
        
        if not text and not conversation:
            raise ValueError("Either 'text' or 'conversation' must be provided")
        
        # Extract target text: use last message in conversation or single text input
        if conversation:
            target_text = conversation[-1].get('message', '')
            # Fallback: some formats use 'text' key instead of 'message'
            if not target_text:
                target_text = conversation[-1].get('text', '')
        else:
            target_text = text
        
        if not target_text:
            raise ValueError("No text found in input - cannot perform analysis")
        
        # Step 1: Get initial classification from XLM-RoBERTa model
        # This provides the baseline classification before any contextual analysis
        initial_classification = self._classify_text_with_xlm_roberta_model(target_text)
        model_label = initial_classification['label']
        model_confidence = initial_classification['confidence']
        label_probabilities = initial_classification['probabilities']
        
        # Step 2: Extract conversational context
        # Analyzes conversation structure, participant engagement, and language patterns
        context = self.context_extractor.extract_context(target_text, conversation)
        
        # Step 3: Detect friendly banter vs real cyberbullying
        # This is crucial for distinguishing playful interactions from harmful content
        is_banter, banter_reasoning, adjusted_confidence = self.banter_detector.detect_banter(
            context, model_label, model_confidence
        )
        
        # Step 4: Resolve conflicts and determine final label
        # Combines model prediction, banter detection, and LLM verification (if available)
        final_label, final_confidence, conflict_info = self._determine_final_label_with_conflict_resolution(
            model_label=model_label,
            model_confidence=model_confidence,
            is_banter=is_banter,
            llm_result=llm_result
        )
        
        # Step 5: Build comprehensive result dictionary
        result = {
            'classification': {
                'label': model_label,
                'confidence': float(model_confidence),
                'probabilities': label_probabilities
            },
            'conversational_analysis': {
                'is_banter': is_banter,
                'reasoning': banter_reasoning,
                'context_used': context.get('has_conversation', False)
            },
            'context': {
                'num_participants': context.get('num_participants', 1),
                'num_messages': context.get('num_messages', 1),
                'reciprocity_score': float(context.get('reciprocity_score', 0.0)),
                'mutual_engagement': context.get('mutual_engagement', False),
                'friendly_indicators': context.get('friendly_indicators', 0),
                'aggressive_indicators': context.get('aggressive_indicators', 0)
            },
            'final_label': final_label,
            'final_confidence': float(final_confidence),
            'conflict_resolution': conflict_info
        }
        
        # Add LLM verification if provided
        if llm_result:
            result['llm_verification'] = llm_result
        
        return result
    
    def _determine_final_label_with_conflict_resolution(self,
                                                         model_label: str,
                                                         model_confidence: float,
                                                         is_banter: bool,
                                                         llm_result: Optional[Dict] = None) -> Tuple[str, float, Dict]:
        """
        Resolve conflicts between model classification, banter detection, and LLM verification.
        
        This method implements a priority-based conflict resolution system to determine the final
        classification label when multiple analysis components disagree. The system ensures that
        friendly banter is correctly identified and not misclassified as toxic content.
        
        Conflict Resolution Priority Rules (in order):
        1. Banter Detection (Highest Priority): 
           - If banter is detected, always override to "Normal" regardless of model/LLM predictions
           - Rationale: Friendly banter should never be classified as toxic
        
        2. Agreement Check:
           - If model and LLM agree on the label, use that label
           - Rationale: Consensus between models increases confidence
        
        3. Confidence-Based Resolution (when models disagree):
           - Model high confidence (>0.8) + LLM low confidence (<0.6) → Trust model
           - Model low confidence (<0.6) + LLM high confidence (>0.8) → Trust LLM
           - Both medium confidence (0.6-0.8) → Trust model as fallback
           - Both high confidence (>0.8) but disagree → Use whichever has higher confidence
           - Model high + LLM medium → Trust model
           - Model medium + LLM high → Trust LLM
        
        4. Final Fallback:
           - Default to model result if no specific rule applies
           - Rationale: Model is the primary classifier and most reliable baseline
        
        Args:
            model_label: Classification label from the XLM-RoBERTa model
            model_confidence: Confidence score from the model (0.0 to 1.0)
            is_banter: Boolean indicating whether friendly banter was detected
            llm_result: Optional dictionary containing LLM verification results with keys:
                       - 'enabled': Whether LLM verification is active
                       - 'llm_label': Label predicted by LLM
                       - 'confidence' or 'llm_confidence': LLM confidence score
                       - 'agrees': Whether LLM agrees with model classification
            
        Returns:
            Tuple containing:
            - final_label (str): The resolved classification label
            - final_confidence (float): The confidence score for the final label
            - conflict_info (dict): Dictionary with conflict resolution details:
              * 'conflict_detected': Boolean indicating if a conflict was found
              * 'resolution_method': String describing which resolution rule was applied
              * 'reasoning': Human-readable explanation of the resolution decision
        """
        # Initialize conflict resolution metadata
        conflict_info = {
            'conflict_detected': False,
            'resolution_method': 'model_default',
            'reasoning': 'Using model classification as default'
        }
        
        # Priority 1: Banter Detection (Highest Priority - Overrides Everything)
        # If banter is detected, the interaction is friendly and should be classified as Normal
        if is_banter:
            conflict_info['resolution_method'] = 'banter_detection'
            conflict_info['reasoning'] = 'Banter detected - overriding to Normal'
            return 'Normal', model_confidence, conflict_info
        
        # If no LLM verification available, use model result directly
        if not llm_result or not llm_result.get('enabled', False):
            conflict_info['resolution_method'] = 'model_only'
            conflict_info['reasoning'] = 'No LLM verification available - using model result'
            return model_label, model_confidence, conflict_info
        
        # Extract LLM verification results
        # Support both 'confidence' and 'llm_confidence' keys for backward compatibility
        llm_label = llm_result.get('llm_label')
        llm_confidence = llm_result.get('llm_confidence') or llm_result.get('confidence', 0.0)
        llm_agrees = llm_result.get('agrees')
        
        # Priority 2: Agreement Check
        # If both model and LLM agree, use the agreed-upon label
        if llm_agrees is True or (llm_label and llm_label == model_label):
            conflict_info['resolution_method'] = 'agreement'
            conflict_info['reasoning'] = f'Model and LLM agree on label: {model_label}'
            return model_label, model_confidence, conflict_info
        
        # Priority 3: Confidence-Based Resolution (when models disagree)
        # Mark that a conflict was detected
        conflict_info['conflict_detected'] = True
        
        # Delegate to helper method for cleaner code organization
        resolution_result = self._resolve_confidence_based_conflict(
            model_label, model_confidence, llm_label, llm_confidence
        )
        if resolution_result:
            final_label, final_confidence, resolution_info = resolution_result
            conflict_info.update(resolution_info)  # Update with method-specific info
            return final_label, final_confidence, conflict_info
        
        # Final Fallback: Default to model result
        # This should rarely be reached, but provides a safety net
        conflict_info['resolution_method'] = 'model_fallback'
        conflict_info['reasoning'] = (
            f'No specific rule matched - defaulting to model result '
            f'(Model: {model_confidence:.2f}, LLM: {llm_confidence:.2f})'
        )
        return model_label, model_confidence, conflict_info
    
    def _resolve_confidence_based_conflict(self,
                                           model_label: str,
                                           model_confidence: float,
                                           llm_label: Optional[str],
                                           llm_confidence: float) -> Optional[Tuple[str, float, Dict]]:
        """
        Resolve conflicts using confidence-based decision rules.
        
        This helper method implements the confidence-based resolution logic when the model
        and LLM disagree. It evaluates the relative confidence levels and selects the most
        reliable prediction.
        
        Args:
            model_label: Label from the classification model
            model_confidence: Confidence score from model (0.0 to 1.0)
            llm_label: Label from LLM (may be None)
            llm_confidence: Confidence score from LLM (0.0 to 1.0)
        
        Returns:
            Optional tuple of (final_label, final_confidence, conflict_info) if a rule matches,
            None if no rule applies (should fall back to default)
        """
        conflict_info = {}
        
        # Case 1: Model high confidence, LLM low confidence → Trust model
        if model_confidence > 0.8 and llm_confidence < 0.6:
            conflict_info = {
                'conflict_detected': True,
                'resolution_method': 'model_high_confidence',
                'reasoning': (
                    f'Model high confidence ({model_confidence:.2f}) > LLM low confidence '
                    f'({llm_confidence:.2f}) - trusting model'
                )
            }
            return (model_label, model_confidence, conflict_info)
        
        # Case 2: Model low confidence, LLM high confidence → Trust LLM
        if model_confidence < 0.6 and llm_confidence > 0.8:
            conflict_info = {
                'conflict_detected': True,
                'resolution_method': 'llm_high_confidence',
                'reasoning': (
                    f'LLM high confidence ({llm_confidence:.2f}) > Model low confidence '
                    f'({model_confidence:.2f}) - trusting LLM'
                )
            }
            if llm_label:
                return (llm_label, llm_confidence, conflict_info)
            else:
                # If LLM label is None, fall back to model
                conflict_info['reasoning'] += ' (but LLM label unavailable, using model)'
                return (model_label, model_confidence, conflict_info)
        
        # Case 3: Both medium confidence (0.6-0.8) → Trust model as fallback
        if 0.6 <= model_confidence <= 0.8 and 0.6 <= llm_confidence <= 0.8:
            conflict_info = {
                'conflict_detected': True,
                'resolution_method': 'model_fallback_medium',
                'reasoning': (
                    f'Both medium confidence (Model: {model_confidence:.2f}, '
                    f'LLM: {llm_confidence:.2f}) - trusting model as fallback'
                )
            }
            return (model_label, model_confidence, conflict_info)
        
        # Case 4: Both high confidence (>0.8) but disagree → Use higher confidence
        if model_confidence > 0.8 and llm_confidence > 0.8:
            if model_confidence >= llm_confidence:
                conflict_info = {
                    'conflict_detected': True,
                    'resolution_method': 'higher_confidence_model',
                    'reasoning': (
                        f'Both high confidence, model higher (Model: {model_confidence:.2f} >= '
                        f'LLM: {llm_confidence:.2f}) - trusting model'
                    )
                }
                return (model_label, model_confidence, conflict_info)
            else:
                conflict_info = {
                    'conflict_detected': True,
                    'resolution_method': 'higher_confidence_llm',
                    'reasoning': (
                        f'Both high confidence, LLM higher (LLM: {llm_confidence:.2f} > '
                        f'Model: {model_confidence:.2f}) - trusting LLM'
                    )
                }
                if llm_label:
                    return (llm_label, llm_confidence, conflict_info)
                else:
                    conflict_info['reasoning'] += ' (but LLM label unavailable, using model)'
                    return (model_label, model_confidence, conflict_info)
        
        # Case 5: Model high, LLM medium → Trust model
        if model_confidence > 0.8 and 0.6 <= llm_confidence <= 0.8:
            conflict_info = {
                'conflict_detected': True,
                'resolution_method': 'model_high_confidence',
                'reasoning': (
                    f'Model high confidence ({model_confidence:.2f}) > LLM medium confidence '
                    f'({llm_confidence:.2f}) - trusting model'
                )
            }
            return (model_label, model_confidence, conflict_info)
        
        # Case 6: Model medium, LLM high → Trust LLM
        if 0.6 <= model_confidence <= 0.8 and llm_confidence > 0.8:
            conflict_info = {
                'conflict_detected': True,
                'resolution_method': 'llm_high_confidence',
                'reasoning': (
                    f'LLM high confidence ({llm_confidence:.2f}) > Model medium confidence '
                    f'({model_confidence:.2f}) - trusting LLM'
                )
            }
            if llm_label:
                return (llm_label, llm_confidence, conflict_info)
            else:
                conflict_info['reasoning'] += ' (but LLM label unavailable, using model)'
                return (model_label, model_confidence, conflict_info)
        
        # No rule matched
        return None
    
    def _classify_text_with_xlm_roberta_model(self, text: str) -> Dict:
        """
        Classify text using the fine-tuned XLM-RoBERTa model.
        
        This method performs the core classification task:
        1. Tokenizes the input text
        2. Runs inference through the XLM-RoBERTa model
        3. Converts logits to probabilities using sigmoid activation
        4. Determines the predicted label based on probability thresholds
        5. Handles multi-label classification scenarios
        
        The model uses a multi-label classification approach where multiple toxic
        categories can be detected simultaneously. However, the final label is
        selected based on the highest probability, with special handling for cases
        where toxic labels exceed the threshold.
        
        Args:
            text: Input text string to classify. Will be truncated to MAX_LENGTH tokens.
        
        Returns:
            Dictionary containing:
            - 'label' (str): The predicted classification label
            - 'confidence' (float): Confidence score for the predicted label (0.0 to 1.0)
            - 'probabilities' (dict): Dictionary mapping each label to its probability score
        """
        # Tokenize input text: convert to model input format
        # Truncation ensures text fits within model's maximum sequence length
        tokenized_inputs = self.tokenizer(
            text,
            return_tensors='pt',  # Return PyTorch tensors
            truncation=True,  # Truncate if text exceeds MAX_LENGTH
            max_length=self.MAX_LENGTH,
            padding=True  # Pad shorter sequences to same length
        )
        # Move inputs to the appropriate device (GPU/CPU)
        tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}
        
        # Run model inference (no gradient computation needed for prediction)
        with torch.no_grad():
            model_outputs = self.model(**tokenized_inputs)
            logits = model_outputs.logits
        
        # Convert logits to probabilities using sigmoid activation
        # Sigmoid is appropriate for multi-label classification (each label is independent)
        label_probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Create probability dictionary for all labels
        probabilities_dict = {
            label: float(prob) for label, prob in zip(self.LABEL_COLUMNS, label_probabilities)
        }
        
        # Get the label with highest probability (initial prediction)
        predicted_label_index = np.argmax(label_probabilities)
        predicted_label = self.LABEL_COLUMNS[predicted_label_index]
        predicted_confidence = float(label_probabilities[predicted_label_index])
        
        # Multi-label handling: Check if any toxic label exceeds threshold
        # This handles cases where the model detects toxicity even if Normal has highest prob
        TOXICITY_THRESHOLD = 0.5
        toxic_labels = [label for label in self.LABEL_COLUMNS if label != 'Normal']
        toxic_label_probabilities = [
            label_probabilities[self.LABEL_COLUMNS.index(label)] for label in toxic_labels
        ]
        max_toxic_probability = max(toxic_label_probabilities) if toxic_label_probabilities else 0.0
        
        # If a toxic label has high probability (>threshold) and exceeds Normal probability,
        # use the toxic label instead (more specific classification)
        normal_probability = label_probabilities[0]  # First label is always 'Normal'
        if max_toxic_probability > TOXICITY_THRESHOLD and max_toxic_probability > normal_probability:
            # Find which toxic label has the highest probability
            toxic_label_index = np.argmax(toxic_label_probabilities)
            predicted_label = toxic_labels[toxic_label_index]
            predicted_confidence = max_toxic_probability
        
        return {
            'label': predicted_label,
            'confidence': predicted_confidence,
            'probabilities': probabilities_dict
        }
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """
        Analyze multiple texts in batch processing mode.
        
        This method processes a list of texts sequentially, calling the main analyze()
        method for each text. Errors for individual texts are caught and reported
        in the results rather than stopping the entire batch.
        
        Args:
            texts: List of text strings to analyze. Each text is processed independently.
        
        Returns:
            List of analysis result dictionaries, one per input text. Each result follows
            the same format as the analyze() method. If an error occurs for a specific
            text, the result will contain an 'error' key with the error message and a
            'text' key with the original text.
        
        Example:
            >>> analyzer = ConversationAnalyzer()
            >>> texts = ["Hello world", "You're an idiot!"]
            >>> results = analyzer.batch_analyze(texts)
            >>> len(results)  # Returns 2 results
            2
        """
        batch_results = []
        for text in texts:
            try:
                analysis_result = self.analyze(text=text)
                batch_results.append(analysis_result)
            except Exception as e:
                # Include error information in result rather than failing entire batch
                batch_results.append({
                    'error': str(e),
                    'text': text
                })
        return batch_results

