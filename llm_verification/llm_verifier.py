"""
LLM Verifier Module

This module provides LLM-based verification of toxic content classifications using
open-source language models via llama.cpp. The verifier acts as a second opinion,
helping to catch false positives and validate model predictions through natural
language understanding.

The verifier uses a prompt-based approach where the LLM analyzes the text/conversation
and provides its own classification along with reasoning. This is then compared with
the model's prediction to resolve conflicts or increase confidence.
"""

import os
from typing import Dict, List, Optional, Tuple
import json

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not installed. LLM verification will be disabled.")


class LLMVerifier:
    """
    Verifies classification results using an open-source LLM via llama.cpp.
    
    This class provides an optional verification layer that uses a large language
    model to validate or challenge the primary classification model's predictions.
    The LLM analyzes the text/conversation and provides its own classification,
    which is then used in conflict resolution to determine the final label.
    
    The verifier is designed to be optional - if no LLM model is provided or if
    llama-cpp-python is not installed, the system continues without LLM verification.
    
    Attributes:
        model_path: Path to the GGUF model file (e.g., llama-2-7b.gguf)
        n_ctx: Context window size for the LLM
        n_threads: Number of threads for LLM inference (None for auto)
        verbose: Whether to enable verbose LLM output
        model: Llama model instance (None if not loaded)
        enabled: Boolean indicating if LLM verification is active
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 n_ctx: int = 2048,
                 n_threads: Optional[int] = None,
                 verbose: bool = False):
        """
        Initialize LLM verifier.
        
        Args:
            model_path: Path to GGUF model file (e.g., llama-2-7b.gguf)
                       If None, will try to use a default or disable verification
            n_ctx: Context window size
            n_threads: Number of threads (None for auto)
            verbose: Enable verbose output
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.verbose = verbose
        self.model = None
        self.enabled = False
        
        if not LLAMA_CPP_AVAILABLE:
            print("llama-cpp-python not available. LLM verification disabled.")
            return
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    verbose=verbose
                )
                self.enabled = True
                print(f"LLM verifier initialized with model: {model_path}")
            except Exception as e:
                print(f"Failed to load LLM model: {e}")
                print("LLM verification will be disabled.")
        else:
            print("No LLM model path provided or model not found.")
            print("LLM verification will be disabled.")
            print("To enable, download a GGUF model (e.g., from Hugging Face) and provide the path.")
    
    def verify(self, 
               text: str,
               conversation: Optional[List[Dict]] = None,
               model_label: str = "Unknown",
               model_confidence: float = 0.0) -> Dict:
        """
        Verify classification using LLM analysis.
        
        This method sends the text/conversation to the LLM along with the model's
        classification, asking the LLM to provide its own analysis. The LLM response
        is then parsed to determine if it agrees with the model and what label it
        would assign.
        
        Args:
            text: Text string that was classified by the model
            conversation: Optional list of conversation messages for context
            model_label: Classification label from the XLM-RoBERTa model
            model_confidence: Confidence score from the model (0.0 to 1.0)
        
        Returns:
            Dictionary containing verification results:
            - 'enabled': Boolean indicating if LLM verification is active
            - 'agrees': Boolean indicating if LLM agrees with model (None if unavailable)
            - 'llm_label': Label predicted by LLM (None if unavailable)
            - 'llm_reasoning': LLM's explanation of its analysis
            - 'confidence': LLM's confidence score (0.0 to 1.0)
        """
        if not self.enabled or not self.model:
            return {
                'enabled': False,
                'agrees': None,
                'llm_label': None,
                'llm_reasoning': 'LLM verification not available',
                'confidence': 0.0
            }
        
        # Construct prompt for LLM verification
        verification_prompt = self._construct_llm_verification_prompt(text, conversation, model_label)
        
        # Generate LLM response
        try:
            llm_response = self.model(
                verification_prompt,
                max_tokens=200,  # Limit response length for efficiency
                temperature=0.3,  # Lower temperature for more consistent, deterministic responses
                stop=["\n\n", "Human:", "User:"],  # Stop sequences to end generation
                echo=False  # Don't echo the prompt in the response
            )
            
            # Extract generated text from response
            llm_generated_text = llm_response['choices'][0]['text'].strip()
            
            # Parse and validate LLM response
            verification_result = self._parse_and_validate_llm_response(llm_generated_text, model_label)
            
            return {
                'enabled': True,
                'agrees': verification_result['agrees'],
                'llm_label': verification_result['llm_label'],
                'llm_reasoning': verification_result['reasoning'],
                'confidence': verification_result.get('confidence', 0.5)
            }
        except Exception as e:
            return {
                'enabled': True,
                'agrees': None,
                'llm_label': None,
                'llm_reasoning': f'Error during LLM verification: {str(e)}',
                'confidence': 0.0
            }
    
    def _construct_llm_verification_prompt(self, 
                                           text: str,
                                           conversation: Optional[List[Dict]],
                                           model_label: str) -> str:
        """
        Construct prompt for LLM verification.
        
        This method builds a detailed prompt that instructs the LLM to analyze
        the text/conversation and determine if it represents friendly banter or
        real cyberbullying. The prompt includes guidelines and asks for structured
        JSON output.
        
        Args:
            text: Text string to analyze
            conversation: Optional conversation thread
            model_label: Label from the classification model (for LLM reference)
        
        Returns:
            String containing the complete prompt for the LLM
        """
        prompt = """You are an expert at analyzing online conversations to distinguish between friendly banter and real cyberbullying.

Task: Determine if the given text/conversation is:
1. Friendly banter between friends (should be labeled as "Normal")
2. Real cyberbullying (should be labeled as one of: "Insult", "Hate Speech", "Flaming", "Sexual Harassment")

Guidelines:
- Friendly banter: Both parties engage playfully, use casual language, emojis, mutual jokes
- Real cyberbullying: One-sided aggression, threats, severe insults, harassment, or one person being defensive

"""
        
        if conversation and len(conversation) > 1:
            prompt += "Conversation thread:\n"
            for i, msg in enumerate(conversation):
                user = msg.get('user', f'User{i+1}')
                message = msg.get('message', '')
                prompt += f"{user}: {message}\n"
            prompt += "\n"
        else:
            prompt += f"Text to analyze: {text}\n\n"
        
        prompt += f"""The classification model labeled this as: "{model_label}"

Please analyze and respond in the following JSON format:
{{
    "label": "Normal" or "Insult" or "Hate Speech" or "Flaming" or "Sexual Harassment",
    "is_banter": true or false,
    "reasoning": "brief explanation of your analysis",
    "confidence": 0.0 to 1.0
}}

Your analysis:"""
        
        return prompt
    
    def _parse_and_validate_llm_response(self, response: str, model_label: str) -> Dict:
        """
        Parse LLM response and determine if it agrees with model classification.
        
        This method attempts to extract structured JSON from the LLM response,
        falling back to simple text parsing if JSON parsing fails. It then
        determines whether the LLM agrees with the model's classification.
        
        Agreement logic:
        - If model says "Normal" and LLM says "Normal" → Agreement
        - If model says toxic and LLM says "Normal" (with is_banter=True) → Agreement
        - If model says toxic and LLM says same toxic label → Agreement
        - Otherwise → Disagreement
        
        Args:
            response: Raw text response from the LLM
            model_label: Label from the classification model for comparison
        
        Returns:
            Dictionary containing:
            - 'agrees': Boolean indicating if LLM agrees with model
            - 'llm_label': Label predicted by LLM
            - 'reasoning': LLM's explanation
            - 'confidence': LLM's confidence score
            - 'is_banter': Boolean indicating if LLM detected banter
        """
        # Try to extract JSON from response (preferred format)
        try:
            # Look for JSON object in the response (may be embedded in text)
            json_start_index = response.find('{')
            json_end_index = response.rfind('}') + 1
            
            if json_start_index >= 0 and json_end_index > json_start_index:
                json_string = response[json_start_index:json_end_index]
                parsed_json_response = json.loads(json_string)
                
                # Extract fields from parsed JSON
                llm_label = parsed_json_response.get('label', 'Unknown')
                is_banter = parsed_json_response.get('is_banter', False)
                reasoning = parsed_json_response.get('reasoning', 'No reasoning provided')
                confidence = parsed_json_response.get('confidence', 0.5)
                
                # Determine if LLM agrees with model classification
                # Agreement rules:
                # 1. Model says "Normal" and LLM says "Normal" → Agreement
                # 2. Model says toxic but LLM says "Normal" (banter) → Agreement (both recognize it's not real bullying)
                # 3. Model says toxic and LLM says same toxic label → Agreement
                if model_label == 'Normal':
                    # Model classified as Normal - LLM should also say Normal
                    agrees = (llm_label == 'Normal')
                else:
                    # Model classified as toxic
                    if is_banter:
                        # LLM thinks it's banter, so should be Normal (agrees with banter override)
                        agrees = (llm_label == 'Normal')
                    else:
                        # LLM thinks it's real bullying - check if labels match
                        agrees = (llm_label == model_label or llm_label != 'Normal')
                
                return {
                    'agrees': agrees,
                    'llm_label': llm_label,
                    'reasoning': reasoning,
                    'confidence': confidence,
                    'is_banter': is_banter
                }
        except (json.JSONDecodeError, KeyError):
            # JSON parsing failed - fall back to simple text parsing
            pass
        
        # Fallback: simple text parsing when JSON extraction fails
        # This handles cases where LLM doesn't return properly formatted JSON
        response_lowercase = response.lower()
        
        # Try to extract label by searching for label names in response
        llm_label = 'Unknown'
        valid_labels = ['Normal', 'Insult', 'Hate Speech', 'Flaming', 'Sexual Harassment']
        for label in valid_labels:
            if label.lower() in response_lowercase:
                llm_label = label
                break
        
        # Check for banter indicators in the response text
        banter_keywords = ['banter', 'friendly', 'playful', 'joking']
        is_banter = any(keyword in response_lowercase for keyword in banter_keywords)
        
        # Determine agreement using same logic as JSON parsing
        if model_label == 'Normal':
            agrees = (llm_label == 'Normal')
        else:
            if is_banter:
                agrees = (llm_label == 'Normal')
            else:
                agrees = (llm_label == model_label or llm_label != 'Normal')
        
        return {
            'agrees': agrees,
            'llm_label': llm_label,
            'reasoning': response[:200],  # Use first 200 characters as reasoning
            'confidence': 0.5,  # Default confidence when parsing is uncertain
            'is_banter': is_banter
        }

