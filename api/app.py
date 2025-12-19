"""
FastAPI Application

This module provides a RESTful API for toxic content classification with conversational
analysis and optional LLM verification. The API accepts single messages or conversation
threads and returns comprehensive analysis including classification, banter detection,
and conflict resolution.

The API is built using FastAPI and provides:
- Single message analysis endpoint
- Conversation thread analysis endpoint
- Batch processing endpoint
- Health check endpoint
"""

import os
import sys
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path for module imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from conversational_analysis import ConversationAnalyzer
from llm_verification import LLMVerifier


# Pydantic models for request/response validation
class ConversationMessage(BaseModel):
    """
    Single message in a conversation thread.
    
    Attributes:
        user: User identifier (string, required)
        message: Message text content (string, required)
        timestamp: Optional timestamp for the message (string, optional)
    """
    user: str = Field(..., description="User identifier")
    message: str = Field(..., description="Message text")
    timestamp: Optional[str] = Field(None, description="Optional timestamp")


class AnalyzeRequest(BaseModel):
    """
    Request model for single message or conversation analysis.
    
    Either 'text' or 'conversation' must be provided. If both are provided,
    'conversation' takes precedence and 'text' is ignored.
    
    Attributes:
        text: Single message text (optional, required if conversation is None)
        conversation: List of conversation messages (optional, required if text is None)
    """
    text: Optional[str] = Field(None, description="Single message text")
    conversation: Optional[List[ConversationMessage]] = Field(None, description="Conversation thread")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "You're such an idiot!",
                "conversation": None
            }
        }


class BatchAnalyzeRequest(BaseModel):
    """
    Request model for batch analysis of multiple texts.
    
    Attributes:
        texts: List of text strings to analyze (required, non-empty)
    """
    texts: List[str] = Field(..., description="List of texts to analyze")


# Initialize FastAPI application
app = FastAPI(
    title="Toxic Content Classification API",
    description=(
        "API for classifying toxic content with conversational analysis and LLM verification. "
        "Supports single message analysis, conversation thread analysis, and batch processing."
    ),
    version="1.0.0"
)

# Global analyzer instances (initialized on application startup)
# These are loaded once when the API starts to avoid repeated model loading
conversation_analyzer: Optional[ConversationAnalyzer] = None
llm_verifier: Optional[LLMVerifier] = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize models and analyzers on application startup.
    
    This function is called automatically when the FastAPI application starts.
    It loads the XLM-RoBERTa classification model and optionally initializes
    the LLM verifier if a model path is provided via environment variable.
    
    The function reads configuration from environment variables:
    - MODEL_PATH: Path to XLM-RoBERTa model (default: './models/xlm-roberta-toxic-classifier')
    - LLM_MODEL_PATH: Path to LLM model for verification (optional)
    
    Raises:
        Exception: If Conversation Analyzer initialization fails (API cannot start without it)
    """
    global conversation_analyzer, llm_verifier
    
    # Get model path from environment variable or use default
    model_path = os.getenv('MODEL_PATH', './models/xlm-roberta-toxic-classifier')
    
    # Initialize Conversation Analyzer (required component)
    print("Initializing Conversation Analyzer...")
    try:
        conversation_analyzer = ConversationAnalyzer(model_path=model_path)
        print("Conversation Analyzer initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Conversation Analyzer: {e}")
        raise  # Cannot proceed without the main analyzer
    
    # Initialize LLM verifier (optional component)
    llm_model_path = os.getenv('LLM_MODEL_PATH', None)
    if llm_model_path:
        print("Initializing LLM Verifier...")
        try:
            llm_verifier = LLMVerifier(model_path=llm_model_path)
            print("LLM Verifier initialized successfully")
        except Exception as e:
            print(f"Failed to initialize LLM Verifier: {e}")
            print("Continuing without LLM verification...")
            llm_verifier = None
    else:
        print("LLM model path not provided. LLM verification disabled.")
        print("Set LLM_MODEL_PATH environment variable to enable.")


@app.get("/")
async def root():
    """
    Root endpoint providing API information.
    
    Returns basic API information including available endpoints and version.
    
    Returns:
        Dictionary with API metadata and endpoint descriptions
    """
    return {
        "message": "Toxic Content Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze single message or conversation",
            "/batch_analyze": "POST - Analyze multiple texts",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring API status.
    
    This endpoint reports the status of the API and its components,
    useful for health monitoring and load balancer checks.
    
    Returns:
        Dictionary containing:
        - 'status': Overall API status (always "healthy" if endpoint is reachable)
        - 'conversation_analyzer': Boolean indicating if analyzer is initialized
        - 'llm_verifier': Boolean indicating if LLM verifier is enabled and active
    """
    return {
        "status": "healthy",
        "conversation_analyzer": conversation_analyzer is not None,
        "llm_verifier": llm_verifier is not None and llm_verifier.enabled if llm_verifier else False
    }


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Analyze text or conversation for toxic content.
    
    Accepts either:
    - Single text message
    - Conversation thread with multiple messages
    
    Returns comprehensive analysis including:
    - Classification label and confidence
    - Conversational analysis (banter detection)
    - LLM verification (if enabled)
    - Final label decision
    """
    if conversation_analyzer is None:
        raise HTTPException(status_code=503, detail="Conversation analyzer not initialized")
    
    # Validate input
    if request.conversation is not None and len(request.conversation) == 0:
        raise HTTPException(status_code=400, detail="Conversation list cannot be empty")
    
    if not request.text and not request.conversation:
        raise HTTPException(status_code=400, detail="Either 'text' or 'conversation' must be provided")
    
    try:
        # Convert conversation Pydantic models to dictionary format if provided
        # This conversion is needed because the analyzer expects dict format
        conversation_messages_list = None
        if request.conversation:
            conversation_messages_list = [
                {
                    'user': msg.user,
                    'message': msg.message,
                    'timestamp': msg.timestamp
                }
                for msg in request.conversation
            ]
        
        # Get LLM verification first if available
        # Note: We need a preliminary classification to provide to the LLM
        llm_verification_result = None
        if llm_verifier and llm_verifier.enabled:
            # Get preliminary model classification for LLM verification
            # The LLM needs to know what the model predicted to provide verification
            preliminary_classification_result = conversation_analyzer.analyze(
                text=request.text,
                conversation=conversation_messages_list
            )
            
            # Extract target text for LLM verification
            # Use single text or last message in conversation
            target_text_for_llm = request.text or (
                conversation_messages_list[-1]['message'] if conversation_messages_list else ''
            )
            
            # Request LLM verification
            llm_verification_result = llm_verifier.verify(
                text=target_text_for_llm,
                conversation=conversation_messages_list,
                model_label=preliminary_classification_result['classification']['label'],
                model_confidence=preliminary_classification_result['classification']['confidence']
            )
        
        # Run full analysis with LLM verification for conflict resolution
        # The analyzer will use LLM result to resolve conflicts if available
        analysis_result = conversation_analyzer.analyze(
            text=request.text,
            conversation=conversation_messages_list,
            llm_result=llm_verification_result
        )
        
        # Ensure final_label is always set (safety check)
        # This should always be set by analyze(), but we verify for robustness
        if 'final_label' not in analysis_result:
            analysis_result['final_label'] = analysis_result.get('classification', {}).get('label', 'Normal')
        
        return analysis_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/batch_analyze")
async def batch_analyze(request: BatchAnalyzeRequest):
    """
    Analyze multiple texts in batch processing mode.
    
    This endpoint processes multiple texts in a single request, useful for
    bulk analysis scenarios. Each text is analyzed independently, and errors
    for individual texts are included in results rather than failing the batch.
    
    Args:
        request: BatchAnalyzeRequest containing list of texts to analyze
    
    Returns:
        Dictionary containing:
        - 'results': List of analysis results, one per input text
        - 'count': Number of texts processed
    
    Raises:
        HTTPException: 503 if analyzer not initialized, 500 if batch processing fails
    """
    if conversation_analyzer is None:
        raise HTTPException(status_code=503, detail="Conversation analyzer not initialized")
    
    try:
        batch_results = conversation_analyzer.batch_analyze(request.texts)
        return {
            "results": batch_results,
            "count": len(batch_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


if __name__ == "__main__":
    # Run the API server
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

