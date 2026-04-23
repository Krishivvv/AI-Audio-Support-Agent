"""
FastAPI Server for Audio Customer Support Agent

This module provides REST API endpoints for testing the audio support pipeline.
Students can use this server to test their implementations via HTTP requests.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
import base64
import logging
import os

from src.pipeline import AudioSupportPipeline, create_pipeline, PipelineConfig


class TextRequest(BaseModel):
    """Request model for text-based queries."""
    text: str
    parameters: Optional[Dict[str, Any]] = {}


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    components: Dict[str, bool]
    message: str


class TextResponse(BaseModel):
    """Response model for text queries."""
    response_text: str
    audio_available: bool
    processing_time_ms: int


class TranscriptData(BaseModel):
    """Transcript data for audio conversations."""
    user_input: str
    agent_response: str


class EnhancedAudioResponse(BaseModel):
    """Response model for audio queries with transcript data."""
    success: bool
    audio_response: str
    transcript: TranscriptData
    processing_time_ms: int


class EnhancedTextResponse(BaseModel):
    """Response model for text queries with processing time."""
    response_text: str
    processing_time_ms: int


app = FastAPI(
    title="Audio Customer Support Agent API",
    description="REST API for testing the STT -> LLM -> TTS pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[AudioSupportPipeline] = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    """
    TODO: Initialize the pipeline on server startup.
    
    Students should configure the pipeline with their API keys and settings.
    """
    global pipeline

    try:
        from dotenv import load_dotenv
        load_dotenv()

        logger.info("Starting Audio Support Agent API server...")

        placeholder_values = [
            "your_llm_api_key_here",
            "YOUR_REAL_OPENAI_API_KEY_HERE",
            "REPLACE_WITH_YOUR_OPENAI_API_KEY",
            "REPLACE_WITH_YOUR_GROQ_API_KEY",
            "",
        ]

        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")

        if groq_key and groq_key not in placeholder_values:
            llm_config = {
                "api_key": groq_key,
                "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                "temperature": 0.7,
            }
            logger.info(f"Using Groq LLM provider (model={llm_config['model']})")
        elif openai_key and openai_key not in placeholder_values:
            llm_config = {
                "api_key": openai_key,
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
            }
            logger.info("Using OpenAI LLM provider (model=gpt-3.5-turbo)")
        else:
            logger.error(
                "No LLM API key set. Provide GROQ_API_KEY or OPENAI_API_KEY in .env and restart."
            )
            return

        stt_config = {
            "model": "base"
        }

        tts_config = {
            "voice": "en-US-AriaNeural"
        }

        pipeline = await create_pipeline(stt_config, llm_config, tts_config)
        logger.info("Pipeline initialized and ready to serve requests!")

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup pipeline resources on server shutdown."""
    global pipeline
    
    if pipeline:
        logger.info("Shutting down pipeline...")
        await pipeline.cleanup()
        pipeline = None


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Audio Customer Support Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of all pipeline components.
    """
    global pipeline

    if not pipeline:
        return HealthResponse(
            status="unhealthy",
            components={
                "pipeline_initialized": False,
                "stt_ready": False,
                "llm_ready": False,
                "tts_ready": False
            },
            message="Pipeline not initialized. Check OPENAI_API_KEY in .env and restart server."
        )

    try:
        components = await pipeline.health_check()
        all_healthy = all(components.values())

        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            components=components,
            message="All components operational" if all_healthy else "Some components not ready — check server logs"
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="error",
            components={},
            message=f"Health check error: {str(e)}"
        )


@app.post("/chat/text", response_model=TextResponse)
async def chat_text(request: TextRequest):
    """
    Process text query through the LLM agent.
    
    This endpoint allows testing the LLM component without audio processing.
    """
    global pipeline

    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Check /health endpoint for details."
        )

    try:
        import time
        start_time = time.time()

        response_text, response_audio = await pipeline.process_text(
            request.text,
            **request.parameters
        )

        processing_time = int((time.time() - start_time) * 1000)

        return TextResponse(
            response_text=response_text,
            audio_available=len(response_audio) > 0,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Text processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/audio", response_model=EnhancedAudioResponse)
async def chat_audio(audio: UploadFile = File(...)):
    """
    TODO: Process audio query through the complete pipeline.
    
    This endpoint handles the full STT -> LLM -> TTS pipeline.
    
    Args:
        audio: Audio file upload (WAV, MP3, etc.)
        
    Returns:
        JSON response with base64 audio + transcript + processing time
    """
    global pipeline

    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Check /health endpoint for details."
        )

    try:
        audio_bytes = await audio.read()

        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file received")

        logger.info(f"Processing audio upload: {len(audio_bytes)} bytes, file: {audio.filename}")

        response_audio, transcript_data, processing_time_ms = await pipeline.process_audio_with_transcript(
            audio_bytes
        )

        encoded_audio = base64.b64encode(response_audio).decode("ascii")

        return EnhancedAudioResponse(
            success=True,
            audio_response=encoded_audio,
            transcript=TranscriptData(
                user_input=transcript_data.user_input,
                agent_response=transcript_data.agent_response,
            ),
            processing_time_ms=processing_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/audio/{text}")
async def text_to_audio(text: str):
    """
    TODO: Convert text to audio using TTS.
    
    Useful for testing TTS component independently.
    
    Args:
        text: Text to convert to speech
        
    Returns:
        Audio file as bytes
    """
    global pipeline

    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        if not pipeline.tts:
            raise HTTPException(status_code=503, detail="TTS component not available")

        audio_bytes = await pipeline.tts.synthesize(text)

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=tts_output.mp3"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/stt")
async def debug_stt(audio: UploadFile = File(...)):
    """
    TODO: Debug endpoint for testing STT component independently.
    
    Args:
        audio: Audio file to transcribe
        
    Returns:
        Transcription result
    """
    global pipeline

    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        audio_bytes = await audio.read()

        if not pipeline.stt:
            raise HTTPException(status_code=503, detail="STT component not available")

        transcription = await pipeline.stt.transcribe(audio_bytes)
        return {
            "transcription": transcription,
            "length_chars": len(transcription),
            "empty": len(transcription.strip()) == 0
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT debug failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # TODO: Students can modify these settings for development
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )